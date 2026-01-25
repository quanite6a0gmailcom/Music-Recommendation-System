import faiss
import numpy as np
from pymongo import MongoClient
import google.generativeai as genai
from sentence_transformers import SentenceTransformer




API_KEY = 'AIzaSyBCqsw-TU9NqhJxr-P1MFNVn0PAHjyQ-uI'
client = MongoClient("mongodb://localhost:27017/")
col = client["music_project_db"]["songs"]

# Config google api
genai.configure(api_key=API_KEY)

vectors = np.load('C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_tracknames.npy')
index = faiss.IndexFlatL2(vectors.shape[1])

index.add(vectors)

all_songs_metadata = list(col.find().sort("faiss_id",1))


def merge_song_list(songlist,songlist1):
    seen_ids = set()
    final_results = []

    # Duyệt qua danh sách Regex trước (Ưu tiên chính xác)
    for s in songlist:
        s_id = str(s['_id']) # Hoặc dùng s['faiss_id']
        if s_id not in seen_ids:
            final_results.append(s)
            seen_ids.add(s_id)

    # Duyệt tiếp danh sách FAISS (Bỏ qua bài nào đã có rồi)
    for s in songlist1:
        s_id = str(s['_id'])
        if s_id not in seen_ids:
            final_results.append(s)
            seen_ids.add(s_id)

    return final_results

def search_by_name_regex(keyword):
    # $regex: Tìm chuỗi khớp
    # $options: 'i' (case-insensitive) -> Không phân biệt hoa thường
    query = {
        "track_name": {"$regex": keyword, "$options": "i"}
    }
    
    # Chỉ lấy 10 kết quả đầu tiên
    results = list(col.find(query).limit(20))
    return results

def search_full_text(keyword):
    # Cú pháp $text - $search
    query = {"$text": {"$search": keyword}}
    
    # Sắp xếp theo độ khớp (score)
    results = col.find(
        query, 
        {"score": {"$meta": "textScore"}}
    ).sort([("score", {"$meta": "textScore"})]).limit(20)
    
    return list(results)

def search_semantic(keyword,limit):
    # B1: Vector hóa từ khóa

    # result = genai.embed_content(
    #             model="models/text-embedding-004",
    #             content=keyword,
    #             task_type="retrieval_query"
    #         )
    # query_vec = np.array([result['embedding']]).astype('float32')

    # 1. Tải model về (Lần đầu chạy sẽ hơi lâu vì phải tải khoảng 80MB)
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    query_vec = model.encode(keyword).reshape(1, -1)
    
    # B2: Tìm trong FAISS
    distances, indices = index.search(query_vec, k=limit)
    
    # B3: Lấy thông tin bài hát từ MongoDB dựa trên ID tìm được
    found_songs = []
    for idx in indices[0]:
        # idx chính là faiss_id
        # Cách nhanh nhất là lấy từ list đã load sẵn vào RAM
        song = all_songs_metadata[idx]
        found_songs.append(song)
        
    return found_songs


def search(name_song):
    # name_song = input("Input name of song that you want to listen:")

    songs = search_by_name_regex(name_song)

    if (len(songs) < 20):
        song1s = search_full_text(name_song)
        songs = merge_song_list(songs,song1s)
    #     if len(songs) < 20:
    #         song2s  = search_semantic(name_song,20-len(songs))
    #         songs = merge_song_list(songs,song2s)
    
    return songs

print(search("Dung lam"))
    # # songs = search_semantic(name_song)
    # for song in songs:
    #     print(song['track_name'] + " " + song['track_artist'])

