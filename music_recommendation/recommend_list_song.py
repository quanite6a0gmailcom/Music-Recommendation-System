import google.generativeai as genai
import pymongo
from bson import ObjectId
import faiss
from recommend_song import *
import json


API_KEY = 'key'

genai.configure(api_key=API_KEY)
def merge_song_list(songlist,songlist1):
    # --- THÊM 2 DÒNG NÀY ---
    if songlist is None: songlist = []
    if songlist1 is None: songlist1 = []
    # -----------------------
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

def search_within_cluster_hybrid(song_id, k=5):
    # ---------------------------------------------------------
    # BƯỚC 1: LẤY THÔNG TIN BÀI GỐC & DANH SÁCH ID CÙNG CỤM
    # ---------------------------------------------------------
    seed_song = col.find_one({"_id": ObjectId(song_id)})
    
    if not seed_song:
        print(f"❌ Không tìm thấy bài: {song_id}")
        return

    cluster_id = seed_song.get('cluster_id')
    seed_faiss_id = seed_song.get('faiss_id')
    
    print(f"🎵 Bài gốc: {seed_song['track_name']} (Cluster: {cluster_id})")

    # Lấy danh sách các bài trong cùng cụm (Lấy cả faiss_id và thông tin hiển thị)
    # Sort theo faiss_id để khớp thứ tự khi map vector
    cluster_songs_cursor = col.find(
        {"cluster_id": cluster_id},
        {"faiss_id": 1, "track_name": 1, "track_artist": 1, "_id": 1,"playlist_genre":1}
    ).sort("faiss_id", 1)
    
    cluster_songs = list(cluster_songs_cursor)
    
    if len(cluster_songs) < k:
        return cluster_songs
    
    # Tách riêng list faiss_id ra để đi cắt vector
    # Ví dụ: cluster_indices = [0, 5, 12, 99...]
    cluster_indices = [s['faiss_id'] for s in cluster_songs]
    num_songs_in_cluster = len(cluster_indices)
    subset_vectors_audio = all_vectors_audio[cluster_indices]
    subset_vectors_lyrics = all_vectors_lyrics[cluster_indices]
    subset_vectors_comments = all_vectors_cmt[cluster_indices]

    # d = subset_vectors_audio.shape[1]
    # mini_index_audio = faiss.IndexFlatL2(d)
    # mini_index_audio.add(subset_vectors_audio)

    # d = subset_vectors_lyrics.shape[1]
    # mini_index_lyric = faiss.IndexFlatL2(d)
    # mini_index_lyric.add(subset_vectors_lyrics)

    # Lấy vector của bài gốc để query
    query_vector_audio = all_vectors_audio[seed_faiss_id].reshape(1, -1)
    query_vector_lyric = all_vectors_lyrics[seed_faiss_id].reshape(1, -1)
    query_vector_comment = all_vectors_cmt[seed_faiss_id].reshape(1,-1)
    
    # # Tìm kiếm (Kết quả trả về là index TRONG TẬP CON, không phải index gốc)
    # distances_audio, local_indices_audio = mini_index_audio.search(query_vector_audio, num_songs_in_cluster)
    # distances_lyric, local_indices_lyric = mini_index_lyric.search(query_vector_lyric, num_songs_in_cluster)

    dis_matrix_audio = euclidean_distances(query_vector_audio,subset_vectors_audio)
    dis_matrix_lyrics = cosine_distances(query_vector_lyric,subset_vectors_lyrics)
    dis_matrix_cmt = cosine_distances(query_vector_comment,subset_vectors_comments)


    hybrid_distance = 0.5 * dis_matrix_audio[0] + 0.25 * dis_matrix_lyrics[0] + 0.25 * dis_matrix_cmt[0]
    local_indices = heapq.nsmallest(10,range(len(hybrid_distance)),key=hybrid_distance.__getitem__)
    local_indices = [cluster_indices[i] for i in local_indices]
    local_indices = local_indices[1:]
    songs = []
    for song in cluster_songs:
        if song['faiss_id'] in local_indices:
            songs.append(song)

    return songs,cluster_id,seed_faiss_id


def search_in_neighborcluster_hybrid(cluster_id,seed_faiss_id, k=5):
    cluster_songs_cursor = col.find(
        {"cluster_id": cluster_id},
        {"faiss_id": 1, "track_name": 1, "track_artist": 1, "_id": 1,"playlist_genre":1}
    ).sort("faiss_id", 1)
    
    cluster_songs = list(cluster_songs_cursor)
    
    if len(cluster_songs) < k:
        return cluster_songs

    # Tách riêng list faiss_id ra để đi cắt vector
    # Ví dụ: cluster_indices = [0, 5, 12, 99...]
    cluster_indices = [s['faiss_id'] for s in cluster_songs]
    num_songs_in_cluster = len(cluster_indices)
    subset_vectors_audio = all_vectors_audio[cluster_indices]
    subset_vectors_lyrics = all_vectors_lyrics[cluster_indices]
    subset_vectors_comments = all_vectors_cmt[cluster_indices]

    # d = subset_vectors_audio.shape[1]
    # mini_index_audio = faiss.IndexFlatL2(d)
    # mini_index_audio.add(subset_vectors_audio)

    # d = subset_vectors_lyrics.shape[1]
    # mini_index_lyric = faiss.IndexFlatL2(d)
    # mini_index_lyric.add(subset_vectors_lyrics)

    # Lấy vector của bài gốc để query
    query_vector_audio = all_vectors_audio[seed_faiss_id].reshape(1, -1)
    query_vector_lyric = all_vectors_lyrics[seed_faiss_id].reshape(1, -1)
    query_vector_comment = all_vectors_cmt[seed_faiss_id].reshape(1,-1)
    
    
    # # Tìm kiếm (Kết quả trả về là index TRONG TẬP CON, không phải index gốc)
    # distances_audio, local_indices_audio = mini_index_audio.search(query_vector_audio, num_songs_in_cluster)
    # distances_lyric, local_indices_lyric = mini_index_lyric.search(query_vector_lyric, num_songs_in_cluster)

    dis_matrix_audio = euclidean_distances(query_vector_audio,subset_vectors_audio)
    dis_matrix_lyrics = cosine_distances(query_vector_lyric,subset_vectors_lyrics)
    dis_matrix_cmt = cosine_distances(query_vector_comment,subset_vectors_comments)


    hybrid_distance = 0.5 * dis_matrix_audio[0] + 0.25 * dis_matrix_lyrics[0] + 0.25 * dis_matrix_cmt[0]
    local_indices = heapq.nsmallest(10,range(len(hybrid_distance)),key=hybrid_distance.__getitem__)
    local_indices = [cluster_indices[i] for i in local_indices]
    local_indices = local_indices[1:]
    songs = []
    for song in cluster_songs:
        if song['faiss_id'] in local_indices:
            songs.append(song)

    return songs

def recommend_hybrid(song_name):
    recommend_list = []
    count = 0
    cluster_songs,current_cluster_id,seed_faiss_id = search_within_cluster_hybrid(song_name,k=20)
    relation = clus.find_one({"cluster_id":current_cluster_id})

    random.shuffle(cluster_songs)
    
    for song in cluster_songs[:11]:  
        recommend_list.append(song)
    
    if not relation:
        return recommend_list
    
    neighbors = relation["nearest_clusters"]
    for neighbor in neighbors:
        neighbor_id = neighbor["cluster_id"]

        cluster_songs = search_in_neighborcluster_hybrid(neighbor_id,seed_faiss_id,k=20)
        count = count + 1
        if count >2:
            break
        found = 0
        random.shuffle(cluster_songs)
        for song in cluster_songs:  
            recommend_list.append(song)
            if found > 5:
                break
            found += 1

   

    return recommend_list


def get_top_cluster(client,col):
    """
    Tìm Cluster ID mà người dùng nghe nhiều nhất dựa trên tổng play_count.
    """

    
    pipeline = [
        # Bước 1: Lọc theo user hiện tại (Nếu bạn muốn tính riêng cho 1 user)
        # {"$match": {"user_id": "demo_user"}}, 
        
        # Bước 2: Gom nhóm theo cluster_id và tính tổng lượt nghe
        {
            "$group": {
                "_id": "$cluster_id",              # Group by Cluster
                "total_plays": {"$sum": "$play_count"} # Sum(play_count)
            }
        },
        
        # Bước 3: Sắp xếp giảm dần theo tổng lượt nghe
        {"$sort": {"total_plays": -1}},
        
        # Bước 4: Lấy 1 kết quả cao nhất
        {"$limit": 1}
    ]
    
    # Thực thi truy vấn
    result = list(col.aggregate(pipeline))
    
    if result:
        top_cluster = result[0]
        cluster_id = top_cluster['_id']
        plays = top_cluster['total_plays']
        
        print(f"🏆 Cluster được nghe nhiều nhất là: {cluster_id} (Tổng {plays} lượt nghe)")
        return cluster_id
    else:
        print("⚠️ Chưa có dữ liệu tương tác nào.")
        return None

# --- CÁCH SỬ DỤNG ---
# --- CẤU HÌNH ---
results = []
client = pymongo.MongoClient("mongodb://localhost:27017/")
his = client["music_project_db"]["user_history"]
top_cluster_id = get_top_cluster(client,his)
songs = list(his.find({"cluster_id": top_cluster_id}))
for song in songs:
    result = col.find_one({"_id": ObjectId(song['song_id'])})
    results.append(result)
    result = recommend_hybrid(song['song_id'])
    results = merge_song_list(results,result)

seen = set()
# Vừa thêm vào list kết quả, vừa thêm vào set seen để đánh dấu
unique_results = [
    x for x in results 
    if str(x.get('_id')) not in seen and not seen.add(str(x.get('_id')))
]

PLAYLIST_NAMING_PROMPT = """
Bạn là một chuyên gia tuyển chọn âm nhạc (Music Curator). 
Dựa trên danh sách các bài hát dưới đây, hãy phân tích dòng nhạc và tâm trạng chung (mood).

Yêu cầu output:
Hãy trả về kết quả dưới dạng JSON thuần túy (không có markdown ```json) với cấu trúc sau:
{{
    "playlist_name": "Tên Playlist Ngắn Gọn (Tiếng Việt, < 10 từ)",
    "description": "Một câu mô tả ngắn (Slogan) cực chất cho playlist này",
    "mood_tags": ["tag1", "tag2", "tag3"]
}}

Danh sách bài hát input:
{songs}
"""
songs_text = ""
for i in range(10):
    song = "-" + unique_results[i]['track_name'] + "("+unique_results[i]['track_artist']+")" + "-"
    songs_text = songs_text + song
final_prompt = PLAYLIST_NAMING_PROMPT.format(songs=songs_text)

try:
    # 3. Gọi Gemini API
    API_KEY = 'AIzaSyBOPwRxWTN5ohEm39yvm4DlS0fPk0Rb6W4'

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-3-flash-preview') # Hoặc model bạn đang dùng
    response = model.generate_content(final_prompt)
    
    # 4. Xử lý kết quả trả về (Parse JSON)
    result_text = response.text.strip()
    
    # Đôi khi AI vẫn trả về ```json, cần xóa đi để parse
    if result_text.startswith("```"):
        result_text = result_text.replace("```json", "").replace("```", "")
        
    data = json.loads(result_text)

    
except Exception as e:
    print(f"Lỗi AI: {e}")
    # Trả về kết quả mặc định nếu lỗi
    data =  {
        "playlist_name": "Playlist Của Tôi",
        "description": "Danh sách nhạc tuyển chọn",
        "mood_tags": ["Mix"]
    }

print(data['playlist_name'])