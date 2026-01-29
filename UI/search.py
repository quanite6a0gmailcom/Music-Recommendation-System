import faiss
import numpy as np
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

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

class search:
    def __init__(self,db,index,all_songs_metadata):
        self.db = db 
        self.index = index
        self.all_songs_metadata = all_songs_metadata

    def search_by_name_regex(self,keyword):
        # $regex: Tìm chuỗi khớp
        # $options: 'i' (case-insensitive) -> Không phân biệt hoa thường
        query = {
            "track_name": {"$regex": keyword, "$options": "i"}
        }
        
        # Chỉ lấy 10 kết quả đầu tiên
        results = list(self.db.find(query).limit(20))
        return results
    
    def search_full_text(self,keyword):
        # Cú pháp $text - $search
        query = {"$text": {"$search": keyword}}
        
        # Sắp xếp theo độ khớp (score)
        results = self.db.find(
            query, 
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(20)
        
        return list(results)
    
    def search_semantic(self,keyword,limit):
        # B1: Vector hóa từ khóa
        # API_KEY = 'key'
        # # Config google api
        # genai.configure(api_key=API_KEY)
        # result = genai.embed_content(
        #             model="models/text-embedding-004",
        #             content=keyword,
        #             task_type="retrieval_query"
        #         )
        # query_vec = np.array([result['embedding']]).astype('float32')

        # 1. Tải model về (Lần đầu chạy sẽ hơi lâu vì phải tải khoảng 80MB)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_vec = model.encode(keyword).reshape(1, -1)
        
        # B2: Tìm trong FAISS
        distances, indices = self.index.search(query_vec, k=limit)
        
        # B3: Lấy thông tin bài hát từ MongoDB dựa trên ID tìm được
        found_songs = []
        for idx in indices[0]:
            # idx chính là faiss_id
            # Cách nhanh nhất là lấy từ list đã load sẵn vào RAM
            song = self.all_songs_metadata[idx]
            found_songs.append(song)
            
        return found_songs
    
    def search(self,name_song):
        # name_song = input("Input name of song that you want to listen:")

        songs = self.search_by_name_regex(name_song)

        if (len(songs) < 20):
            song1s = self.search_full_text(name_song)
            songs = merge_song_list(songs,song1s)
            if len(songs) < 20:
                song2s  = self.search_semantic(name_song,20-len(songs))
                songs = merge_song_list(songs,song2s)
        
        return songs