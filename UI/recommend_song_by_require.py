import google.generativeai as genai
import os
import json
import re 
import pymongo
import faiss
import numpy as np

class recommend_song_by_require:
    def __init__(self,db_songs,audio_vectors,VALID_GENRES,GENRE_MAPPING):
        self.db_songs = db_songs
        self.audio_vectors = audio_vectors
        self.VALID_GENRES = VALID_GENRES
        self.GENRE_MAPPING = GENRE_MAPPING
    
    def get_music_params_from_llm(self,user_input):
        API_KEY = 'key'

        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-3-flash-preview')
        
        prompt = f"""
            Bạn là chuyên gia tư vấn âm nhạc. Dựa trên tâm trạng/yêu cầu: "{user_input}".

            Hãy trả về JSON gồm:
            1. "target_genre": Chọn 1 hoặc 2 thể loại phù hợp nhất TỪ DANH SÁCH SAU ĐÂY (Tuyệt đối không bịa ra thể loại khác):
            {json.dumps(self.VALID_GENRES, ensure_ascii=False)}

            2. "search_keywords": (List) Các từ khóa cụ thể để tìm trong database (Ví dụ: nếu chọn 'C-Pop', từ khóa có thể là 'mandopop', 'taiwanese pop').

            3. "audio_features": (Object) Các chỉ số mục tiêu (energy, valence,acousticness,instrumentalnesss,speechiness) từ 0.0 - 1.0.

            Ví dụ Output mong muốn:
            {{
            "target_genre": ["V-Pop", "Ballad & Lofi"],
            "search_keywords": ["vietnam indie", "lo-fi", "v-pop"],
            "audio_features": {{ "energy": 0.3, "valence": 0.4 }}
            }}
            """
        
        response = model.generate_content(prompt)
        clean_json = response.text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(clean_json)
    
    def build_gemini_query(self,gemini_output, buffer=0.15):
        """
        gemini_output: JSON nhận được từ Gemini (có genres và audio_features)
        buffer: Độ lệch cho phép (mặc định +/- 0.15)
        """
        
        final_query = {}
        criteria_list = [] # Dùng cho $and

        # --- BƯỚC 1: XỬ LÝ GENRE (THỂ LOẠI) ---
        target_genres = gemini_output.get("target_genre", [])
        
        # Gom tất cả keyword cần tìm
        keywords = []
        for g in target_genres:
            if g in self.GENRE_MAPPING:
                keywords.extend(self.GENRE_MAPPING[g])
        
        # Nếu có keyword, tạo query Regex
        # Logic: Tìm bài hát mà trường 'genres' chứa ÍT NHẤT 1 trong các từ khóa
        if keywords:
            regex_list = [re.compile(re.escape(k), re.IGNORECASE) for k in keywords]
            criteria_list.append({"playlist_genre": {"$in": regex_list}})

        # --- BƯỚC 2: XỬ LÝ AUDIO FEATURES (ENERGY, VALENCE...) ---
        features = gemini_output.get("audio_features", {})
        
        for feature_name, value in features.items():
            # Chỉ xử lý nếu giá trị là số hợp lệ
            if isinstance(value, (int, float)):
                # Tạo khoảng min-max, chặn đầu đuôi không quá 0.0 và 1.0
                min_val = max(0.0, value - buffer)
                max_val = min(1.0, value + buffer)
                key_name = f"features.{feature_name}"
                # Thêm điều kiện vào list
                criteria_list.append({
                    key_name: {"$gte": min_val, "$lte": max_val}
                })

        # --- BƯỚC 3: TỔNG HỢP ---
        if criteria_list:
            final_query = {"$and": criteria_list}
        else:
            final_query = {} # Tìm tất cả nếu không có điều kiện

        return final_query
    
    def search(self,input_text):
        gemini_response = self.get_music_params_from_llm(input_text)
        energy = gemini_response['audio_features']['energy']
        valence = gemini_response['audio_features']['valence']
        acousticness = gemini_response['audio_features']['acousticness']
        instrumentalness = gemini_response['audio_features']['instrumentalness']
        speechiness = gemini_response['audio_features']['speechiness']

        audio_vec = [energy,valence,acousticness,instrumentalness,speechiness]

        query = self.build_gemini_query(gemini_response)
        results = list(self.db_songs.find(query).limit(20))
        if len(results) < 30:
            d = self.audio_vectors.shape[1]
            index = faiss.IndexFlatL2(d)
            index.add(self.audio_vectors)

            audio_vec = np.array(audio_vec,dtype='float32')
            query_vector = audio_vec.reshape(1,-1)
            distances,local_indices = index.search(query_vector,30-len(results))

            for local_idx in np.random.permutation(local_indices[0]):
                query = {"faiss_id":int(local_idx)}
                found_song = list(self.db_songs.find(query))[0]
                results.append(found_song)

        # for song in results:
        #     print(f"🎵 {song['track_name']} - Energy: {song['features'].get('energy')} - Genres: {song.get('playlist_genre')}")

        # print(len(results))
        print(gemini_response)
        seen = set()
        # Vừa thêm vào list kết quả, vừa thêm vào set seen để đánh dấu
        unique_results = [
            x for x in results 
            if str(x.get('_id')) not in seen and not seen.add(str(x.get('_id')))
        ]
        return unique_results

