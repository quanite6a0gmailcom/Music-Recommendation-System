import pandas as pd
from pymongo import MongoClient

# 1. Đọc dữ liệu từ CSV
df = pd.read_csv("C:\\Music-Recommendation-System\\store data\\final data\\spotify_songs_final_comments_merged.csv")

# 2. Chuẩn bị List để chứa dữ liệu đã biến đổi
data_to_insert = []

print("⏳ Đang xử lý dữ liệu...")
for index, row in df.iterrows():
    # TẠO CẤU TRÚC LỒNG NHAU TẠI ĐÂY
    document = {
        "track_name": row['track_name'],
        "track_artist": row['track_artist'],
        "track_popularity" :row['track_popularity'],
        "playlist_genre": row['playlist_genre'],
        "lyrics": row['lyrics'],
        "faiss_id": int(row['faiss_id']),
        "cluster_id": int(row['cluster_id']),
        
        # Gom các cột chỉ số vào key "features"
        "features": {
          # Chuyển về float cho chuẩn số
            "energy": float(row['energy']),
            "valence": float(row['valence']),
            "acousticness": float(row['acousticness']),
            "instrumentalness": float(row['instrumentalness']),
            "speechiness": float(row['speechiness']),
            
        }
    }
    data_to_insert.append(document)

# 3. Lưu vào MongoDB (insert_many cho nhanh)
client = MongoClient("mongodb://localhost:27017/")
col = client["music_project_db"]["songs"]

# Xóa dữ liệu cũ làm sạch (tùy chọn)
# col.drop()

if data_to_insert:
    col.insert_many(data_to_insert)
    print(f"✅ Đã import thành công {len(data_to_insert)} bài hát theo cấu trúc mới!")