import urllib.parse
from pymongo import MongoClient

# --- 1. CẤU HÌNH KẾT NỐI ---
# Thay đổi đường dẫn nếu bạn dùng MongoDB Atlas
MONGO_URI = "mongodb://localhost:27017/" 
DB_NAME = "music_project_db"          # Tên Database của bạn
COLLECTION_NAME = "songs"    # Tên Collection bài hát

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    col = db[COLLECTION_NAME]
    print(f"✅ Đã kết nối thành công đến: {DB_NAME}.{COLLECTION_NAME}")
except Exception as e:
    print(f"❌ Lỗi kết nối MongoDB: {e}")
    exit()

# --- 2. HÀM XỬ LÝ LOGIC ---
def create_youtube_search_url(song_name, artist_name):
    """
    Tạo link tìm kiếm YouTube từ tên bài hát và ca sĩ.
    """
    if not song_name: 
        return None
        
    # Xử lý nếu thiếu tên ca sĩ
    if not artist_name:
        artist_name = ""
        
    # Tạo chuỗi query: "Tên bài + Tên ca sĩ"
    query = f"{song_name} {artist_name}"
    
    # Mã hóa URL (Ví dụ: dấu cách thành %20 hoặc +)
    encoded_query = urllib.parse.quote_plus(query)
    
    # Trả về link
    return f"https://www.youtube.com/results?search_query={encoded_query}"

# --- 3. QUY TRÌNH ĐỌC VÀ CẬP NHẬT (BATCH UPDATE) ---
def batch_update_links():
    # Lấy tất cả bài hát
    # Mẹo: Chỉ lấy các trường cần thiết để tiết kiệm RAM (id, track_name, artist_name)
    cursor = col.find({}, {"track_name": 1, "track_artist": 1})
    
    total_docs = col.count_documents({})
    print(f"🔄 Bắt đầu xử lý {total_docs} bài hát...")
    
    count = 0
    updated_count = 0
    
    for doc in cursor:
        try:
            song_id = doc['_id']
            name = doc.get('track_name', '')
            artist = doc.get('track_artist', '')
            
            # 1. Gọi hàm tạo link
            search_link = create_youtube_search_url(name, artist)
            
            if search_link:
                # 2. Thực hiện Update vào MongoDB
                # Dùng $set để chỉ cập nhật trường 'youtube_search_link', giữ nguyên các trường khác
                col.update_one(
                    {"_id": song_id},
                    {"$set": {"youtube_search_link": search_link}}
                )
                updated_count += 1
                
            count += 1
            
            # In tiến độ mỗi 100 bài
            if count % 100 == 0:
                print(f"⏳ Đã xử lý: {count}/{total_docs} bài...")
                
        except Exception as e:
            print(f"⚠️ Lỗi tại ID {doc.get('_id')}: {e}")

    print("------------------------------------------------")
    print(f"🎉 HOÀN TẤT! Đã cập nhật link cho {updated_count} bài hát.")

# --- 4. CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # Hỏi xác nhận trước khi chạy để tránh update nhầm
    confirm = input("Bạn có chắc muốn cập nhật Database không? (y/n): ")
    if confirm.lower() == 'y':
        batch_update_links()
    else:
        print("Đã hủy bỏ.")