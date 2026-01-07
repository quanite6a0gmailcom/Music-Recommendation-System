from youtube_search import YoutubeSearch
from youtube_comment_downloader import YoutubeCommentDownloader, SORT_BY_POPULAR
import pandas as pd
import re
import time
import os
import csv

# --- CẤU HÌNH ---
INPUT_FILE = r'spotify_songs.csv/spotify_songs_final.csv'
OUTPUT_FILE = r"spotify_songs.csv/spotify_songs_final_comments_merged.csv"
MAX_COMMENTS = 50  # Số lượng comment muốn lấy mỗi bài để gộp

#Read input file
df = pd.read_csv(INPUT_FILE,encoding='utf-8-sig')
total_songs = len(df)
df['comment'] = ""

#Read output file
start_index = 0
file_exists = os.path.isfile(OUTPUT_FILE)

if file_exists:
    # Đếm số dòng đã có trong file output để biết cần chạy tiếp từ đâu
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        # Trừ 1 vì dòng đầu là header
        row_count = sum(1 for row in f) - 1
        start_index = row_count
        print(f"🔄 Detected old files. The {start_index} files have been run. Continuing...")
else:
    print("🚀 Start a completely new run....")

def get_video_id(keyword):
    """Tìm video ID từ từ khóa"""
    try:
        results = YoutubeSearch(keyword, max_results=1).to_dict()
        if results:
            return results[0]['id']
    except:
        return None
    return None

def clean_text(text):
    """
    Hàm làm sạch comment:
    1. Chuyển thành chữ thường.
    2. Xóa Link (http...).
    3. Xóa ký tự đặc biệt (icon, emoji...), chỉ giữ lại chữ và số.
    4. Xóa dấu xuống dòng (\n).
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Chuyển chữ thường
    text = text.lower()
    
    # 2. Xóa URL/Link
    text = re.sub(r'http\S+', '', text)
    
    # 3. Xóa các ký tự không phải là chữ (giữ lại tiếng Việt và số)
    # \w bao gồm [a-zA-Z0-9_] và các ký tự unicode tiếng Việt
    # Nếu muốn giữ dấu câu (.,?!), hãy bỏ dòng này
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # 4. Thay thế xuống dòng bằng khoảng trắng (QUAN TRỌNG ĐỂ THÀNH 1 DÒNG)
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # 5. Xóa khoảng trắng thừa (ví dụ: "  a   b " -> "a b")
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def process_comments(track_name,track_artist):
    downloader = YoutubeCommentDownloader()
    final_data = []

    search_query = f"{track_name} {track_artist}"
    print(f"🔎 Đang tìm: {search_query}...", end=" ")
    
    video_id = get_video_id(search_query)
    
    if video_id:
        try:
            # Lấy comment (Generator)
            generator = downloader.get_comments(video_id, sort_by=SORT_BY_POPULAR)
            
            raw_comments = []
            count = 0
            
            # Vòng lặp lấy từng comment
            for comment in generator:
                text = comment['text']
                
                # --- LÀM SẠCH NGAY LẬP TỨC ---
                cleaned_text = clean_text(text)
                
                if cleaned_text: # Nếu comment không rỗng sau khi clean
                    raw_comments.append(cleaned_text)
                    count += 1
                    
                if count >= MAX_COMMENTS:
                    break
            
            # --- GỘP TẤT CẢ THÀNH 1 DÒNG DUY NHẤT ---
            # Nối các comment bằng dấu chấm "." hoặc khoảng trắng " "
            merged_text = " . ".join(raw_comments)
            return merged_text
            print(f"✅ Đã gộp {count} comment.")
            
        except Exception as e:
            print(f"❌ Lỗi tải comment: {e}")
            return 'nomal'
    else:
        print("❌ Không tìm thấy video.")
        return 'nomal'
        
    # Nghỉ nhẹ để tránh chặn IP
    time.sleep(2)


# 3. Mở file Output ở chế độ 'a' (Append - Ghi nối tiếp)
# buffer=1: Ghi xuống ổ cứng ngay lập tức sau mỗi dòng (tránh mất điện mất dữ liệu)
with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8', buffering=1) as f:
    
    writer = csv.writer(f)
    
    # Nếu file mới tinh, ghi Header trước
    if not file_exists:
        writer.writerow(['track_artist','track_name','track_popularity','playlist_genre','energy','valence','acousticness','instrumentalness','speechiness','lyrics','comments'])

    # 4. Vòng lặp chính (Bắt đầu từ start_index)
    # df.iloc[start_index:] giúp cắt bỏ phần đã chạy rồi
    for index, row in df.iloc[start_index:].iterrows():
        
        track_name = row['track_name']
        track_artist = row['track_artist']
        track_popularity = row['track_popularity']
        playlist_genre = row['playlist_genre']
        energy = row['energy']
        valence = row['valence']
        acousticness = row['acousticness']
        instrumentalness = row['instrumentalness']
        speechiness = row['speechiness']
        lyrics = row['lyrics']
        term = f"{track_name} {track_artist}"
        
        try:
            # Tìm lời
            comments = process_comments(track_name,track_artist)
            
            # GHI NGAY LẬP TỨC XUỐNG FILE
            writer.writerow([track_name,track_artist,track_popularity,playlist_genre,energy,valence,acousticness,instrumentalness,speechiness,lyrics,comments])
            
            # In tiến độ cho đỡ sốt ruột
            # (index + 1) vì index bắt đầu từ 0
            print(f"[{index+1}/{total_songs}] ✅ Xong: {term}")
            
        except Exception as e:
            print(f"❌ Lỗi: {term}")
            writer.writerow([track_name,track_artist,track_popularity,playlist_genre,energy,valence,acousticness,instrumentalness,speechiness,lyrics, "normal"])
        
        # Ngủ nhẹ 0.5s để server không chặn (quan trọng với 1 triệu request)
        # Nếu mạng khỏe có thể giảm xuống 0.1
        time.sleep(0.3)

print("🎉 Đã hoàn thành toàn bộ!")