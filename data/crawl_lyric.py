import pandas as pd
import csv
import os
import syncedlyrics
import re
import time

# --------------------CONFIG----------------
INPUT_FILE = r'spotify_songs.csv\spotify_songs.csv'
OUTPUT_FILE = r'spotify_songs.csv\spotify_songs_processed.csv'

#Read input file
cols_to_use = ['track_artist','track_name','track_popularity','track_album_name','playlist_genre','energy','valence','acousticness','instrumentalness','speechiness','key','loudness','liveness','mode','tempo','danceability']
df = pd.read_csv(r'spotify_songs.csv\spotify_songs.csv',usecols=cols_to_use,encoding="utf8")
total_songs = len(df)
df['lyrics'] = " "


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

def crawl_lyrics(track_name, artist_name):
    # 1. Tạo từ khóa tìm kiếm
    keyword = f"{track_name} {artist_name}"
    print(f"Searching: {keyword}...")
    
    # 2. Gọi thư viện
    lrc = syncedlyrics.search(keyword)
    
    if lrc:
        # 3. Dùng Regex để xóa tất cả những gì nằm trong dấu ngoặc vuông []
        # r'\[.*?\]' nghĩa là tìm mọi thứ bắt đầu bằng [ và kết thúc bằng ]
        plain_text = re.sub(r'\[.*?\]', '', lrc)
        
        # Xóa các dòng trống dư thừa
        plain_text = "\n".join([line.strip() for line in plain_text.split('\n') if line.strip()])
        plain_text = plain_text.strip().replace('\n', ' ')
        return plain_text
    else:
        return None

# 3. Mở file Output ở chế độ 'a' (Append - Ghi nối tiếp)
# buffer=1: Ghi xuống ổ cứng ngay lập tức sau mỗi dòng (tránh mất điện mất dữ liệu)
with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8', buffering=1) as f:
    
    writer = csv.writer(f)
    
    # Nếu file mới tinh, ghi Header trước
    if not file_exists:
        writer.writerow(['track_artist','track_name','track_popularity','track_album_name','playlist_genre','energy','valence','acousticness','instrumentalness','speechiness','key','loudness','liveness','mode','tempo','danceability','lyrics'])

    # 4. Vòng lặp chính (Bắt đầu từ start_index)
    # df.iloc[start_index:] giúp cắt bỏ phần đã chạy rồi
    for index, row in df.iloc[start_index:].iterrows():
        cols_to_use = ['track_artist','track_name','track_popularity','track_album_name','playlist_genre','energy','valence','acousticness','instrumentalness','speechiness','key','loudness','liveness','mode','tempo','danceability']
        
        track_name = row['track_name']
        track_artist = row['track_artist']
        track_popularity = row['track_popularity']
        track_album_name = row['track_album_name']
        playlist_genre = row['playlist_genre']
        energy = row['energy']
        valence = row['valence']
        acousticness = row['acousticness']
        instrumentalness = row['instrumentalness']
        speechiness = row['speechiness']
        key = row['key']
        loudness = row['loudness']
        liveness = row['liveness']
        mode = row['mode']
        tempo = row['tempo']
        danceability = row['danceability']

        term = f"{track_name} {track_artist}"
        
        try:
            # Tìm lời
            lyrics = crawl_lyrics(track_name,track_artist)
            
            # GHI NGAY LẬP TỨC XUỐNG FILE
            writer.writerow([track_name,track_artist,track_popularity,track_album_name,playlist_genre,energy,valence,acousticness,instrumentalness,speechiness,key,loudness,liveness,mode,tempo,danceability,lyrics])
            
            # In tiến độ cho đỡ sốt ruột
            # (index + 1) vì index bắt đầu từ 0
            print(f"[{index+1}/{total_songs}] ✅ Xong: {term}")
            
        except Exception as e:
            print(f"❌ Lỗi: {term}")
            writer.writerow([track_name,track_artist,track_popularity,track_album_name,playlist_genre,energy,valence,acousticness,instrumentalness,speechiness,key,loudness,liveness,mode,tempo,danceability, "Error"])
        
        # Ngủ nhẹ 0.5s để server không chặn (quan trọng với 1 triệu request)
        # Nếu mạng khỏe có thể giảm xuống 0.1
        time.sleep(0.3)

print("🎉 Đã hoàn thành toàn bộ!")