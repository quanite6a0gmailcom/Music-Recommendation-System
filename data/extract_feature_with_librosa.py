from extract_feature import *
from dowload_song import *
import pandas as pd
import csv
import os

INPUT_FILE = r"spotify_songs.csv/clean_apple_music_trending.csv"
OUTPUT_FILE = r"spotify_songs.csv/clean_apple_music_trending_features_librosa.csv"

# --- Read input CSV ---
cols_to_use = ['track_name','artist_name','popularity']
df_songs = pd.read_csv(INPUT_FILE, usecols=cols_to_use,encoding='utf-8')
total_songs = len(df_songs)

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

# 3. Mở file Output ở chế độ 'a' (Append - Ghi nối tiếp)
with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8', buffering=1) as f:
    
    writer = csv.writer(f)
    
    # Nếu file mới tinh, ghi Header trước
    if not file_exists:
        writer.writerow(['track_name','track_artist','bpm', 'rms_mean', 'rms_var', 'zcr_mean', 'zcr_var', 'centroid_mean', 'centroid_var', 'rolloff_mean', 'rolloff_var', 'flatness_mean', 'flatness_var', 'mfcc_mean_0', 'mfcc_var_0', 'mfcc_mean_1', 'mfcc_var_1', 'mfcc_mean_2', 'mfcc_var_2', 'mfcc_mean_3', 'mfcc_var_3', 'mfcc_mean_4', 'mfcc_var_4', 'mfcc_mean_5', 'mfcc_var_5', 'mfcc_mean_6', 'mfcc_var_6', 'mfcc_mean_7', 'mfcc_var_7', 'mfcc_mean_8', 'mfcc_var_8', 'mfcc_mean_9', 'mfcc_var_9', 'mfcc_mean_10', 'mfcc_var_10', 'mfcc_mean_11', 'mfcc_var_11', 'mfcc_mean_12', 'mfcc_var_12', 'chroma_mean', 'chroma_var', 'contrast_mean', 'contrast_var'])

    # 4. Vòng lặp chính (Bắt đầu từ start_index)
    # df.iloc[start_index:] giúp cắt bỏ phần đã chạy rồi
    for index, row in df_songs.iloc[start_index:].iterrows():
        cols_to_use = ['track_artist','track_name','track_popularity','track_album_name','playlist_genre','energy','valence','acousticness','instrumentalness','speechiness','key','loudness','liveness','mode','tempo','danceability']
        
        track_name = row['track_name']
        track_artist = row['artist_name']
        track_popularity = row['popularity']

        term = f"{track_name} {track_artist}"
        
        try:
            # Tìm lời
            path = download_clean_start(track_name,track_artist)

            file_mp3 = "dataset_audio/audio.mp3" # Đường dẫn file của bạn

            # 1. Lấy dữ liệu
            vector = extract_librosa_features(file_mp3)
            bpm = vector[0] 
            rms_mean = vector[1] 
            rms_var = vector[2] 
            zcr_mean = vector[3] 
            zcr_var = vector[4] 
            centroid_mean = vector[5] 
            centroid_var = vector[6] 
            rolloff_mean = vector[7] 
            rolloff_var = vector[8] 
            flatness_mean = vector[9] 
            flatness_var = vector[10] 
            mfcc_mean_0 = vector[11] 
            mfcc_var_0 = vector[12] 
            mfcc_mean_1 = vector[13] 
            mfcc_var_1 = vector[14] 
            mfcc_mean_2 = vector[15] 
            mfcc_var_2 = vector[16] 
            mfcc_mean_3 = vector[17] 
            mfcc_var_3 = vector[18] 
            mfcc_mean_4 = vector[19] 
            mfcc_var_4 = vector[20] 
            mfcc_mean_5 = vector[21] 
            mfcc_var_5 = vector[22] 
            mfcc_mean_6 = vector[23] 
            mfcc_var_6 = vector[24] 
            mfcc_mean_7 = vector[25]
            mfcc_var_7 = vector[26]
            mfcc_mean_8 = vector[27]
            mfcc_var_8 = vector[28]
            mfcc_mean_9 = vector[29]
            mfcc_var_9 = vector[30]
            mfcc_mean_10 = vector[31]
            mfcc_var_10 = vector[32]
            mfcc_mean_11 = vector[33]
            mfcc_var_11 = vector[34] 
            mfcc_mean_12 = vector[35]
            mfcc_var_12 = vector[36]
            chroma_mean = vector[37]
            chroma_var = vector[38]
            contrast_mean = vector[39]
            contrast_var = vector[40]
            # # 2. Lấy tên
            # col_names = get_feature_names_librosa()
            
            # GHI NGAY LẬP TỨC XUỐNG FILE
            writer.writerow([track_name,track_artist,bpm, rms_mean, rms_var, zcr_mean, zcr_var, centroid_mean, centroid_var, rolloff_mean, rolloff_var, flatness_mean, flatness_var, mfcc_mean_0, mfcc_var_0, mfcc_mean_1, mfcc_var_1, mfcc_mean_2, mfcc_var_2, mfcc_mean_3, mfcc_var_3, mfcc_mean_4, mfcc_var_4, mfcc_mean_5, mfcc_var_5, mfcc_mean_6, mfcc_var_6, mfcc_mean_7, mfcc_var_7, mfcc_mean_8, mfcc_var_8, mfcc_mean_9, mfcc_var_9, mfcc_mean_10, mfcc_var_10, mfcc_mean_11, mfcc_var_11, mfcc_mean_12, mfcc_var_12, chroma_mean, chroma_var, contrast_mean, contrast_var])
            

            print(f"[{index+1}/{total_songs}] ✅ Xong: {term}")
            
        except Exception as e:
            print(f"❌ Lỗi: {term}")
            writer.writerow([track_name,track_artist,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        

        import time
        time.sleep(0.3)

print("🎉 Đã hoàn thành toàn bộ!")