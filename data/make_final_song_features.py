import pandas as pd
import numpy as np



# Read audio file and extract features using pandas 
NEW_SONG_FILE = r"spotify_songs.csv/spotify_genres.csv"
NEW_SONG_POP_FILE = r"spotify_songs.csv/new_song_final_dataset.csv"

DATASET_FILE = r"spotify_songs.csv/new_song_final_dataset_genre.csv"

new_song_pop_df = pd.read_csv(NEW_SONG_FILE, encoding='utf-8-sig', on_bad_lines='skip')
extracted_features_df = pd.read_csv(NEW_SONG_POP_FILE, encoding='utf-8-sig', on_bad_lines='skip')
# 1.1. Xử lý bảng Features (Bảng A)
extracted_features_df['track_artist'] = extracted_features_df['track_artist'].fillna("Unknown").astype(str).str.strip()
extracted_features_df['track_name'] = extracted_features_df['track_name'].fillna("Unknown").astype(str).str.strip()

# 1.2. Xử lý bảng Popularity (Bảng B)
new_song_pop_df['track_artist'] = new_song_pop_df['track_artist'].fillna("Unknown").astype(str).str.strip()
new_song_pop_df['track_name'] = new_song_pop_df['track_name'].fillna("Unknown").astype(str).str.strip()

# (Tùy chọn nâng cao) Chuyển tất cả về chữ thường để khớp chính xác hơn
# extracted_features_df['track_name'] = extracted_features_df['track_name'].str.lower()
# new_song_pop_df['track_name'] = new_song_pop_df['track_name'].str.lower()


# --- BƯỚC 2: THỰC HIỆN MERGE ---
# Sử dụng key là cả 'track_name' VÀ 'track_artist' để tránh nhầm bài trùng tên của ca sĩ khác

final_df = pd.merge(
    new_song_pop_df,
    extracted_features_df,  
    on=['track_name', 'track_artist'], # Khóa để gộp
    how='inner' # Quan trọng: Xem giải thích bên dưới để chọn 'inner' hay 'left'
)

# --- BƯỚC 3: XỬ LÝ TRÙNG LẶP (NẾU CÓ) ---
# Đôi khi merge xong sẽ sinh ra các dòng trùng lặp, nên lọc lại 1 lần cuối
final_df = final_df.drop_duplicates(subset=['track_name', 'track_artist'])

# --- BƯỚC 4: LƯU KẾT QUẢ XUỐNG FILE MỚI ---
final_df.to_csv(DATASET_FILE, index=False, encoding='utf-8-sig')
print(f"Merge thành công! Tổng số bài hát: {len(final_df)}")