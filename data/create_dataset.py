import pandas as pd
import numpy as np



# Read audio file and extract features using pandas 
AUDIO_SPOTIFY_FILE = r"spotify_songs.csv/spotify_songs.csv"
EXTRACTED_FEATURES_FILE = r"spotify_songs.csv/spotify_songs_features_librosa_clean.csv"

DATASET_FILE = r"spotify_songs.csv/dataset.csv"

source_df = pd.read_csv(AUDIO_SPOTIFY_FILE, encoding='utf-8')
extracted_features_df = pd.read_csv(EXTRACTED_FEATURES_FILE, encoding='utf-8')

# keep='first': Giữ lại dòng đầu tiên tìm thấy, xóa các dòng trùng phía sau
# inplace=True: Thay đổi trực tiếp trên dataframe hiện tại (không cần gán lại biến)
source_df.drop_duplicates(subset=['track_name', 'track_artist'], keep='first', inplace=True)
extracted_features_df.drop_duplicates(subset=['track_name', 'track_artist'], keep='first', inplace=True)

print(f"Số dòng còn lại sau khi xóa trùng: {len(source_df)}")
print(f"Số dòng còn lại sau khi xóa trùng: {len(extracted_features_df)}")

key_cols = ['track_name', 'track_artist']

result_df = pd.merge(source_df, extracted_features_df, on=key_cols, how='inner')

print(f"Số dòng sau khi lấy giao của 2 file: {len(result_df)}")
result_df.to_csv(DATASET_FILE, index=False, encoding='utf-8')
print(f"✅ Lưu file dataset tại: {DATASET_FILE}")