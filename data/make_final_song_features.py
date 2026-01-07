import pandas as pd
import numpy as np



# Read audio file and extract features using pandas 
NEW_SONG_FILE = r"spotify_songs.csv/clean_apple_music_trending.csv"
NEW_SONG_POP_FILE = r"spotify_songs.csv/clean_apple_music_trending_predicted_features_processed.csv"

DATASET_FILE = r"spotify_songs.csv/new_song_final_dataset.csv"

source_df = pd.read_csv(NEW_SONG_FILE, encoding='utf-8-sig')
extracted_features_df = pd.read_csv(NEW_SONG_POP_FILE, encoding='utf-8-sig', on_bad_lines='skip')
extracted_features_df = extracted_features_df.reset_index(drop=True)
new_song_pop_df = source_df.reset_index(drop=True)

result_df = pd.concat([source_df, extracted_features_df], axis=1)

print(f"Số dòng sau khi lấy giao của 2 file: {len(result_df)}")
result_df.to_csv(DATASET_FILE, index=False, encoding='utf-8')
print(f"✅ Lưu file dataset tại: {DATASET_FILE}")