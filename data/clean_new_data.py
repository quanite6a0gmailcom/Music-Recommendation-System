import pandas as pd

INPUT = r'spotify_songs.csv/apple_music_trending.csv'
OUTPUT = r'spotify_songs.csv/clean_apple_music_trending.csv'
df = pd.read_csv(INPUT)
print("Số bài hát ban đầu:", len(df))
# Loại bỏ bản ghi trùng lặp và bản ghi thiếu dữ liệu
df = df.drop_duplicates(subset=['track_name', 'artist_name'])
df = df.dropna(subset=['popularity'])
df.to_csv(OUTPUT, index=False)
print(len(df), "bài hát sau khi làm sạch.")
print(f"✅ Đã làm sạch và lưu dữ liệu vào '{OUTPUT}' thành công!")
