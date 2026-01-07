import pandas as pd 
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import csv
import os
import time
df = pd.read_csv('C:\\Music-Recommendation-System\\data\\spotify_songs.csv\\new_song_final_dataset.csv',encoding='utf-8-sig')

# --- 1. CẤU HÌNH API  ---
CLIENT_ID = '79837479d0a7488a8c60b1422c466208' 
CLIENT_SECRET = 'b404497a0ee942a581c4d8664ae5477f'

try:
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)
except Exception:
    print("❌ Lỗi: Chưa điền Client ID/Secret.")
    exit()

def get_spotify_genre(song_name, artist_name):
    # Cú pháp tìm kiếm chính xác
    query = f"track:{song_name} artist:{artist_name}"
    
    try:
        # BƯỚC 1: Tìm bài hát để lấy ID Nghệ sĩ
        results = sp.search(q=query, type='track', limit=1)
        items = results['tracks']['items']
        
        if not items:
            return None # Không tìm thấy bài
            
        track = items[0]
        artist_id = track['artists'][0]['id'] # Lấy ID ca sĩ chính
        
        # BƯỚC 2: Truy vấn thông tin Nghệ sĩ để lấy Genre
        artist_info = sp.artist(artist_id)
        genres_list = artist_info.get('genres', [])
        
        # Xử lý danh sách thể loại
        if genres_list:
            # Nối các thể loại lại bằng dấu phẩy (VD: "pop, dance pop")
            genres_str = ", ".join(genres_list)
        else:
            genres_str = "Unknown"

        return {
            'track_name': song_name,
            'track_artist': artist_name,
            'playlist_genre': genres_str
        }

    except Exception as e:
        print(f"Lỗi: {e}")
        return None

if __name__ == "__main__":
    # Danh sách bài hát cần lấy
    filename = r"spotify_songs.csv/spotify_genres.csv"
    data_to_save = []

    print(f"{'Tên bài hát':<30} {'Nghệ sĩ':<20} {'Thể loại (Genres)'}")
    print("=" * 80)

    for row in df.itertuples():
        track_name = row.track_name
        track_artist = row.track_artist
        data = get_spotify_genre(track_name, track_artist)
        
        if data:
            print(f"{data['track_name'][:28]:<30} {data['track_artist'][:18]:<20} {data['playlist_genre']}")
            data_to_save.append(data)
        else:
            print(f"{track_name:<30} {'---':<20} (Không tìm thấy)")
        
        # Nghỉ 0.5 giây để tránh spam API
        time.sleep(0.5)

    # --- LƯU RA CSV ---
    if data_to_save:
        # Dùng mode 'w' để ghi mới, hoặc 'a' để ghi nối tiếp
        with open(filename, mode='w', encoding='utf-8-sig', newline='') as f:
            headers = ['track_name', 'track_artist', 'playlist_genre']
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(data_to_save)
            
        print(f"\n✅ Đã lưu file: {filename}")



