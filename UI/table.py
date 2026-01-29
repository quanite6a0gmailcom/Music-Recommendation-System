import requests
import streamlit as st

@st.cache_data(ttl=3600) # Cache dữ liệu 1 tiếng để đỡ phải tải lại liên tục
def get_apple_music_chart(country_code='vn', limit=20):
    """
    Lấy Top bài hát từ Apple Music RSS theo quốc gia.
    country_code: 'vn' (Việt Nam), 'us' (Mỹ), 'kr' (Hàn Quốc), 'jp' (Nhật)
    """
    try:
        # URL RSS Feed chính chủ của Apple
        url = f"https://rss.applemarketingtools.com/api/v2/{country_code}/music/most-played/{limit}/songs.json"
        
        response = requests.get(url, timeout=5)
        data = response.json()
        
        # Parse dữ liệu JSON trả về
        songs = []
        results = data['feed']['results']
        
        for item in results:
            # Lấy ảnh chất lượng cao hơn (thay đổi kích thước trong URL)
            img_url = item['artworkUrl100'].replace("100x100", "300x300")
            
            song_obj = {
                "id": item['id'],               # ID của Apple Music
                "track_name": item['name'],
                "track_artist": item['artistName'],
                "image": img_url,
                "region": country_code,
                "source": "apple_music"         # Đánh dấu đây là nhạc online
            }
            songs.append(song_obj)
            
        return songs
    except Exception as e:
        print(f"Lỗi lấy Apple Music: {e}")
        return []