import streamlit as st
import urllib.parse
import random
import pymongo
import faiss
import numpy as np
from bson.objectid import ObjectId
from sentence_transformers import SentenceTransformer 


# Import các module của bạn
from search import *
from recommend_song import *
from recommend_song_by_require import *
from table import * # Uncomment nếu có file table.py
from recommend_list_song import *

# --- CẤU HÌNH ---
VALID_GENRES = ["V-Pop", "Vinahouse", "K-Pop", "K-Ballad", "J-Pop & Anime", "C-Pop", "US-UK Pop", "Hip-Hop & Rap", "R&B & Soul", "Rock & Metal", "EDM & Electronic", "Ballad & Lofi", "Indie"]
GENRE_MAPPING = {
    "V-Pop": ["v-pop", "vietnam indie", "vietnamese hip hop"],
    "Vinahouse": ["vinahouse", "stutter house"],
    "K-Pop": ["k-pop", "k-rap", "noise music"],
    "K-Ballad": ["k-ballad", "soundtrack"],
    "J-Pop & Anime": ["j-pop", "anime", "j-rock", "kayokyoku", "japanese indie"],
    "C-Pop": ["mandopop", "cantopop", "c-pop", "taiwanese pop", "gufeng", "chinese r&b"],
    "US-UK Pop": ["pop", "country", "soft pop"],
    "Hip-Hop & Rap": ["hip hop", "rap", "grime", "drill", "trap", "west coast hip hop"],
    "R&B & Soul": ["r&b", "soul", "alternative r&b", "pop soul"],
    "Rock & Metal": ["rock", "metalcore", "deathcore", "alternative rock", "classic rock"],
    "EDM & Electronic": ["edm", "electronic", "house", "progressive house"],
    "Ballad & Lofi": ["folk", "bedroom pop", "soft pop", "acoustic"],
    "Indie": ["indie", "alternative", "bedroom pop"],
    "Latin": ["latin", "reggaeton", "urbano latino"]
}

# --- 1. CẤU HÌNH & KẾT NỐI DATABASE ---
st.set_page_config(page_title="Music AI Hub", page_icon="🎵", layout="wide")

@st.cache_resource
def load_resources():
    print("⏳ Đang tải tài nguyên hệ thống...")
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db_songs = client["music_project_db"]["songs"]
    clus = client["music_project_db"]["cluster_graph"]
    his = client["music_project_db"]["user_history"]
    
    vectors = np.load('C:\\Music-Recommendation-System\\store data\\embeddings2\\genre_vectors_tracknames.npy')
    all_vectors_lyrics = np.load("C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_lyrics.npy") 
    all_vectors_audio = np.load("C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_audio.npy") 
    all_vectors_cmt = np.load("C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_comments.npy") 
    
    index_meta = faiss.IndexFlatL2(vectors.shape[1])
    index_meta.add(vectors)
    
    index_cmt = faiss.IndexFlatL2(all_vectors_cmt.shape[1])
    index_cmt.add(all_vectors_cmt)

    model_bert = SentenceTransformer('all-MiniLM-L6-v2')

    all_songs_metadata = list(db_songs.find().sort("faiss_id", 1))
    
    search_engine = search(db_songs, index_meta, all_songs_metadata) 
    recommender = recommend_song(db_songs, clus, all_vectors_audio, all_vectors_lyrics, all_vectors_cmt)
    recommender_by_require = recommend_song_by_require(db_songs,all_vectors_audio,VALID_GENRES,GENRE_MAPPING)
    recommender_list_song = recommend_list_song(db_songs,his,clus,all_vectors_audio,all_vectors_lyrics,all_vectors_cmt)
    return db_songs, search_engine, recommender,recommender_by_require,recommender_list_song, model_bert, index_cmt, all_songs_metadata

db_songs, search1, recommend1, require1,listsong1, model_bert, index_cmt, all_songs_metadata = load_resources()
DEFAULT_ICON = "https://cdn-icons-png.flaticon.com/512/651/651717.png"

# --- 2. HÀM XỬ LÝ LOGIC ---
def get_recommendations(song_id_str):
    try:
        return recommend1.recommend_hybrid(str(song_id_str))
    except Exception as e:
        st.error(f"Lỗi gợi ý: {e}")
        return []

def search_by_emotion(query_text):
    results = require1.search(query_text)     
    return results

def select_song(song_data):
    st.session_state.current_song = song_data
    if '_id' in song_data:
        str_id = str(song_data['_id'])
        st.session_state.recommendations = get_recommendations(str_id)
    elif 'id' in song_data: 
        st.session_state.recommendations = [] 

# --- HÀM GIẢ LẬP CHO TAB MỚI ---
def get_mock_playlist_data():
    """
    Hàm này tạo dữ liệu giả lập cho Tab 'Dành cho bạn'.
    Nó trả về một tên Playlist ngẫu nhiên và danh sách bài hát ngẫu nhiên từ DB.
    """
    results,playlist_names = listsong1.recommend_list()
    # playlist_names = [
    #     {"name": "Giai Điệu Chữa Lành 🌿", "desc": "Âm nhạc xoa dịu tâm hồn bạn."},
    #     {"name": "Năng Lượng Bùng Nổ ⚡", "desc": "Sạc đầy pin cho ngày mới năng động."},
    #     {"name": "Góc Quán Quen ☕", "desc": "Nhạc Chill nhẹ nhàng như ly cà phê."},
    #     {"name": "Hồi Ức Thanh Xuân 📸", "desc": "Những bài hát gợi nhớ kỷ niệm xưa."}
    # ]
    
    # 2. Chọn random 10 bài hát từ Metadata đã load (Giả lập list bài hát)
    # Nếu DB có dữ liệu thì lấy, không thì dùng list rỗng
    mock_songs = []
    if all_songs_metadata:
        mock_songs = random.sample(all_songs_metadata, min(10, len(all_songs_metadata)))
    
    return playlist_names, results

# --- 3. SESSION STATE ---
if 'current_song' not in st.session_state: st.session_state.current_song = None
if 'recommendations' not in st.session_state: st.session_state.recommendations = []

# ==========================================
# GIAO DIỆN: SIDEBAR
# ==========================================
DB_MUSIC_MOCK = [
    {"id": 1, "track_name": "Lạc Trôi", "track_artist": "Sơn Tùng M-TP", "playlist_genre": "V-Pop", "region": "Việt Nam", "views": 250},
    {"id": 2, "track_name": "Shape of You", "track_artist": "Ed Sheeran", "playlist_genre": "Pop", "region": "US-UK", "views": 900},
    {"id": 3, "track_name": "Flower", "track_artist": "Jisoo", "playlist_genre": "K-Pop", "region": "K-Pop", "views": 600},
]

with st.sidebar:
    st.header("🏆 BXH Apple Music")
    
    # 1. Menu chọn vùng
    region_map = {
        "Việt Nam 🇻🇳": "vn",
        "US-UK 🇺🇸": "us",
        "Hàn Quốc 🇰🇷": "kr",
        "Nhật Bản 🇯🇵": "jp"
    }
    
    selected_region_name = st.selectbox("Chọn quốc gia:", list(region_map.keys()))
    country_code = region_map[selected_region_name]
    
    st.caption(f"Top 20 bài hát thịnh hành tại {selected_region_name}")
    st.markdown("---")
    
    # 2. Gọi hàm lấy dữ liệu thật
    # Hiển thị spinner xoay xoay cho chuyên nghiệp
    with st.spinner("Đang tải BXH..."):
        chart_data = get_apple_music_chart(country_code=country_code, limit=20)
    
    # 3. Hiển thị danh sách
    if chart_data:
        for idx, song in enumerate(chart_data):
            # Layout: Hạng | Ảnh | Tên
            col_rank, col_img, col_info = st.columns([0.7, 1.3, 3])
            
            with col_rank:
                # Top 1, 2, 3 tô màu cho đẹp
                if idx == 0: color = "red"
                elif idx == 1: color = "orange"
                elif idx == 2: color = "green"
                else: color = "grey"
                st.markdown(f"<h3 style='color:{color}; margin:0'>{idx+1}</h3>", unsafe_allow_html=True)
                
            with col_img:
                st.image(song['image'], use_container_width=True)
                
            with col_info:
                st.markdown(f"**{song['track_name']}**")
                st.caption(f"{song['track_artist']}")
                
                # Nút chọn
                # Lưu ý: Bài từ Apple Music có thể KHÔNG có trong Database Vector của bạn
                # Nên khi chọn, ta chỉ cho nó vào Player để nghe Youtube, 
                # chứ không chạy Recommend được (trừ khi bạn code thêm logic Search Vector theo tên)
                if st.button("Play", key=f"am_{song['id']}"):
                    select_song(song)
                    st.rerun()
            
            st.divider()
    else:
        st.error("Không tải được dữ liệu. Kiểm tra mạng!")

# ==========================================
# GIAO DIỆN CHÍNH (SỬ DỤNG TABS)
# ==========================================
st.title("🎵 AI Music Explorer")

# TẠO 2 TAB: Tab 1 chứa code cũ, Tab 2 chứa phần mới
tab_explore, tab_foryou = st.tabs(["🏠 Khám phá & Tìm kiếm", "🎧 Dành riêng cho bạn"])

# ----------------------------------------------------------------
# TAB 1: KHÁM PHÁ (TOÀN BỘ CODE CŨ CỦA BẠN NẰM Ở ĐÂY)
# ----------------------------------------------------------------
with tab_explore:
    st.write("### 🔍 Tìm kiếm bài hát")

    search_mode = st.radio(
        "Chế độ tìm kiếm:", 
        ["🔤 Theo Tên Bài Hát", "🧠 Theo Cảm Xúc/Ngữ Cảnh (AI)"], 
        horizontal=True,
        label_visibility="collapsed"
    )

    search_results = []
    query = ""

    if search_mode == "🔤 Theo Tên Bài Hát":
        query = st.text_input("Nhập tên bài hát...", placeholder="Ví dụ: Sơn Tùng, Lạc Trôi...")
        if query:
            search_results = search1.search(query) 

    else: 
        query = st.text_area("Mô tả cảm xúc hoặc hoàn cảnh...", height=70, 
                             placeholder="Ví dụ: \n- Nhạc buồn thất tình đi dưới mưa \n- Nhạc sôi động để tập Gym...")
        if query and st.button("✨ Phân tích & Tìm kiếm", type="primary"):
            with st.spinner("🤖 AI đang đọc hiểu cảm xúc của bạn..."):
                search_results = search_by_emotion(query) 

    # HIỂN THỊ KẾT QUẢ TÌM KIẾM
    if query and search_results:
        st.write(f"Kết quả cho: '{query}'")
        for idx, song in enumerate(search_results): # Thêm idx để tránh lỗi key trùng
            with st.container(border=True):
                c1, c2, c3 = st.columns([1, 4, 1])
                with c1: st.image(DEFAULT_ICON, width=40)
                with c2:
                    st.subheader(song.get('track_name', 'Unknown'))
                    st.caption(song.get('track_artist', 'Unknown'))
                    if 'playlist_genre' in song:
                        st.caption(f"Genre: {song['playlist_genre']}")
                with c3:
                    safe_id = str(song.get('_id', random.randint(0,10000)))
                    if st.button("Chọn", key=f"search_{safe_id}_{idx}"): # Fix key trùng
                        select_song(song)
                        st.rerun()
    elif query and not search_results:
        st.warning("Không tìm thấy bài hát nào phù hợp.")

    st.divider()

    # KHU VỰC "PLAYER" & GỢI Ý (LOGIC CŨ)
    if st.session_state.current_song:
        curr = st.session_state.current_song
        if isinstance(curr, ObjectId):
            curr = db_songs.find_one({"_id": curr})
            st.session_state.current_song = curr

        if curr:
            with st.container(border=True):
                st.info("💿 ĐANG CHỌN")
                c1, c2, c3 = st.columns([1, 3, 1])
                with c1:
                    st.markdown(f"""<div style="display:flex; justify-content:center;"><img src="{DEFAULT_ICON}" width="100"></div>""", unsafe_allow_html=True)
                with c2:
                    st.header(curr.get('track_name', 'No Name'))
                    st.write(f"👤 {curr.get('track_artist', 'Unknown')} | 🌍 {curr.get('playlist_genre', 'Unknown')}")
                with c3:
                    st.write("")
                    yt_link = curr.get('youtube_search_link')
                    if not yt_link:
                        encoded = urllib.parse.quote_plus(f"{curr.get('track_name')} {curr.get('track_artist')}")
                        yt_link = f"https://www.youtube.com/results?search_query={encoded}"
                    st.link_button("▶️ Play", yt_link, type="primary", use_container_width=True)

            # DANH SÁCH GỢI Ý CŨ
            st.write("### 🤖 Gợi ý tiếp theo cho bạn:")
            if st.session_state.recommendations:
                for idx, rec_song in enumerate(st.session_state.recommendations):
                    with st.container(border=True):
                        col_img, col_info, col_btn = st.columns([1, 6, 2])
                        with col_img: st.image(DEFAULT_ICON, width=50)
                        with col_info:
                            st.markdown(f"**{rec_song.get('track_name', 'Unknown')}**")
                            st.caption(f"{rec_song.get('track_artist', 'Unknown')}")
                        with col_btn:
                            st.write("")
                            safe_id = str(rec_song.get('_id', idx))
                            if st.button("▶ Chọn", key=f"rec_{safe_id}_{idx}", use_container_width=True):
                                select_song(rec_song)
                                st.rerun()
            else:
                st.info("Chưa có gợi ý nào.")

# ----------------------------------------------------------------
# TAB 2: DÀNH RIÊNG CHO BẠN (PHẦN MỚI THÊM VÀO VỚI DỮ LIỆU GIẢ LẬP)
# ----------------------------------------------------------------
with tab_foryou:
    # 1. Gọi hàm lấy dữ liệu giả lập
    # (Mỗi lần reload sẽ random ra một tên playlist khác nhau)
    playlist_info, mock_songs = get_mock_playlist_data()
    
    # 2. Hiển thị Header Playlist (Tên + Mô tả)
    with st.container(border=True):
        col_icon, col_text = st.columns([1, 5])
        with col_icon:
            st.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=80)
        with col_text:
            # Tên Playlist được AI (giả lập) đặt
            st.markdown(f"<h1 style='color: #FF4B4B; margin:0'>{playlist_info['playlist_name']}</h1>", unsafe_allow_html=True)
            st.markdown(f"*{playlist_info['description']}*")
            
            if st.button("🔄 Tạo Playlist Khác"):
                st.cache_data.clear() # Xóa cache để random lại
                st.rerun()

    st.divider()

    # 3. Hiển thị danh sách bài hát trong Playlist giả lập
    if mock_songs:
        for idx, song in enumerate(mock_songs):
            with st.container(border=True):
                c1, c2, c3 = st.columns([0.5, 4, 1])
                with c1:
                    st.markdown(f"**#{idx+1}**")
                with c2:
                    st.markdown(f"**{song.get('track_name', 'Unknown')}**")
                    st.caption(song.get('track_artist', 'Unknown'))
                with c3:
                    # Nút Play vẫn hoạt động bình thường (gọi hàm select_song)
                    safe_id = str(song.get('_id', idx))
                    if st.button("▶", key=f"mix_{safe_id}_{idx}"):
                        select_song(song) # Chuyển sang Tab 1 để phát nhạc
                        st.rerun()
    else:
        st.warning("Chưa có dữ liệu bài hát trong Database để giả lập.")