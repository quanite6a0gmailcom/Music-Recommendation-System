# 🎵 AI Music Explorer - Hệ thống Gợi ý Âm nhạc Đa phương thức

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B.svg?logo=Streamlit&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-47A248.svg?logo=MongoDB&logoColor=white)
![FAISS](https://img.shields.io/badge/Search-FAISS-yellow)
![AI Model](https://img.shields.io/badge/Model-Hybrid%20Fusion-purple)

## 📖 Giới thiệu (Overview)

**AI Music Explorer** là hệ thống gợi ý âm nhạc thông minh được xây dựng nhằm giải quyết vấn đề "Khoảng cách ngữ nghĩa" (Semantic Gap) trong các hệ thống nghe nhạc truyền thống.

Khác với các ứng dụng chỉ dựa trên giai điệu (Audio Features) hoặc siêu dữ liệu (Metadata), hệ thống này kết hợp sức mạnh của **Xử lý ngôn ngữ tự nhiên (NLP)** để thấu hiểu:
1.  **Nội dung lời bài hát (Lyrics).**
2.  **Cảm xúc cộng đồng (Social Sentiment)** thông qua phân tích bình luận.
3.  **Giai điệu bài hát (Audio Features)** từ Spotify.

Hệ thống cho phép người dùng tìm kiếm nhạc theo "Vibe" (ví dụ: *"Nhạc buồn thất tình đi dưới mưa"*) và tự động tạo danh sách phát cá nhân hóa (Daily Mix).

## ✨ Tính năng chính (Key Features)

### 1. 🔍 Tìm kiếm thông minh (Semantic Search)
* **Search by Text:** Tìm kiếm theo tên bài hát, ca sĩ, thể loại.
* **Search by Emotion/Context:** Nhập mô tả tự nhiên, hệ thống sử dụng model **Sentence-BERT** để hiểu ý định và trả về bài hát phù hợp ngữ cảnh (Ví dụ: *"Nhạc sôi động để tập gym"*).

### 2. 🤖 Gợi ý lai (Hybrid Recommendation Engine)
Hệ thống sử dụng thuật toán **Weighted Late Fusion** để tổng hợp độ tương đồng từ 3 nguồn vector:
* `Vector Audio`: Tempo, Energy, Valence, Danceability...
* `Vector Lyrics`: Ý nghĩa ca từ.
* `Vector Social`: Cảm xúc từ bình luận người dùng.

### 3. 🎧 Cá nhân hóa (Personalization)
* **Lịch sử nghe nhạc:** Lưu trữ thời gian thực các bài hát người dùng đã tương tác vào MongoDB.
* **Daily Mix:** Tự động phân tích lịch sử nghe gần nhất để tạo Playlist trộn lẫn các bài hát mới phù hợp với Gu hiện tại.
* **AI Playlist Naming:** Sử dụng **Google Gemini API** để tự động đặt tên và viết mô tả cực "chill" cho Playlist vừa tạo.

### 4. ⚡ Hiệu năng cao
* Tối ưu hóa tốc độ truy vấn trên tập dữ liệu lớn bằng **FAISS (Facebook AI Similarity Search)**.
* Sử dụng **Clustering (K-Means)** để gom nhóm các bài hát tương đồng.

---

## 🛠️ Công nghệ sử dụng (Tech Stack)

| Hạng mục | Công nghệ / Thư viện |
| :--- | :--- |
| **Ngôn ngữ** | Python 3.10+ |
| **Frontend** | Streamlit |
| **Database** | MongoDB (NoSQL) |
| **Vector Search** | FAISS |
| **NLP / Embedding** | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| **Generative AI** | Google Gemini API (GenAI) |
| **Data Processing** | Pandas, NumPy, Scikit-learn |

---

## ⚙️ Cài đặt và Chạy dự án (Installation)

### 1. Yêu cầu tiên quyết
* Python 3.8 trở lên.
* MongoDB đã được cài đặt và đang chạy tại `localhost:27017`.

### 2. Clone dự án
```bash
git clone [https://github.com/username/music-recommendation-system.git](https://github.com/username/music-recommendation-system.git)
cd music-recommendation-system