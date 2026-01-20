import numpy as np
import pandas as pd
import google.generativeai as genai
import tqdm
import time

# ----CONFIG----

FILE_DATA = 'C:\\Music-Recommendation-System\\store data\\final data\\spotify_songs_final_comments_merged.csv'
API_KEY = 'GEMINI_API_KEY'
BATCH_SIZE = 70
SLEEP_TIME = 2
OUTPUT_FILE_GENRE = 'C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_genres.npy'
OUTPUT_FILE_LYRIC = 'C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_lyrics.npy'
OUTPUT_FILE_COMMENT = 'C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_comments.npy'
OUTPUT_FILE_TRACK_NAME = 'C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_tracknames.npy'
OUTPUT_FILE_TRACK_ARTIST = 'C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_trackartists.npy'
OUTPUT_FILE_TRACK_AUDIO = 'C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_audio.npy'





# Config google api
genai.configure(api_key=API_KEY)

# MAIN PROCESS FUNCTION
df = pd.read_csv(FILE_DATA,encoding='utf-8-sig')

df_genres = df['playlist_genre']
genres = df_genres.to_numpy()

df_lyrics = df['lyrics']
lyrics = df_lyrics.to_numpy()

df_comments = df['comments']
comments = df_comments.to_numpy()

df_tracknames = df['track_name']
tracknames = df_tracknames.to_numpy()

df_trackartists = df['track_artist']
trackartists = df_trackartists.to_numpy()

# --- HÀM XỬ LÝ CHÍNH ---
def generate_and_save_embeddings(texts,OUTPUT_FILE):
    all_embeddings = []
    
    # Tính tổng số batch
    total_items = len(texts)
    # range(start, stop, step) -> Nhảy cóc theo batch_size
    # Ví dụ: 0, 50, 100, 150...
    
    print(f"🚀 Bắt đầu xử lý {total_items} bài hát...")
    print(f"📦 Chia thành {total_items // BATCH_SIZE + 1} gói (batches).")

    # Dùng tqdm để hiện thanh loading
    for i in tqdm.tqdm(range(0, total_items, BATCH_SIZE), desc="Đang Vector hóa"):
        # 1. Cắt lấy 1 gói 50 bài
        batch_texts = texts[i : i + BATCH_SIZE]
        
        try:
            # 2. Gửi lên Google (task_type='retrieval_document' để lưu DB)
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=batch_texts,
                task_type="retrieval_document"
            )
            
            # 3. Lấy kết quả và thêm vào danh sách tổng
            embeddings = result['embedding']
            all_embeddings.extend(embeddings)
            
            # 4. Ngủ một chút để Google không mắng
            time.sleep(SLEEP_TIME)
            
        except Exception as e:
            print(f"\n❌ Lỗi ở batch bắt đầu từ index {i}: {e}")
            # Mẹo: Nếu lỗi, có thể break hoặc lưu tạm những gì đã làm được
            # Ở đây ta chọn dừng chương trình để sửa lỗi
            break

    # --- LƯU FILE ---
    # Chuyển list thường thành numpy array (float32 là chuẩn cho FAISS)
    final_array = np.array(all_embeddings, dtype='float32')
    
    print("\n💾 Đang lưu xuống ổ cứng...")
    np.save(OUTPUT_FILE, final_array)
    
    print(f"✅ HOÀN TẤT! Đã lưu {len(final_array)} vector vào file '{OUTPUT_FILE}'.")
    print(f"Kích thước file: {final_array.shape}")

def save_track_audio(OUTPUT_FILE):
    csv_path = "C:\\Music-Recommendation-System\\store data\\final data\\spotify_songs_final_comments_merged.csv"
    # Các cột bạn muốn trích xuất (Đảm bảo tên đúng y hệt trong CSV)
    feature_cols = [
        "energy","valence","acousticness","instrumentalness","speechiness"
    ]

    # 2. ĐỌC DỮ LIỆU
    print("📂 Đang đọc file CSV...")
    df = pd.read_csv(csv_path)

    # --- QUAN TRỌNG: ĐỒNG BỘ THỨ TỰ ---
    # Nếu bạn đã có faiss_id trong CSV, hãy sort theo nó để đảm bảo
    # dòng 0 trong file npy khớp với faiss_id 0 trong MongoDB.
    if 'faiss_id' in df.columns:
        df = df.sort_values(by='faiss_id')
        print("✅ Đã sắp xếp lại theo faiss_id.")

    # 3. LẤY DỮ LIỆU VÀ XỬ LÝ
    print("⚙️ Đang trích xuất các cột:", feature_cols)
    # Chỉ lấy các cột định nghĩa ở trên
    data = df[feature_cols].values

    final_vectors = data.astype('float32')

    np.save(OUTPUT_FILE_TRACK_AUDIO, final_vectors)

    print(f"🎉 Thành công! Đã lưu {len(final_vectors)} dòng vào '{OUTPUT_FILE_TRACK_AUDIO}'.")
    print(f"   Kích thước vector (Dimension): {final_vectors.shape[1]}")

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    print("Starting embedding genre")
    generate_and_save_embeddings(genres,OUTPUT_FILE_GENRE)
    print("Starting embedding lyric")
    generate_and_save_embeddings(lyrics,OUTPUT_FILE_LYRIC)
    print("Starting embedding comment")
    generate_and_save_embeddings(comments,OUTPUT_FILE_COMMENT)
    print("Starting embedding track name")
    generate_and_save_embeddings(tracknames,OUTPUT_FILE_TRACK_NAME)
    print("Starting embedding track artist")
    generate_and_save_embeddings(trackartists,OUTPUT_FILE_TRACK_ARTIST)
    print("Save track audio to npy file")
    save_track_audio(OUTPUT_FILE_TRACK_AUDIO)

