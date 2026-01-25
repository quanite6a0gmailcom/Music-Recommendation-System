import pymongo
import faiss
import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances
import heapq

# --- CẤU HÌNH ---
client = pymongo.MongoClient("mongodb://localhost:27017/")
col = client["music_project_db"]["songs"]
all_vectors_lyrics = np.load("C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_lyrics.npy") 
all_vectors_audio = np.load("C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_audio.npy") 
clus = client["music_project_db"]["cluster_graph"]

def search_within_cluster(song_name, k=5):
    # ---------------------------------------------------------
    # BƯỚC 1: LẤY THÔNG TIN BÀI GỐC & DANH SÁCH ID CÙNG CỤM
    # ---------------------------------------------------------
    seed_song = col.find_one({"track_name": {"$regex": f"^{song_name}$", "$options": "i"}})
    
    if not seed_song:
        print(f"❌ Không tìm thấy bài: {song_name}")
        return

    cluster_id = seed_song.get('cluster_id')
    seed_faiss_id = seed_song.get('faiss_id')
    
    print(f"🎵 Bài gốc: {seed_song['track_name']} (Cluster: {cluster_id})")

    # Lấy danh sách các bài trong cùng cụm (Lấy cả faiss_id và thông tin hiển thị)
    # Sort theo faiss_id để khớp thứ tự khi map vector
    cluster_songs_cursor = col.find(
        {"cluster_id": cluster_id},
        {"faiss_id": 1, "track_name": 1, "track_artist": 1, "_id": 0,"playlist_genre":1}
    ).sort("faiss_id", 1)
    
    cluster_songs = list(cluster_songs_cursor)
    
    if len(cluster_songs) < k:
        print("⚠️ Cụm này ít bài quá, lấy hết luôn.")
        return cluster_songs
    
    # Tách riêng list faiss_id ra để đi cắt vector
    # Ví dụ: cluster_indices = [0, 5, 12, 99...]
    cluster_indices = [s['faiss_id'] for s in cluster_songs]
    num_songs_in_cluster = len(cluster_indices)
    subset_vectors_audio = all_vectors_audio[cluster_indices]
    subset_vectors_lyrics = all_vectors_lyrics[cluster_indices]

    # d = subset_vectors_audio.shape[1]
    # mini_index_audio = faiss.IndexFlatL2(d)
    # mini_index_audio.add(subset_vectors_audio)

    # d = subset_vectors_lyrics.shape[1]
    # mini_index_lyric = faiss.IndexFlatL2(d)
    # mini_index_lyric.add(subset_vectors_lyrics)

    # Lấy vector của bài gốc để query
    query_vector_audio = all_vectors_audio[seed_faiss_id].reshape(1, -1)
    query_vector_lyric = all_vectors_lyrics[seed_faiss_id].reshape(1, -1)
    
    # # Tìm kiếm (Kết quả trả về là index TRONG TẬP CON, không phải index gốc)
    # distances_audio, local_indices_audio = mini_index_audio.search(query_vector_audio, num_songs_in_cluster)
    # distances_lyric, local_indices_lyric = mini_index_lyric.search(query_vector_lyric, num_songs_in_cluster)

    dis_matrix_audio = euclidean_distances(query_vector_audio,subset_vectors_audio)
    dis_matrix_lyrics = euclidean_distances(query_vector_lyric,subset_vectors_lyrics)

    hybrid_distance = 0.5 * dis_matrix_audio[0] + 0.5 * dis_matrix_lyrics[0]
    local_indices = heapq.nsmallest(10,range(len(hybrid_distance)),key=hybrid_distance.__getitem__)
    local_indices = [cluster_indices[i] for i in local_indices]
    local_indices = local_indices[1:]
    songs = []
    for song in cluster_songs:
        if song['faiss_id'] in local_indices:
            songs.append(song)

    return songs



# songs = search_within_cluster("夜に駆ける")
# random.shuffle(songs)
print(len(col.distinct("playlist_genre")))

