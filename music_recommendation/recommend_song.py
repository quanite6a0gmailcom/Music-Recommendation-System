import pymongo
import faiss
import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances,cosine_distances
import heapq
from bson import ObjectId

# --- CẤU HÌNH ---
client = pymongo.MongoClient("mongodb://localhost:27017/")
col = client["music_project_db"]["songs"]
all_vectors = np.load("C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_lyrics.npy") # Load toàn bộ vector vào RAM (Mẹ)
clus = client["music_project_db"]["cluster_graph"]
all_vectors_lyrics = np.load("C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_lyrics.npy") 
all_vectors_audio = np.load("C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_audio.npy") 
all_vectors_cmt = np.load("C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_comments.npy") 

def search_within_cluster(song_id, k=5):
    # ---------------------------------------------------------
    # BƯỚC 1: LẤY THÔNG TIN BÀI GỐC & DANH SÁCH ID CÙNG CỤM
    # ---------------------------------------------------------

    seed_song = col.find_one({"_id": ObjectId(song_id)})
    
    if not seed_song:
        print(f"❌ Không tìm thấy bài: {song_id}")
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

    # ---------------------------------------------------------
    # BƯỚC 2: TẠO BỘ VECTOR CON (SUBSET)
    # ---------------------------------------------------------
    # Dùng Numpy Advanced Indexing để trích xuất các dòng tương ứng
    # subset_vectors chỉ chứa các vector thuộc cụm này
    subset_vectors = all_vectors[cluster_indices]

    # ---------------------------------------------------------
    # BƯỚC 3: CHẠY FAISS TRÊN TẬP CON
    # ---------------------------------------------------------
    # Tạo index tạm thời (Rất nhanh vì dữ liệu ít)
    d = subset_vectors.shape[1]
    mini_index = faiss.IndexFlatL2(d)
    mini_index.add(subset_vectors)

    # Lấy vector của bài gốc để query
    query_vector = all_vectors[seed_faiss_id].reshape(1, -1)
    
    # Tìm kiếm (Kết quả trả về là index TRONG TẬP CON, không phải index gốc)
    distances, local_indices = mini_index.search(query_vector, k + 15)
    return cluster_songs,local_indices,seed_faiss_id,cluster_id


def search_in_neighborcluster(cluster_id,root_faiss_id, k=5):
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

    # ---------------------------------------------------------
    # BƯỚC 2: TẠO BỘ VECTOR CON (SUBSET)
    # ---------------------------------------------------------
    # Dùng Numpy Advanced Indexing để trích xuất các dòng tương ứng
    # subset_vectors chỉ chứa các vector thuộc cụm này
    subset_vectors = all_vectors[cluster_indices]

    # ---------------------------------------------------------
    # BƯỚC 3: CHẠY FAISS TRÊN TẬP CON
    # ---------------------------------------------------------
    # Tạo index tạm thời (Rất nhanh vì dữ liệu ít)
    d = subset_vectors.shape[1]
    mini_index = faiss.IndexFlatL2(d)
    mini_index.add(subset_vectors)

    # Lấy vector của bài gốc để query
    query_vector = all_vectors[root_faiss_id].reshape(1, -1)
    
    # Tìm kiếm (Kết quả trả về là index TRONG TẬP CON, không phải index gốc)
    distances, local_indices = mini_index.search(query_vector, k+15)
    return cluster_songs,local_indices

def search_within_cluster_hybrid(song_id, k=5):
    # ---------------------------------------------------------
    # BƯỚC 1: LẤY THÔNG TIN BÀI GỐC & DANH SÁCH ID CÙNG CỤM
    # ---------------------------------------------------------
    seed_song = col.find_one({"_id": ObjectId(song_id)})
    
    if not seed_song:
        print(f"❌ Không tìm thấy bài: {song_id}")
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
        return cluster_songs
    
    # Tách riêng list faiss_id ra để đi cắt vector
    # Ví dụ: cluster_indices = [0, 5, 12, 99...]
    cluster_indices = [s['faiss_id'] for s in cluster_songs]
    num_songs_in_cluster = len(cluster_indices)
    subset_vectors_audio = all_vectors_audio[cluster_indices]
    subset_vectors_lyrics = all_vectors_lyrics[cluster_indices]
    subset_vectors_comments = all_vectors_cmt[cluster_indices]

    # d = subset_vectors_audio.shape[1]
    # mini_index_audio = faiss.IndexFlatL2(d)
    # mini_index_audio.add(subset_vectors_audio)

    # d = subset_vectors_lyrics.shape[1]
    # mini_index_lyric = faiss.IndexFlatL2(d)
    # mini_index_lyric.add(subset_vectors_lyrics)

    # Lấy vector của bài gốc để query
    query_vector_audio = all_vectors_audio[seed_faiss_id].reshape(1, -1)
    query_vector_lyric = all_vectors_lyrics[seed_faiss_id].reshape(1, -1)
    query_vector_comment = all_vectors_cmt[seed_faiss_id].reshape(1,-1)
    
    # # Tìm kiếm (Kết quả trả về là index TRONG TẬP CON, không phải index gốc)
    # distances_audio, local_indices_audio = mini_index_audio.search(query_vector_audio, num_songs_in_cluster)
    # distances_lyric, local_indices_lyric = mini_index_lyric.search(query_vector_lyric, num_songs_in_cluster)

    dis_matrix_audio = euclidean_distances(query_vector_audio,subset_vectors_audio)
    dis_matrix_lyrics = cosine_distances(query_vector_lyric,subset_vectors_lyrics)
    dis_matrix_cmt = cosine_distances(query_vector_comment,subset_vectors_comments)


    hybrid_distance = 0.5 * dis_matrix_audio[0] + 0.25 * dis_matrix_lyrics[0] + 0.25 * dis_matrix_cmt[0]
    local_indices = heapq.nsmallest(10,range(len(hybrid_distance)),key=hybrid_distance.__getitem__)
    local_indices = [cluster_indices[i] for i in local_indices]
    local_indices = local_indices[1:]
    songs = []
    for song in cluster_songs:
        if song['faiss_id'] in local_indices:
            songs.append(song)

    return songs,cluster_id,seed_faiss_id


def search_in_neighborcluster_hybrid(cluster_id,seed_faiss_id, k=5):
    cluster_songs_cursor = col.find(
        {"cluster_id": cluster_id},
        {"faiss_id": 1, "track_name": 1, "track_artist": 1, "_id": 0,"playlist_genre":1}
    ).sort("faiss_id", 1)
    
    cluster_songs = list(cluster_songs_cursor)
    
    if len(cluster_songs) < k:
        return cluster_songs

    # Tách riêng list faiss_id ra để đi cắt vector
    # Ví dụ: cluster_indices = [0, 5, 12, 99...]
    cluster_indices = [s['faiss_id'] for s in cluster_songs]
    num_songs_in_cluster = len(cluster_indices)
    subset_vectors_audio = all_vectors_audio[cluster_indices]
    subset_vectors_lyrics = all_vectors_lyrics[cluster_indices]
    subset_vectors_comments = all_vectors_cmt[cluster_indices]

    # d = subset_vectors_audio.shape[1]
    # mini_index_audio = faiss.IndexFlatL2(d)
    # mini_index_audio.add(subset_vectors_audio)

    # d = subset_vectors_lyrics.shape[1]
    # mini_index_lyric = faiss.IndexFlatL2(d)
    # mini_index_lyric.add(subset_vectors_lyrics)

    # Lấy vector của bài gốc để query
    query_vector_audio = all_vectors_audio[seed_faiss_id].reshape(1, -1)
    query_vector_lyric = all_vectors_lyrics[seed_faiss_id].reshape(1, -1)
    query_vector_comment = all_vectors_cmt[seed_faiss_id].reshape(1,-1)
    
    
    # # Tìm kiếm (Kết quả trả về là index TRONG TẬP CON, không phải index gốc)
    # distances_audio, local_indices_audio = mini_index_audio.search(query_vector_audio, num_songs_in_cluster)
    # distances_lyric, local_indices_lyric = mini_index_lyric.search(query_vector_lyric, num_songs_in_cluster)

    dis_matrix_audio = euclidean_distances(query_vector_audio,subset_vectors_audio)
    dis_matrix_lyrics = cosine_distances(query_vector_lyric,subset_vectors_lyrics)
    dis_matrix_cmt = cosine_distances(query_vector_comment,subset_vectors_comments)


    hybrid_distance = 0.5 * dis_matrix_audio[0] + 0.25 * dis_matrix_lyrics[0] + 0.25 * dis_matrix_cmt[0]
    local_indices = heapq.nsmallest(10,range(len(hybrid_distance)),key=hybrid_distance.__getitem__)
    local_indices = [cluster_indices[i] for i in local_indices]
    local_indices = local_indices[1:]
    songs = []
    for song in cluster_songs:
        if song['faiss_id'] in local_indices:
            songs.append(song)

    return songs

def recommend_hybrid(song_name):
    recommend_list = []
    cluster_songs,current_cluster_id,seed_faiss_id = search_within_cluster_hybrid(song_name,k=20)
    relation = clus.find_one({"cluster_id":current_cluster_id})

    random.shuffle(cluster_songs)
    
    for song in cluster_songs[:11]:  
        recommend_list.append(song)
    
    if not relation:
        return recommend_list
    
    neighbors = relation["nearest_clusters"]
    for neighbor in neighbors:
        neighbor_id = neighbor["cluster_id"]

        cluster_songs = search_in_neighborcluster_hybrid(neighbor_id,seed_faiss_id,k=20)
        found = 0
        random.shuffle(cluster_songs)
        for song in cluster_songs:  
            recommend_list.append(song)
            if found > 5:
                break
            found += 1

    for found_song in recommend_list:
        print(f"{found_song['track_name']} - {found_song.get('track_artist', '')} - {found_song.get('playlist_genre', '')} ")

def recommend(song_name):
    recommend_list = []
    cluster_songs,local_indices,seed_faiss_id,current_cluster_id = search_within_cluster(song_name,k=10)
    relation = clus.find_one({"cluster_id":current_cluster_id})

    for local_idx in np.random.permutation(local_indices[0]):
        # local_idx: Là số thứ tự trong danh sách cluster_songs (0, 1, 2...)
        # KHÔNG PHẢI là faiss_id gốc
        
        found_song = cluster_songs[local_idx]
        
        # Bỏ qua chính bài gốc
        if found_song['faiss_id'] == seed_faiss_id:
            continue
        recommend_list.append(found_song)
    
    if not relation:
        return recommend_list
    
    neighbors = relation["nearest_clusters"]
    for neighbor in neighbors:
        neighbor_id = neighbor["cluster_id"]

        cluster_songs,local_indices = search_in_neighborcluster(neighbor_id,seed_faiss_id,k=5)
        found = 0
        for local_idx in np.random.permutation(local_indices[0]):
            # local_idx: Là số thứ tự trong danh sách cluster_songs (0, 1, 2...)
            # KHÔNG PHẢI là faiss_id gốc
            
            found_song = cluster_songs[local_idx]
            
            recommend_list.append(found_song)
            if found > 5:
                break
            found += 1

    for found_song in recommend_list:
        print(f"{found_song['track_name']} - {found_song.get('track_artist', '')} - {found_song.get('playlist_genre', '')} ")

# --- CHẠY THỬ ---
if __name__ == "__main__":
    # search_within_cluster("Đừng Làm Trái Tim Anh Đau", k=5)
    recommend_hybrid("6969227a474d0281c65fc12e")