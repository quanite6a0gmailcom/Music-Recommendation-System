import pymongo
import faiss
import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances,cosine_distances
import heapq
from bson import ObjectId

class recommend_song:
    def __init__(self,db_songs,clus_db,all_vector_audio,all_vector_lyrics,all_vector_cmt):
        self.db_songs = db_songs
        self.clus = clus_db
        self.all_vectors_audio = all_vector_audio
        self.all_vectors_lyrics = all_vector_lyrics
        self.all_vectors_cmt = all_vector_cmt
    
    def search_within_cluster_hybrid(self,song_id, k=5):
        # ---------------------------------------------------------
        # BƯỚC 1: LẤY THÔNG TIN BÀI GỐC & DANH SÁCH ID CÙNG CỤM
        # ---------------------------------------------------------
        seed_song = self.db_songs.find_one({"_id": ObjectId(song_id)})
        
        if not seed_song:
            print(f"❌ Không tìm thấy bài: {song_id}")
            return

        cluster_id = seed_song.get('cluster_id')
        seed_faiss_id = seed_song.get('faiss_id')
        
        print(f"🎵 Bài gốc: {seed_song['track_name']} (Cluster: {cluster_id})")

        # Lấy danh sách các bài trong cùng cụm (Lấy cả faiss_id và thông tin hiển thị)
        # Sort theo faiss_id để khớp thứ tự khi map vector
        cluster_songs_cursor = self.db_songs.find(
            {"cluster_id": cluster_id},
            {"faiss_id": 1, "track_name": 1, "track_artist": 1, "_id": 1,"playlist_genre":1}
        ).sort("faiss_id", 1)
        
        cluster_songs = list(cluster_songs_cursor)
        
        if len(cluster_songs) < k:
            return cluster_songs
        
        # Tách riêng list faiss_id ra để đi cắt vector
        # Ví dụ: cluster_indices = [0, 5, 12, 99...]
        cluster_indices = [s['faiss_id'] for s in cluster_songs]
        num_songs_in_cluster = len(cluster_indices)
        subset_vectors_audio = self.all_vectors_audio[cluster_indices]
        subset_vectors_lyrics = self.all_vectors_lyrics[cluster_indices]
        subset_vectors_comments = self.all_vectors_cmt[cluster_indices]

        # d = subset_vectors_audio.shape[1]
        # mini_index_audio = faiss.IndexFlatL2(d)
        # mini_index_audio.add(subset_vectors_audio)

        # d = subset_vectors_lyrics.shape[1]
        # mini_index_lyric = faiss.IndexFlatL2(d)
        # mini_index_lyric.add(subset_vectors_lyrics)

        # Lấy vector của bài gốc để query
        query_vector_audio = self.all_vectors_audio[seed_faiss_id].reshape(1, -1)
        query_vector_lyric = self.all_vectors_lyrics[seed_faiss_id].reshape(1, -1)
        query_vector_comment = self.all_vectors_cmt[seed_faiss_id].reshape(1,-1)
        
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


    def search_in_neighborcluster_hybrid(self,cluster_id,seed_faiss_id, k=5):
        cluster_songs_cursor = self.db_songs.find(
            {"cluster_id": cluster_id},
            {"faiss_id": 1, "track_name": 1, "track_artist": 1, "_id": 1,"playlist_genre":1}
        ).sort("faiss_id", 1)
        
        cluster_songs = list(cluster_songs_cursor)
        
        if len(cluster_songs) < k:
            return cluster_songs

        # Tách riêng list faiss_id ra để đi cắt vector
        # Ví dụ: cluster_indices = [0, 5, 12, 99...]
        cluster_indices = [s['faiss_id'] for s in cluster_songs]
        num_songs_in_cluster = len(cluster_indices)
        subset_vectors_audio = self.all_vectors_audio[cluster_indices]
        subset_vectors_lyrics = self.all_vectors_lyrics[cluster_indices]
        subset_vectors_comments = self.all_vectors_cmt[cluster_indices]

        # d = subset_vectors_audio.shape[1]
        # mini_index_audio = faiss.IndexFlatL2(d)
        # mini_index_audio.add(subset_vectors_audio)

        # d = subset_vectors_lyrics.shape[1]
        # mini_index_lyric = faiss.IndexFlatL2(d)
        # mini_index_lyric.add(subset_vectors_lyrics)

        # Lấy vector của bài gốc để query
        query_vector_audio = self.all_vectors_audio[seed_faiss_id].reshape(1, -1)
        query_vector_lyric = self.all_vectors_lyrics[seed_faiss_id].reshape(1, -1)
        query_vector_comment = self.all_vectors_cmt[seed_faiss_id].reshape(1,-1)
        
        
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

    def recommend_hybrid(self,song_name):
        recommend_list = []
        cluster_songs,current_cluster_id,seed_faiss_id = self.search_within_cluster_hybrid(song_name,k=20)
        relation = self.clus.find_one({"cluster_id":current_cluster_id})

        random.shuffle(cluster_songs)
        
        for song in cluster_songs[:11]:  
            recommend_list.append(song)
        
        if not relation:
            return recommend_list
        
        neighbors = relation["nearest_clusters"]
        for neighbor in neighbors:
            neighbor_id = neighbor["cluster_id"]

            cluster_songs = self.search_in_neighborcluster_hybrid(neighbor_id,seed_faiss_id,k=20)
            found = 0
            random.shuffle(cluster_songs)
            for song in cluster_songs:  
                recommend_list.append(song)
                if found > 5:
                    break
                found += 1

        return recommend_list
        # for found_song in recommend_list:
        #     print(f"{found_song['track_name']} - {found_song.get('track_artist', '')} - {found_song.get('playlist_genre', '')} ")