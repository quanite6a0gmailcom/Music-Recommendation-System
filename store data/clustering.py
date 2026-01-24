import numpy as np
from sklearn.cluster import KMeans
from pymongo import MongoClient
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

# 1. Cấu hình
client = MongoClient("mongodb://localhost:27017/")
db = client["music_project_db"]
col_graph = db["cluster_graph"] # Collection mới để lưu quan hệ cụm

# --- CẤU HÌNH ---
VECTOR_FILE = 'C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_genres.npy' # File vector bạn đã tạo ở bước trước
NUM_CLUSTERS = 30                  # Bạn muốn chia thành bao nhiêu cộng đồng? (30k bài thì 50-100 cụm là đẹp)
DATA_FILE = "C:\\Music-Recommendation-System\\store data\\final data\\spotify_songs_final_comments_merged.csv"
# 1. Load Vector
print("1. Đang tải vector...")
vectors = np.load(VECTOR_FILE)
df = pd.read_csv(DATA_FILE,encoding='utf-8-sig')
df['faiss_id'] = range(len(df))

# 2. Thực hiện K-Means
print(f"2. Đang phân chia thành {NUM_CLUSTERS} cộng đồng (Clustering)...")
# random_state=42 để kết quả cố định mỗi lần chạy
kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init=10)
kmeans.fit(vectors)

# labels_ là danh sách chứa mã số cộng đồng của từng bài hát
# Ví dụ: [0, 5, 5, 1, 0...] nghĩa là bài 0 thuộc cụm 0, bài 1 thuộc cụm 5...
cluster_labels = kmeans.labels_
df['cluster_id'] = cluster_labels

# print("Save to file csv")
# df.to_csv(DATA_FILE,index=False,encoding='utf-8-sig')

# print("✅ Phân cụm hoàn tất!")
# print(f"Ví dụ 10 bài đầu tiên thuộc các cụm: {cluster_labels[:10]}")
# 1. Lấy tọa độ các tâm cụm (Centroids)
# centroids shape: (50, 384) nếu bạn có 50 cụm và vector 384 chiều
centroids = kmeans.cluster_centers_

# 2. Tính ma trận khoảng cách
# Kết quả là ma trận 50x50
dist_matrix = euclidean_distances(centroids, centroids)
np.fill_diagonal(dist_matrix, np.inf)

# 3. Chuẩn bị dữ liệu để lưu
data_to_insert = []
TOP_K_NEIGHBORS = 5 # Lưu 5 cụm gần nhất cho mỗi cụm


for i in range(len(dist_matrix)):
    # Lấy khoảng cách từ cụm i đến tất cả các cụm khác
    dists = dist_matrix[i]
    
    # Lấy index của 5 cụm gần nhất (sắp xếp tăng dần)
    nearest_indices = np.argsort(dists)[:TOP_K_NEIGHBORS]
    
    # Tạo danh sách neighbors chi tiết
    neighbors = []
    for idx in nearest_indices:
        neighbors.append({
            "cluster_id": int(idx),
            "distance": float(dists[idx])
        })
    
    # Tạo document
    doc = {
        "cluster_id": i,
        "nearest_clusters": neighbors
    }
    data_to_insert.append(doc)

# 4. Lưu vào MongoDB
col_graph.delete_many({}) # Xóa cũ
col_graph.insert_many(data_to_insert)

print("✅ Đã lưu sơ đồ quan hệ các cụm vào MongoDB!")

# # 3. Hiển thị đẹp bằng Pandas DataFrame
# df_dist = pd.DataFrame(dist_matrix)

# print("Ma trận khoảng cách giữa các cụm (10x10 góc trên):")
# print(df_dist.iloc[:10, :10]) # Chỉ in 10 cụm đầu xem thử
# df_dist.to_csv('out.csv',index=False,encoding='utf-8-sig')

# # 4. Xem cụ thể khoảng cách giữa 2 cụm bất kỳ
# c1 = 0
# c2 = 5
# print(f"\nKhoảng cách giữa Cụm {c1} và Cụm {c2}: {dist_matrix[c1][c2]:.4f}")
