import numpy as np
from sklearn.cluster import KMeans
from pymongo import MongoClient
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
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

print("Save to file csv")
df.to_csv(DATA_FILE,index=False,encoding='utf-8-sig')

print("✅ Phân cụm hoàn tất!")
print(f"Ví dụ 10 bài đầu tiên thuộc các cụm: {cluster_labels[:10]}")

