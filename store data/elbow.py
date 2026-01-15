import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 1. Load vector (Lấy mẫu 3000 bài thôi cho nhanh)
vectors = np.load('C:\\Music-Recommendation-System\\store data\\embeddings\\genre_vectors_genres.npy')

# 2. Thử nghiệm các giá trị K khác nhau
k_range = range(10, 130, 10) # Thử K = 10, 20, 30 ... 140
inertias = []

print("Computing the elbow")
for k in k_range:
    # n_init=3 để chạy nhanh hơn (mặc định là 10)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=3)
    kmeans.fit(vectors)
    inertias.append(kmeans.inertia_)
    print(f"   -> K={k}, Inertia={kmeans.inertia_:.0f}")

# 3. Vẽ biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'bo-') # Đường màu xanh, có chấm tròn
plt.xlabel('Số lượng cụm (K)')
plt.ylabel('Inertia (Độ lỏng lẻo)')
plt.title('Phương pháp Khuỷu tay (Elbow Method)')
plt.grid(True)
plt.show()