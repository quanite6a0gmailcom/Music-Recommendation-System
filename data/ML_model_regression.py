import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import MinMaxScaler
from keras import layers, models
import tensorflow as tf 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import joblib

dataset_path = r'spotify_songs.csv/dataset.csv'
OUTPUT_FEATURES_REGRESS = ['energy','valence','acousticness','instrumentalness','speechiness']
INPUT_FEATURES_REGRESS = ['bpm', 'rms_mean', 'rms_var', 'zcr_mean', 'zcr_var', 'centroid_mean', 'centroid_var', 'rolloff_mean', 'rolloff_var', 'flatness_mean', 'flatness_var', 'mfcc_mean_0', 'mfcc_var_0', 'mfcc_mean_1', 'mfcc_var_1', 'mfcc_mean_2', 'mfcc_var_2', 'mfcc_mean_3', 'mfcc_var_3', 'mfcc_mean_4', 'mfcc_var_4', 'mfcc_mean_5', 'mfcc_var_5', 'mfcc_mean_6', 'mfcc_var_6', 'mfcc_mean_7', 'mfcc_var_7', 'mfcc_mean_8', 'mfcc_var_8', 'mfcc_mean_9', 'mfcc_var_9', 'mfcc_mean_10', 'mfcc_var_10', 'mfcc_mean_11', 'mfcc_var_11', 'mfcc_mean_12', 'mfcc_var_12', 'chroma_mean', 'chroma_var', 'contrast_mean', 'contrast_var']


df = pd.read_csv(dataset_path)

y = df[OUTPUT_FEATURES_REGRESS]
X = df[INPUT_FEATURES_REGRESS]

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Load thành công!")
print(f"- Số lượng mẫu huấn luyện (Train): {len(X_train)} bài")
print(f"- Số lượng mẫu kiểm thử (Test):  {len(X_test)} bài")
print(f"- Số lượng đặc trưng đầu vào (Input Features): {X_train.shape[1]}")
print(f"- Số lượng mục tiêu dự đoán (Outputs): {y_train.shape[1]}")

scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_test = scaler_y.fit_transform(y_test)
y_train = scaler_y.fit_transform(y_train)

model = models.Sequential()
model.add(layers.Input(shape=(X_train.shape[1],)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(y_train.shape[1], activation='tanh'))
opt = tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.004) # Mặc định là 0.001
model.compile(optimizer=opt, loss='mse', metrics=['mae'])
print("⏳ Đang huấn luyện mô hình...")

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=500, min_lr=1e-6, verbose=1)
early_stopper = EarlyStopping(monitor='val_loss', patience=2000, restore_best_weights=True, verbose=1)
history = model.fit(X_train, y_train, epochs=5000, batch_size=32, validation_split=0.2, verbose=1, callbacks=[lr_scheduler, early_stopper])

model.save(r'model_weight/spotify_regression_model.keras')
joblib.dump(scaler, r'model_weight/scaler_X.save')
joblib.dump(scaler_y, r'model_weight/scaler_y.save')
# Hàm vẽ biểu đồ quá trình huấn luyện
def plot_training_history(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Kiểm tra xem bạn dùng 'mae' hay 'mean_absolute_error' trong metrics
    if 'mae' in history.history:
        mae = history.history['mae']
        val_mae = history.history['val_mae']
    else:
        mae = history.history['mean_absolute_error']
        val_mae = history.history['val_mean_absolute_error']

    epochs = range(1, len(loss) + 1)

    # Tạo khung vẽ 2 biểu đồ cạnh nhau
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Biểu đồ 1: Loss (MSE) ---
    ax1.plot(epochs, loss, 'y', label='Training Loss')
    ax1.plot(epochs, val_loss, 'r', label='Validation Loss')
    ax1.set_title('Training and Validation Loss (MSE)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # --- Biểu đồ 2: Metric (MAE) ---
    ax2.plot(epochs, mae, 'b', label='Training MAE')
    ax2.plot(epochs, val_mae, 'g', label='Validation MAE')
    ax2.set_title('Training and Validation MAE')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('MAE')
    ax2.legend()
    ax2.grid(True)

    plt.show()

# Gọi hàm để vẽ
plot_training_history(history)
# --- BƯỚC 1: Dự đoán trên tập Test ---
# Kết quả trả về đang ở dạng [0, 1] do bạn đã scale trước đó
print("Đang dự đoán...")
y_pred_scaled = model.predict(X_test)

# --- BƯỚC 2: Đưa về giá trị gốc (Inverse Transform) ---
# Dùng scaler_y để đưa cả dự đoán và thực tế về đơn vị ban đầu (ví dụ: nhịp/phút, độ lớn...)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test)

# --- BƯỚC 3: Tính toán các chỉ số sai số ---

# 1. MAE (Mean Absolute Error): Sai số tuyệt đối trung bình
# Ví dụ: MAE = 0.05 nghĩa là trung bình mô hình đoán lệch 0.05 đơn vị
mae = mean_absolute_error(y_test_original, y_pred)

# 2. RMSE (Root Mean Squared Error): Căn bậc hai của sai số bình phương trung bình
# RMSE thường phạt nặng các sai số lớn (outliers)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))

# 3. R2 Score (R-squared): Độ phù hợp của mô hình (từ -inf đến 1)
# R2 càng gần 1 càng tốt. Nếu âm nghĩa là mô hình đoán còn tệ hơn đoán bừa.
r2 = r2_score(y_test_original, y_pred)

print("-" * 30)
print(f"Kết quả đánh giá trên tập Test:")
print(f"MAE  (Sai số trung bình): {mae:.4f}")
print(f"RMSE (Độ lệch chuẩn sai số): {rmse:.4f}")
print(f"R2 Score (Độ chính xác): {r2:.4f}")
print("-" * 30)

# --- BƯỚC 4: Xem thử vài mẫu cụ thể ---
# So sánh 5 dòng đầu tiên để xem mắt thường
print("\nSo sánh Thực tế vs Dự đoán (5 mẫu đầu):")
for i in range(5):
    print(f"Mẫu {i+1}: Thực tế = {y_test_original[i]}, Dự đoán = {y_pred[i]}")

def plot_prediction_scatter(y_true, y_pred, title="So sánh Thực tế vs Dự đoán"):
    plt.figure(figsize=(8, 8))
    
    # Vẽ các điểm dữ liệu
    # Trục X là giá trị thực, Trục Y là giá trị mô hình đoán
    plt.scatter(y_true, y_pred, alpha=0.3, color='blue')

    # Vẽ đường chéo chuẩn 45 độ (Đường hoàn hảo)
    # Nếu điểm nằm trên đường này nghĩa là đoán trúng phóc 100%
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Dự đoán hoàn hảo')

    plt.title(title)
    plt.xlabel('Giá trị Thực tế')
    plt.ylabel('Giá trị Dự đoán')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- CÁCH DÙNG ---
# Chọn 1 cột để vẽ (ví dụ cột số 1 là 'Energy')
# Nhớ dùng dữ liệu ĐÃ NGHỊCH ĐẢO (đơn vị gốc)
col_index = 1 
col_name = "Energy" # Tên cột tương ứng

# Lấy cột dữ liệu tương ứng
y_test_col = y_test_original[:, col_index]
y_pred_col = y_pred[:, col_index]

plot_prediction_scatter(y_test_col, y_pred_col, title=f"Đánh giá mô hình: {col_name}")
# 3. KHỞI TẠO MÔ HÌNH
# Random Forest không hỗ trợ multi-output tự nhiên tốt bằng việc bao bọc nó
# MultiOutputRegressor sẽ tạo ra 1 model riêng cho từng cột Y (Dance, Energy...) nhưng chạy song song
# model = MultiOutputRegressor(RandomForestRegressor(
#     n_estimators=100,  # Số lượng cây (càng nhiều càng chính xác nhưng chậm)
#     max_depth=20,      # Độ sâu tối đa để tránh học vẹt
#     random_state=42,
#     n_jobs=-1          # Dùng tất cả nhân CPU để chạy cho nhanh
# ))

# # 4. HUẤN LUYỆN
# print("⏳ Đang huấn luyện mô hình...")
# model.fit(X_train, y_train)

# # 5. ĐÁNH GIÁ
# print("✅ Đã xong! Đang kiểm tra độ chính xác...")
# y_pred = model.predict(X_test)

# y_test = y_test.to_numpy()

# # Đánh giá từng chỉ số Spotify
# spotify_features = ['energy', 'valence', 'acousticness', 'instrumentalness', 'speechiness']

# for i, feature in enumerate(spotify_features):
#     # R2 Score: Càng gần 1 càng tốt (1 là đoán trúng phóc, 0 là đoán bừa)
#     # MSE: Càng nhỏ càng tốt
#     r2 = r2_score(y_test[:, i], y_pred[:, i])
#     mse = mean_squared_error(y_test[:, i], y_pred[:, i])
#     print(f"- {feature:<15}: R2 Score = {r2:.4f} | MSE = {mse:.4f}")

# # 6. DỰ ĐOÁN BÀI MỚI
# # vector_moi = extract_features("bai_hat_moi.mp3")
# # ket_qua = model.predict([vector_moi])