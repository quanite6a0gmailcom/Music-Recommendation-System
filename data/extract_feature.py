import librosa
import numpy as np
import os
import warnings
# Tắt cảnh báo không cần thiết
warnings.filterwarnings('ignore')

def extract_librosa_features(file_path):
    """
    Input: Đường dẫn file MP3
    Output: Vector 1 chiều (numpy array) chứa các đặc trưng Mean & Var.
    """
    if not os.path.exists(file_path):
        print(f"❌ Không tìm thấy file: {file_path}")
        return None

    try:
        # 1. Load Audio
        # Lấy 30 giây đầu (chuẩn phân tích của Spotify) để chạy cho nhanh
        y, sr = librosa.load(file_path, duration=30)
        
        # Tách phần harmonic (giai điệu) và percussive (tiếng gõ/trống)
        # Giúp phân tích kỹ hơn về cấu trúc bài hát
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        features = []

        # --- NHÓM 1: RHYTHM & TEMPO ---
        # Tính Tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(float(tempo)) # [0] BPM

        # --- NHÓM 2: ENERGY & DYNAMICS ---
        # RMS Energy (Độ to)
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms)) # [1] RMS Mean
        features.append(np.var(rms))  # [2] RMS Var

        # Zero Crossing Rate (Độ nhiễu/Gõ)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr)) # [3] ZCR Mean
        features.append(np.var(zcr))  # [4] ZCR Var

        # --- NHÓM 3: SPECTRAL (Màu sắc âm thanh) ---
        # Spectral Centroid (Độ sáng - Brightness) -> Valence
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(centroid))
        features.append(np.var(centroid))

        # Spectral Rolloff (Hình dáng phổ)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(rolloff))
        features.append(np.var(rolloff))
        
        # Spectral Flatness (Độ phẳng - Noise vs Tone) -> Acousticness
        flatness = librosa.feature.spectral_flatness(y=y)
        features.append(np.mean(flatness))
        features.append(np.var(flatness))

        # --- NHÓM 4: TIMBRE (Quan trọng nhất cho Instrumentalness/Genre) ---
        # MFCC (Mel-frequency cepstral coefficients)
        # Lấy 13 hệ số đầu tiên
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Tính Mean và Var cho từng hệ số (tổng cộng 26 đặc trưng)
        for i in range(13):
            features.append(np.mean(mfcc[i]))
            features.append(np.var(mfcc[i]))

        # --- NHÓM 5: HARMONY (Giai điệu/Key) ---
        # Chroma STFT (12 nốt nhạc) -> Key
        chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr)
        features.append(np.mean(chroma)) # Trung bình độ mạnh của các nốt
        features.append(np.var(chroma))

        # Spectral Contrast (Độ tương phản đỉnh/đáy phổ)
        # Rất tốt để phân biệt cấu trúc bài hát
        contrast = librosa.feature.spectral_contrast(y=y_harmonic, sr=sr)
        features.append(np.mean(contrast))
        features.append(np.var(contrast))

        return np.array(features, dtype=np.float32)

    except Exception as e:
        print(f"❌ Lỗi xử lý {file_path}: {e}")
        return None

# --- HÀM LẤY TÊN CỘT (Để lưu CSV) ---
def get_feature_names_librosa():
    names = [
        "bpm", 
        "rms_mean", "rms_var", 
        "zcr_mean", "zcr_var",
        "centroid_mean", "centroid_var",
        "rolloff_mean", "rolloff_var",
        "flatness_mean", "flatness_var"
    ]
    
    # MFCC names
    for i in range(13):
        names.append(f"mfcc_mean_{i}")
        names.append(f"mfcc_var_{i}")
        
    names.extend([
        "chroma_mean", "chroma_var",
        "contrast_mean", "contrast_var"
    ])
    return names

# --- CHẠY THỬ ---
if __name__ == "__main__":
    file_mp3 = "dataset_audio/audio.mp3" # Đường dẫn file của bạn
    
    # 1. Lấy dữ liệu
    vector = extract_librosa_features(file_mp3)
    
    # 2. Lấy tên
    col_names = get_feature_names_librosa()
    
    print(vector[0])
    # if vector is not None:
    #     print(f"✅ Trích xuất thành công: {len(vector)} chiều dữ liệu.")
        
    #     # In ra đẹp mắt
    #     import pandas as pd
    #     df = pd.DataFrame([vector], columns=col_names)
    #     print(df.T) # Xoay dọc để dễ nhìn