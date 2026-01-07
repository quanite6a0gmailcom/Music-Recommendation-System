import yt_dlp
import os
import glob # Thư viện tìm file cực mạnh

def clear_folder(folder_path):
    """Hàm xóa TẤT CẢ các file trong một thư mục"""
    # 1. Nếu folder chưa có thì tạo mới và thoát luôn (vì có gì đâu mà xóa)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return

    # 2. Lấy danh sách tất cả file trong folder
    # Dấu * nghĩa là "tất cả mọi thứ"
    files = glob.glob(os.path.join(folder_path, "*"))
    
    print(f"🧹 Đang dọn dẹp {len(files)} file rác...")
    
    for f in files:
        try:
            # Chỉ xóa file, không xóa thư mục con (nếu có)
            if os.path.isfile(f):
                os.remove(f)
        except Exception as e:
            print(f"⚠️ Không thể xóa {f}: {e}")

def download_clean_start(song_name, artist, output_folder="dataset_audio"):
    # --- BƯỚC 1: XÓA SẠCH SẼ TRƯỚC ---
    clear_folder(output_folder)
    
    # --- BƯỚC 2: CẤU HÌNH TẢI ---
    def range_func(info_dict, ydl):
        return [{'start_time': 0, 'end_time': 70}] # Lấy 1p10s

    ydl_opts = {
        'format': 'bestaudio/best',
        # Tên file cố định là audio.mp3
        'outtmpl': os.path.join(output_folder, 'audio.%(ext)s'),
        'download_ranges': range_func,
        'force_keyframes_at_cuts': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '128',
        }],
        'default_search': 'ytsearch1:',
        'quiet': True,
        'noplaylist': True,
    }

    print(f"⬇️ Đang tải mới: {song_name}...")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(f"{song_name} {artist} official audio", download=True)
            
            # File đích cuối cùng
            final_path = os.path.join(output_folder, "audio.mp3")
            
            if os.path.exists(final_path):
                return final_path
            
    except Exception as e:
        print(f"❌ Lỗi tải: {e}")
        return None

# --- CHẠY VÒNG LẶP TEST ---
if __name__ == "__main__":
    songs = [
        ("Nơi này có anh", "Sơn Tùng M-TP"),
        ("Mang tiền về cho mẹ", "Đen Vâu"),
    ]

    for name, artist in songs:
        print("------------------------------------------------")
        # Mỗi lần chạy hàm này, folder sẽ sạch trơn trước khi tải file mới
        path = download_clean_start(name, artist)
        
        if path:
            print(f"✅ Đã có file sạch tại: {path}")
            # [GỌI HÀM TRÍCH XUẤT ĐẶC TRƯNG TẠI ĐÂY]
            # extract_features(path)...
            
            import time
            time.sleep(1) # Nghỉ tí cho máy đỡ mệt