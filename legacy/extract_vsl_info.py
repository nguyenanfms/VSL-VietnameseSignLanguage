import os
import cv2
import json
import time
import sys
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reconfigure stdout to support UTF-8 if possible, to avoid Windows encoding issues
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

def get_video_metadata(video_path, gloss):
    """
    Trích xuất metadata của một video dùng OpenCV.
    """
    video_filename = os.path.basename(video_path)
    videoid = os.path.splitext(video_filename)[0]
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "videoid": videoid,
                "fps": 0.0,
                "resolution": "0x0",
                "gloss": gloss,
                "num_frames": 0,
                "duration": 0.0,
                "error": "Cannot open video file"
            }
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Nếu fps hợp lệ, tính thời gian video
        duration = 0.0
        if fps > 0:
            duration = round(num_frames / fps, 2)
            
        resolution = f"{width}x{height}"
        
        cap.release()
        
        return {
            "videoid": videoid,
            "fps": round(fps, 2) if fps > 0 else 0.0,
            "resolution": resolution,
            "gloss": gloss,
            "num_frames": num_frames,
            "duration": duration
        }
    except Exception as e:
        return {
            "videoid": videoid,
            "fps": 0.0,
            "resolution": "0x0",
            "gloss": gloss,
            "num_frames": 0,
            "duration": 0.0,
            "error": str(e)
        }

def main():
    # Thư mục chứa dataset
    vsl_dir = os.path.join(os.getcwd(), "VSL_FULL_FRONT")
    if not os.path.exists(vsl_dir):
        # Thử đường dẫn khác nếu có trong data/
        vsl_dir = os.path.join(os.getcwd(), "data", "VSL_FULL_FRONT")
        
    if not os.path.exists(vsl_dir):
        print(f"Directory not found: {vsl_dir}")
        return
        
    print(f"Scanning directory: {vsl_dir}")
    
    # Gom danh sách tất cả các video cần xử lý
    tasks = []
    # Các định dạng video được hỗ trợ
    video_extensions = ('.mp4', '.avi', '.mkv', '.mov', '.flv')
    
    # Duyệt qua các thư mục gloss
    for root, dirs, files in os.walk(vsl_dir):
        # gloss là tên thư mục con trực tiếp của VSL_FULL_FRONT
        rel_path = os.path.relpath(root, vsl_dir)
        if rel_path == ".":
            continue
            
        # Lấy tên gloss (thường là phần đầu tiên trong rel_path)
        gloss = rel_path.split(os.sep)[0]
        
        for file in files:
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)
                tasks.append((video_path, gloss))
                
    total_videos = len(tasks)
    print(f"Total videos found: {total_videos}")
    
    if total_videos == 0:
        print("No videos found to process.")
        return
        
    # Xử lý song song bằng ThreadPoolExecutor để tăng tốc độ
    results = []
    print("Extracting video metadata...")
    
    start_time = time.time()
    
    # Sử dụng 16 luồng xử lý đồng thời
    with ThreadPoolExecutor(max_workers=16) as executor:
        # Submit tất cả công việc
        futures = {executor.submit(get_video_metadata, path, gloss): (path, gloss) for path, gloss in tasks}
        
        # tqdm hiển thị thanh tiến trình
        for future in tqdm(as_completed(futures), total=total_videos, desc="Scanning videos"):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                path, gloss = futures[future]
                print(f"\nError processing {path}: {e}")
                
    elapsed_time = time.time() - start_time
    print(f"\nFinished scanning in {elapsed_time:.2f} seconds.")
    
    # Lưu kết quả ra file JSON
    output_json_path = os.path.join(os.getcwd(), "vsl_full_front.json")
    print(f"Saving information to file: {output_json_path}")
    
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print("Process completed successfully!")

if __name__ == "__main__":
    main()
