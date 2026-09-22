import cv2
import numpy as np
import mediapipe as mp
import os
import math

# Khởi tạo giải pháp Pose của MediaPipe
mp_pose = mp.solutions.pose

# Danh sách 6 điểm mốc liên quan đến tay/vai 
LANDMARKS = {
    "LShoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "RShoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "LElbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "RElbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "LWrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "RWrist": mp_pose.PoseLandmark.RIGHT_WRIST,
}

# BIẾN TOÀN CỤC TRÊN MỖI TIẾN TRÌNH (NHÂN CPU)
# Giúp khởi tạo MediaPipe ĐÚNG MỘT LẦN DUY NHẤT cho mỗi nhân và tái sử dụng mãi mãi
_GLOBAL_POSE_DETECTOR = None

def _get_pose_detector():
    global _GLOBAL_POSE_DETECTOR
    if _GLOBAL_POSE_DETECTOR is None:
        # Khởi tạo một lần duy nhất khi nhân CPU đó nhận video đầu tiên
        _GLOBAL_POSE_DETECTOR = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    return _GLOBAL_POSE_DETECTOR

def _angle_2d(a, b, c):
    """Tính góc ABC (độ) trong mặt phẳng 2D, luôn trả về int."""
    ba_x, ba_y = a[0] - b[0], a[1] - b[1]
    bc_x, bc_y = c[0] - b[0], c[1] - b[1]

    norm_ba = math.hypot(ba_x, ba_y)
    norm_bc = math.hypot(bc_x, bc_y)
    if norm_ba == 0 or norm_bc == 0:
        return 0

    cos_val = (ba_x * bc_x + ba_y * bc_y) / (norm_ba * norm_bc)
    cos_val = max(-1.0, min(1.0, cos_val))
    return int(round(math.degrees(math.acos(cos_val))))

def frame_active_from_landmarks(lm, theta=160, vis_th=0.6, point_th=0.5):
    """TBL: Kiểm tra frame active bằng cách tính góc khuỷu tay."""
    if any(float(lm[idx].visibility) < vis_th for idx in LANDMARKS.values()):
        return 0

    elbow_pairs = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    ]

    angles = []
    for s_idx, e_idx, w_idx in elbow_pairs:
        s_lm, e_lm, w_lm = lm[s_idx], lm[e_idx], lm[w_idx]
        if min(s_lm.visibility, e_lm.visibility, w_lm.visibility) < point_th:
            continue
        angle = _angle_2d((s_lm.x, s_lm.y), (e_lm.x, e_lm.y), (w_lm.x, w_lm.y))
        if angle > 0:
            angles.append(angle)

    if not angles:
        return 0
    
    elbow_angle = int(round(sum(angles) / len(angles)))
    return 1 if elbow_angle < theta else 0

def process_single_front_video(video_id, src_video_path, output_root, theta=160, target_size=224):
    """
    Quy trình siêu tối ưu: Đọc video 1 lần vào RAM, Cache tọa độ xương khớp, 
    Sử dụng chung một instance MediaPipe toàn cục trên mỗi Core.
    """
    try:
        cap = cv2.VideoCapture(src_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 25.0
            
        frames = []          # Lưu toàn bộ các frame vào RAM để Pass 2 dùng luôn
        s_raw = []           # Chuỗi trạng thái active/inactive
        coord_cache = []     # Lưu cache tọa độ phục vụ cho việc tính Bounding Box
        
        # Lấy bộ detector dùng chung của nhân CPU hiện tại
        pose = _get_pose_detector()
        
        # ===== PASS 1: ĐỌC LUỒNG DUY NHẤT & TRÍCH XUẤT TOÀN BỘ DỮ LIỆU =====
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            frames.append(frame) # Giữ lại frame trong bộ nhớ RAM tạm thời
            
            # Xử lý MediaPipe
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.pose_landmarks:
                s_raw.append(0)
                coord_cache.append(None)
                continue
                
            lm = result.pose_landmarks.landmark
            is_active = frame_active_from_landmarks(lm, theta=theta)
            s_raw.append(is_active)
            
            # CACHE TỌA ĐỘ: Chỉ lưu lại những điểm cần thiết cho việc Crop sau này
            coord_cache.append({
                'nose_x': lm[mp_pose.PoseLandmark.NOSE].x,
                'nose_y': lm[mp_pose.PoseLandmark.NOSE].y,
                'l_sh_x': lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                'r_sh_x': lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
            })
        cap.release()
        
        total_frames = len(frames)
        if total_frames == 0:
            return f"⏩ Skip ID {video_id}: Video hỏng hoặc không có frame."
            
        # Trích xuất các đoạn hành động liên tục
        segments = []
        i = 0
        while i < total_frames:
            if s_raw[i] == 0:
                i += 1
                continue
            j = i
            while j < total_frames and s_raw[j] == 1: j += 1
            segments.append((i, j - 1))
            i = j
            
        # Gộp các đoạn quá gần nhau (< 0.8 giây) để tránh nhiễu bẻ đôi video
        max_gap_frames = int(round(fps * 0.8)) 
        merged_segments = []
        
        if segments:
            current_start, current_end = segments[0]
            for next_start, next_end in segments[1:]:
                if next_start - current_end <= max_gap_frames:
                    current_end = next_end
                else:
                    merged_segments.append((current_start, current_end))
                    current_start, current_end = next_start, next_end
            merged_segments.append((current_start, current_end))
        else:
            merged_segments = []

        # Lọc các đoạn đạt chuẩn thời lượng tối thiểu t_min = 0.67 giây 
        valid_segments = []
        for start_idx, end_idx in merged_segments:
            duration = ((end_idx + 1) - start_idx) / fps
            if duration >= 0.67:
                valid_segments.append((start_idx, end_idx))
                
        if len(valid_segments) == 0:
            return f"⏩ Skip ID {video_id}: Không tìm thấy đoạn active hợp lệ."
            
        elif len(valid_segments) > 1:
            # Video bị đứt gãy thành nhiều phần -> Coi như dữ liệu lỗi, loại bỏ luôn!
            return f"⏩ Skip ID {video_id}: Phát hiện {len(valid_segments)} phân đoạn (nghi ngờ lỗi/nhiễu) -> Đã loại bỏ."
        
        # Lúc này chắc chắn len(valid_segments) == 1 (Video hoàn hảo)
        start_frame, end_frame = valid_segments[0]

        # ===== PASS 2: CẮT KHÔNG GIAN (CROP) & RESIZE TRỰC TIẾP TỪ RAM =====
        os.makedirs(output_root, exist_ok=True)
        h, w, _ = frames[0].shape
        
        for seg_idx, (start_frame, end_frame) in enumerate(valid_segments):
            mid_frame_idx = (start_frame + end_frame) // 2
            padding_frames = int(round(fps * 0.4))
            
            # Áp dụng nới rộng khoảng thời gian trước và sau hành động
            start_frame = max(0, start_frame - padding_frames)
            end_frame = min(total_frames - 1, end_frame + padding_frames)
            
            # THAY ĐỔI LỚN: Lấy tọa độ trực tiếp từ cache đã tính ở Pass 1
            mid_coords = coord_cache[mid_frame_idx]
            
            box_size = min(h, w)
            x1, y1 = (w - box_size) // 2, (h - box_size) // 2
            
            if mid_coords is not None:
                pixel_nose_x = int(mid_coords['nose_x'] * w)
                pixel_l_sh_x = int(mid_coords['l_sh_x'] * w)
                pixel_r_sh_x = int(mid_coords['r_sh_x'] * w)
                pixel_nose_y = int(mid_coords['nose_y'] * h)
                
                shoulder_width = abs(pixel_l_sh_x - pixel_r_sh_x)
                box_size = int(shoulder_width * 3.6) 
                box_size = min(box_size, min(h, w))
                
                x1 = pixel_nose_x - box_size // 2
                y1 = pixel_nose_y - int(shoulder_width * 0.6)
            
            if y1 < 0: y1 = 0
            if y1 + box_size > h: y1 = h - box_size
            if x1 < 0: x1 = 0
            if x1 + box_size > w: x1 = w - box_size
            
            # Ghi video mới
            # Nếu chỉ có 1 đoạn active duy nhất -> Lưu thẳng không cần đuôi
            if len(valid_segments) == 1:
                out_path = os.path.join(output_root, f"{video_id}.mp4")
            else:
            # Nếu có nhiều đoạn -> Đánh số phân mảnh _00, _01... để không bị đè file
                out_path = os.path.join(output_root, f"{video_id}_{seg_idx:02d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_writer = cv2.VideoWriter(out_path, fourcc, fps, (target_size, target_size))
            
            # Lấy trực tiếp từ mảng frames trong RAM, tốc độ xử lý nhanh như chớp
            for f_idx in range(start_frame, end_frame + 1):
                frame = frames[f_idx]
                cropped = frame[y1:y1+box_size, x1:x1+box_size]
                resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
                out_writer.write(resized)
                
            out_writer.release()
            
        return f"✅ Đã xử lý thành công ID: {video_id}"
    except Exception as e:
        return f"❌ Lỗi tại ID {video_id}: {str(e)}"

def video_worker(video_path):
    """Hàm worker trung gian cho ProcessPoolExecutor"""
    try:
        output_root = "..\\VSL_FULL_FRONT_CROPPED_TO224x224_V2"
        parent_dir = os.path.basename(os.path.dirname(video_path))
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        class_output_dir = os.path.join(output_root, parent_dir)
        
        res = process_single_front_video(video_id, video_path, class_output_dir)
        return res
    except Exception as e:
        return f"❌ Lỗi hệ thống tại {video_path}: {str(e)}"