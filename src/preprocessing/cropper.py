"""
Module cat khung hinh (Spatial Crop) va co dan ve kich thuoc chuan 224x224.
"""

import os
from pathlib import Path
from typing import Union
import cv2
import mediapipe as mp
from .tbl import frame_active_from_landmarks

mp_pose = mp.solutions.pose
_GLOBAL_POSE_DETECTOR = None


def get_pose_detector():
    """Khoi tao singleton instance cho MediaPipe Pose tren moi worker thread/process."""
    global _GLOBAL_POSE_DETECTOR
    if _GLOBAL_POSE_DETECTOR is None:
        _GLOBAL_POSE_DETECTOR = mp_pose.Pose(static_image_mode=False, model_complexity=1)
    return _GLOBAL_POSE_DETECTOR


def process_single_front_video(
    video_id: str,
    src_video_path: Union[str, Path],
    output_root: Union[str, Path],
    theta: int = 160,
    target_size: int = 224,
    shoulder_crop_ratio: float = 3.6,
    padding_sec: float = 0.4,
    min_duration_sec: float = 0.67,
    max_gap_sec: float = 0.8,
) -> str:
    """
    Quy trinh 2-pass:
    - Pass 1: Quet video, xac dinh cac doan chuyen dong active (TBL)
    - Pass 2: Tinh Bounding Box co the va cat ve target_size x target_size
    """
    try:
        src_video_path = str(src_video_path)
        output_root = str(output_root)

        cap = cv2.VideoCapture(src_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        frames = []
        s_raw = []
        coord_cache = []

        pose = get_pose_detector()

        # Pass 1: Doc luong video va phan tich chuyen dong
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not result.pose_landmarks:
                s_raw.append(0)
                coord_cache.append(None)
                continue

            lm = result.pose_landmarks.landmark
            is_active = frame_active_from_landmarks(lm, theta=theta)
            s_raw.append(is_active)

            coord_cache.append({
                "nose_x": lm[mp_pose.PoseLandmark.NOSE].x,
                "nose_y": lm[mp_pose.PoseLandmark.NOSE].y,
                "l_sh_x": lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                "r_sh_x": lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
            })
        cap.release()

        total_frames = len(frames)
        if total_frames == 0:
            return f"Bo qua ID {video_id}: Video khong co frame hoac loi tep."

        # Xac dinh cac phan doan active lien tiep
        segments = []
        i = 0
        while i < total_frames:
            if s_raw[i] == 0:
                i += 1
                continue
            j = i
            while j < total_frames and s_raw[j] == 1:
                j += 1
            segments.append((i, j - 1))
            i = j

        # Gop cac doan gan nhau
        max_gap_frames = int(round(fps * max_gap_sec))
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

        # Loc theo thoi luong toi thieu
        valid_segments = []
        for start_idx, end_idx in merged_segments:
            duration = ((end_idx + 1) - start_idx) / fps
            if duration >= min_duration_sec:
                valid_segments.append((start_idx, end_idx))

        if len(valid_segments) == 0:
            return f"Bo qua ID {video_id}: Khong tim thay doan ky hieu hop le."
        if len(valid_segments) > 1:
            return f"Bo qua ID {video_id}: Phat hien {len(valid_segments)} phan doan (loi do nhieu)."

        os.makedirs(output_root, exist_ok=True)
        h, w, _ = frames[0].shape

        # Pass 2: Spatial crop & resize
        start_frame, end_frame = valid_segments[0]
        mid_frame_idx = (start_frame + end_frame) // 2
        padding_frames = int(round(fps * padding_sec))

        start_frame = max(0, start_frame - padding_frames)
        end_frame = min(total_frames - 1, end_frame + padding_frames)

        mid_coords = coord_cache[mid_frame_idx]
        box_size = min(h, w)
        x1, y1 = (w - box_size) // 2, (h - box_size) // 2

        if mid_coords is not None:
            pixel_nose_x = int(mid_coords["nose_x"] * w)
            pixel_l_sh_x = int(mid_coords["l_sh_x"] * w)
            pixel_r_sh_x = int(mid_coords["r_sh_x"] * w)
            pixel_nose_y = int(mid_coords["nose_y"] * h)

            shoulder_width = abs(pixel_l_sh_x - pixel_r_sh_x)
            box_size = int(shoulder_width * shoulder_crop_ratio)
            box_size = min(box_size, min(h, w))

            x1 = pixel_nose_x - box_size // 2
            y1 = pixel_nose_y - int(shoulder_width * 0.6)

        if y1 < 0:
            y1 = 0
        if y1 + box_size > h:
            y1 = h - box_size
        if x1 < 0:
            x1 = 0
        if x1 + box_size > w:
            x1 = w - box_size

        out_path = os.path.join(output_root, f"{video_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (target_size, target_size))

        for f_idx in range(start_frame, end_frame + 1):
            frame = frames[f_idx]
            cropped = frame[y1 : y1 + box_size, x1 : x1 + box_size]
            resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)
            out_writer.write(resized)

        out_writer.release()
        return f"Thanh cong ID: {video_id}"
    except Exception as e:
        return f"Loi tai ID {video_id}: {str(e)}"
