"""
Module trich xuat 76 diem keypoints 3D su dung MediaPipe Holistic.
Bao gom 34 diem body (co diem neck tong hop) + 42 diem hands (21 moi tay).
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from glob import glob
from typing import Union, List
from tqdm import tqdm
from .normalizer import (
    BODY_LANDMARKS,
    HAND_LANDMARKS,
    SingleBodyDictNormalize,
    SingleHandDictNormalize,
)

HANDS_LANDMARKS = [id + suffix for id in HAND_LANDMARKS for suffix in ["_0", "_1"]]
LANDMARKS = BODY_LANDMARKS + HANDS_LANDMARKS

POSE_MAP = {
    "nose": 0, "leftEyeInner": 1, "leftEye": 2, "leftEyeOuter": 3,
    "rightEyeInner": 4, "rightEye": 5, "rightEyeOuter": 6,
    "leftEar": 7, "rightEar": 8, "mouthLeft": 9, "mouthRight": 10,
    "leftShoulder": 11, "rightShoulder": 12,
    "leftElbow": 13, "rightElbow": 14, "leftWrist": 15, "rightWrist": 16,
    "leftPinky": 17, "rightPinky": 18, "leftIndex": 19, "rightIndex": 20,
    "leftThumb": 21, "rightThumb": 22,
    "leftHip": 23, "rightHip": 24,
    "leftKnee": 25, "rightKnee": 26, "leftAnkle": 27, "rightAnkle": 28,
    "leftHeel": 29, "rightHeel": 30, "leftFootIndex": 31, "rightFootIndex": 32,
}

HAND_MAP = {
    "wrist": 0, "thumbCMC": 1, "thumbMP": 2, "thumbIP": 3, "thumbTip": 4,
    "indexMCP": 5, "indexPIP": 6, "indexDIP": 7, "indexTip": 8,
    "middleMCP": 9, "middlePIP": 10, "middleDIP": 11, "middleTip": 12,
    "ringMCP": 13, "ringPIP": 14, "ringDIP": 15, "ringTip": 16,
    "littleMCP": 17, "littlePIP": 18, "littleDIP": 19, "littleTip": 20,
}


def extract_sign_language_features(
    video_path: Union[str, Path],
    npy_out_path: Union[str, Path],
    holistic_model=None,
) -> bool:
    """
    Trich xuat 76 keypoints 3D tu 1 file video va luu thanh ma tran NumPy [num_frames, 76, 3].
    """
    video_path = str(video_path)
    npy_out_path = Path(npy_out_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    close_holistic = False
    if holistic_model is None:
        holistic_model = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)
        close_holistic = True

    video_sequence_dict = {name: [] for name in LANDMARKS}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic_model.process(rgb_frame)

        pose_data = {name: (0.0, 0.0, 0.0) for name in BODY_LANDMARKS}
        if results.pose_landmarks:
            for name, idx in POSE_MAP.items():
                lm = results.pose_landmarks.landmark[idx]
                pose_data[name] = (lm.x, lm.y, lm.z)

            ls = pose_data["leftShoulder"]
            rs = pose_data["rightShoulder"]
            pose_data["neck"] = (
                (ls[0] + rs[0]) / 2,
                (ls[1] + rs[1]) / 2,
                (ls[2] + rs[2]) / 2,
            )

        for name in BODY_LANDMARKS:
            video_sequence_dict[name].append(pose_data[name])

        for suffix, hand_landmarks in [("_0", results.left_hand_landmarks), ("_1", results.right_hand_landmarks)]:
            hand_data = {name + suffix: (0.0, 0.0, 0.0) for name in HAND_LANDMARKS}
            if hand_landmarks:
                for name, idx in HAND_MAP.items():
                    lm = hand_landmarks.landmark[idx]
                    hand_data[name + suffix] = (lm.x, lm.y, lm.z)
            for name in HAND_LANDMARKS:
                video_sequence_dict[name + suffix].append(hand_data[name + suffix])

    cap.release()
    if close_holistic:
        holistic_model.close()

    body_norm = SingleBodyDictNormalize()
    hand_norm = SingleHandDictNormalize()

    video_sequence_dict = body_norm(video_sequence_dict)
    video_sequence_dict = hand_norm(video_sequence_dict)

    seq_len = len(video_sequence_dict["neck"])
    if seq_len == 0:
        return False

    frames_list = []
    for i in range(seq_len):
        frame_features = [video_sequence_dict[name][i] for name in LANDMARKS]
        frames_list.append(frame_features)

    np_array_data = np.array(frames_list, dtype=np.float32)

    npy_out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy_out_path), np_array_data)
    return True


def batch_extract_keypoints(
    input_dir: Union[str, Path],
    output_keypoints_dir: Union[str, Path],
    batch_idx: int = 1,
    total_batches: int = 1,
) -> None:
    """
    Trich xuat keypoints theo tung batch danh sach class de tranh qua tai bo nho tren may ca nhan/Kaggle.
    """
    input_dir = Path(input_dir)
    output_keypoints_dir = Path(output_keypoints_dir)

    all_classes = sorted([f.name for f in input_dir.iterdir() if f.is_dir()])
    total_classes = len(all_classes)
    if total_classes == 0:
        print("Khong tim thay class nao trong thu muc dau vao.")
        return

    chunks = np.array_split(all_classes, total_batches)
    active_classes = list(chunks[batch_idx - 1])

    print(f"Tong so class: {total_classes}. Dang chay batch {batch_idx}/{total_batches}")
    print(f"So luong class phu trach trong batch nay: {len(active_classes)}")

    holistic_model = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=1)

    for class_name in active_classes:
        class_input = input_dir / class_name
        video_files = glob(str(class_input / "*.mp4"))
        for v_path in tqdm(video_files, desc=f"Batch {batch_idx} -> {class_name}"):
            vid = Path(v_path).stem
            out_npy = output_keypoints_dir / class_name / f"{vid}.npy"
            if out_npy.exists():
                continue
            extract_sign_language_features(v_path, out_npy, holistic_model=holistic_model)

    holistic_model.close()
    print(f"Hoan tat trich xuat toan bo batch {batch_idx}/{total_batches}!")
