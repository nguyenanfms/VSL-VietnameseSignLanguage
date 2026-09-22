"""
Module truc quan hoa skeleton tu file keypoints .npy thanh video animation 3 panels:
1. Full Body Norm (3D)
2. Left Hand Norm
3. Right Hand Norm
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Union
from .keypoints import LANDMARKS

BODY_CONNECTIONS = [
    ("nose", "neck"), ("neck", "rightShoulder"), ("neck", "leftShoulder"),
    ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
    ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist"),
    ("rightShoulder", "leftShoulder"),
    ("leftShoulder", "leftHip"), ("rightShoulder", "rightHip"),
    ("leftHip", "rightHip"),
    ("leftHip", "leftKnee"), ("leftKnee", "leftAnkle"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
]

FINGER_CHAINS = [
    ["wrist", "thumbCMC", "thumbMP", "thumbIP", "thumbTip"],
    ["wrist", "indexMCP", "indexPIP", "indexDIP", "indexTip"],
    ["wrist", "middleMCP", "middlePIP", "middleDIP", "middleTip"],
    ["wrist", "ringMCP", "ringPIP", "ringDIP", "ringTip"],
    ["wrist", "littleMCP", "littlePIP", "littleDIP", "littleTip"],
]


def create_normalized_motion_video(
    npy_path: Union[str, Path],
    out_video_path: Union[str, Path],
    fps: int = 25,
    panel_size: int = 300,
) -> bool:
    """Tao video animation 3-panel tu ma tran toa do chuan hoa .npy."""
    npy_path = Path(npy_path)
    out_video_path = Path(out_video_path)

    if not npy_path.exists():
        return False

    lm_idx = {name: i for i, name in enumerate(LANDMARKS)}
    data = np.load(str(npy_path))
    total_frames = len(data)
    s = panel_size

    out_video_path.parent.mkdir(parents=True, exist_ok=True)
    out_writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (s * 3, s))

    def to_pixels(name, kp_frame):
        if name not in lm_idx:
            return None
        x, y, _ = kp_frame[lm_idx[name]]
        if abs(x - 0.5) < 1e-4 and abs(y - 0.5) < 1e-4:
            return None
        return (int(x * (s - 30)) + 15, int(y * (s - 30)) + 15)

    for t in range(total_frames):
        kp_frame = data[t] + 0.5

        img_body = np.zeros((s, s, 3), dtype=np.uint8)
        img_left_hand = np.zeros((s, s, 3), dtype=np.uint8)
        img_right_hand = np.zeros((s, s, 3), dtype=np.uint8)

        # Ve skeleton co the
        for a, b in BODY_CONNECTIONS:
            pa = to_pixels(a, kp_frame)
            pb = to_pixels(b, kp_frame)
            if pa and pb:
                cv2.line(img_body, pa, pb, (0, 255, 0), 2)
                cv2.circle(img_body, pa, 4, (0, 0, 255), -1)

        # Ve skeleton ban tay
        for suffix, canvas, color in [("_0", img_left_hand, (255, 165, 0)), ("_1", img_right_hand, (0, 165, 255))]:
            for chain in FINGER_CHAINS:
                for i in range(len(chain) - 1):
                    pa = to_pixels(chain[i] + suffix, kp_frame)
                    pb = to_pixels(chain[i + 1] + suffix, kp_frame)
                    if pa and pb:
                        cv2.line(canvas, pa, pb, color, 2)
                        cv2.circle(canvas, pa, 3, (255, 255, 255), -1)

        cv2.putText(img_body, "1. Full Body Norm (3D)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img_left_hand, "2. Left Hand Norm", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        cv2.putText(img_right_hand, "3. Right Hand Norm", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        combined = np.hstack((img_body, img_left_hand, img_right_hand))
        out_writer.write(combined)

    out_writer.release()
    return True
