"""
Module thuat toan Temporal Boundary Localization (TBL) dua tren goc khuyu tay.
"""

import math
from typing import Tuple, List, Any
import mediapipe as mp

mp_pose = mp.solutions.pose

LANDMARKS = {
    "LShoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "RShoulder": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "LElbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "RElbow": mp_pose.PoseLandmark.RIGHT_ELBOW,
    "LWrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "RWrist": mp_pose.PoseLandmark.RIGHT_WRIST,
}


def angle_2d(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> int:
    """
    Tinh goc ABC (do) trong mat phang 2D tai dinh B.
    """
    ba_x, ba_y = a[0] - b[0], a[1] - b[1]
    bc_x, bc_y = c[0] - b[0], c[1] - b[1]

    norm_ba = math.hypot(ba_x, ba_y)
    norm_bc = math.hypot(bc_x, bc_y)
    if norm_ba == 0 or norm_bc == 0:
        return 0

    cos_val = (ba_x * bc_x + ba_y * bc_y) / (norm_ba * norm_bc)
    cos_val = max(-1.0, min(1.0, cos_val))
    return int(round(math.degrees(math.acos(cos_val))))


def frame_active_from_landmarks(
    lm: List[Any],
    theta: int = 160,
    vis_th: float = 0.6,
    point_th: float = 0.5,
) -> int:
    """
    Kiem tra mot frame co trang thai dang ky hieu (active) hay khong.
    Frame duoc xem la active khi goc khuyu tay nho hon nguong theta (mac dinh: 160 do).
    """
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
        ang = angle_2d((s_lm.x, s_lm.y), (e_lm.x, e_lm.y), (w_lm.x, w_lm.y))
        if ang > 0:
            angles.append(ang)

    if not angles:
        return 0

    elbow_angle = int(round(sum(angles) / len(angles)))
    return 1 if elbow_angle < theta else 0
