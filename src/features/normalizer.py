"""
Module chuan hoa toa do keypoints body va hands dua tren bounding box.
"""

from typing import Dict, List, Tuple

BODY_LANDMARKS = [
    "nose", "leftEyeInner", "leftEye", "leftEyeOuter",
    "rightEyeInner", "rightEye", "rightEyeOuter",
    "leftEar", "rightEar", "mouthLeft", "mouthRight",
    "leftShoulder", "rightShoulder",
    "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftPinky", "rightPinky", "leftIndex", "rightIndex",
    "leftThumb", "rightThumb",
    "leftHip", "rightHip", "leftKnee", "rightKnee",
    "leftAnkle", "rightAnkle",
    "leftHeel", "rightHeel", "leftFootIndex", "rightFootIndex",
    "neck",
]

HAND_LANDMARKS = [
    "wrist", "indexTip", "indexDIP", "indexPIP", "indexMCP",
    "middleTip", "middleDIP", "middlePIP", "middleMCP",
    "ringTip", "ringDIP", "ringPIP", "ringMCP",
    "littleTip", "littleDIP", "littlePIP", "littleMCP",
    "thumbTip", "thumbIP", "thumbMP", "thumbCMC",
]


class SingleBodyDictNormalize:
    """
    Chuan hoa toa do body landmarks theo bounding box
    tao boi 6 anchor points (nose, shoulders, hips, neck) voi scale 1.6x.
    Dua toa do ve khoang [-0.5, 0.5].
    """

    ANCHOR_LANDMARKS = [
        "nose", "leftShoulder", "rightShoulder",
        "leftHip", "rightHip", "neck",
    ]

    def __call__(self, row: Dict[str, List[Tuple[float, float, float]]]) -> Dict[str, List[Tuple[float, float, float]]]:
        sequence_size = len(row["leftEar"])

        for i in range(sequence_size):
            x_coords = [row[name][i][0] for name in self.ANCHOR_LANDMARKS if row[name][i][0] != 0]
            y_coords = [row[name][i][1] for name in self.ANCHOR_LANDMARKS if row[name][i][1] != 0]

            if not x_coords or not y_coords:
                continue

            min_x, max_x = min(x_coords), max(x_coords)
            min_y, max_y = min(y_coords), max(y_coords)

            dx = (max_x - min_x) * 1.6
            dy = (max_y - min_y) * 1.6
            if dx <= 0 or dy <= 0:
                continue

            center_x = (max_x + min_x) / 2
            center_y = (max_y + min_y) / 2

            box_min_x = center_x - dx / 2
            box_min_y = center_y - dy / 2

            for key in BODY_LANDMARKS:
                x, y, z = row[key][i]
                if x == 0 and y == 0:
                    continue
                row[key][i] = (
                    (x - box_min_x) / dx - 0.5,
                    (y - box_min_y) / dy - 0.5,
                    z,
                )
        return row


class SingleHandDictNormalize:
    """
    Chuan hoa toa do hand landmarks theo bounding box rieng biet
    cua tung ban tay (tay trai suffix _0, tay phai suffix _1).
    Dua toa do ve khoang [-0.5, 0.5].
    """

    def __call__(self, row: Dict[str, List[Tuple[float, float, float]]]) -> Dict[str, List[Tuple[float, float, float]]]:
        sequence_size = len(row["leftEar"])

        for suffix in ["_0", "_1"]:
            for i in range(sequence_size):
                x_coords = [row[name + suffix][i][0] for name in HAND_LANDMARKS if row[name + suffix][i][0] != 0]
                y_coords = [row[name + suffix][i][1] for name in HAND_LANDMARKS if row[name + suffix][i][1] != 0]

                if not x_coords or not y_coords:
                    continue

                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                dx, dy = max_x - min_x, max_y - min_y

                if dx <= 0 or dy <= 0:
                    continue

                for name in HAND_LANDMARKS:
                    x, y, z = row[name + suffix][i]
                    if x == 0 and y == 0:
                        continue
                    row[name + suffix][i] = (
                        (x - min_x) / dx - 0.5,
                        (y - min_y) / dy - 0.5,
                        z,
                    )
        return row
