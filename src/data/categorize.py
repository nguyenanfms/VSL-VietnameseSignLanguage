"""
Module phan loai video vao cac thu muc tuong ung theo ten gloss (tu vung ky hieu).
"""

import os
import re
import shutil
from pathlib import Path
from typing import Union
import pandas as pd
from tqdm import tqdm


def safe_dirname(name: str) -> str:
    """Loai bo ky tu dac biet khong hop le trong ten thu muc."""
    return re.sub(r'[\\/:*?"<>|]', "", name).strip()


def categorize_videos_by_gloss(
    json_path: Union[str, Path],
    src_video_dir: Union[str, Path],
    output_root: Union[str, Path],
    ext: str = ".mp4",
) -> dict:
    """
    Phan loai video tu src_video_dir vao output_root/{gloss}/{video_id}.ext
    dua tren file JSON metadata.
    """
    df = pd.read_json(json_path)
    print(f"Tong so video trong metadata: {len(df)}")
    print(f"So luong gloss: {df['gloss'].nunique()}")

    os.makedirs(output_root, exist_ok=True)

    success_count = 0
    error_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Phan loai video"):
        video_id = str(row.get("videoid", row.get("video_id", "")))
        gloss = str(row.get("gloss", "")).strip()

        src_path = os.path.join(src_video_dir, f"{video_id}{ext}")
        if not os.path.exists(src_path):
            error_count += 1
            continue

        gloss_dir = os.path.join(output_root, safe_dirname(gloss))
        os.makedirs(gloss_dir, exist_ok=True)

        dst_path = os.path.join(gloss_dir, f"{video_id}{ext}")
        try:
            shutil.copy2(src_path, dst_path)
            success_count += 1
        except Exception as e:
            error_count += 1
            print(f"Loi sao chep video {video_id}: {e}")

    report = {"success": success_count, "error": error_count}
    print(f"Phan loai thanh cong: {success_count} video, loi: {error_count} video")
    return report
