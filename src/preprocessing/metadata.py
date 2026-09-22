"""
Module trich xuat metadata video (do phan giai, FPS, duration, num_frames) ra file JSON.
"""

import os
import json
import cv2
import concurrent.futures
from pathlib import Path
from typing import Union, Dict, Any, List
from tqdm import tqdm


def get_video_metadata(video_path: Union[str, Path], gloss: str) -> Dict[str, Any]:
    """Trich xuat thong tin metadata cua 1 video su dung OpenCV."""
    video_path = str(video_path)
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
                "error": "Khong the mo file video",
            }

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
            "duration": duration,
        }
    except Exception as e:
        return {
            "videoid": videoid,
            "fps": 0.0,
            "resolution": "0x0",
            "gloss": gloss,
            "num_frames": 0,
            "duration": 0.0,
            "error": str(e),
        }


def extract_metadata_after_preprocessing(
    preprocessed_dir: Union[str, Path],
    output_json_path: Union[str, Path],
    max_workers: int = 16,
) -> List[Dict[str, Any]]:
    """
    Quet thu muc video da duoc tien xu ly (cat va resize)
    va trich xuat metadata vao file JSON moi.
    """
    preprocessed_dir = Path(preprocessed_dir)
    output_json_path = Path(output_json_path)

    if not preprocessed_dir.exists():
        print(f"Thu muc khong ton tai: {preprocessed_dir}")
        return []

    tasks = []
    video_extensions = (".mp4", ".avi", ".mkv", ".mov", ".flv")

    for root, _, files in os.walk(preprocessed_dir):
        rel_path = os.path.relpath(root, preprocessed_dir)
        if rel_path == ".":
            continue
        gloss = rel_path.split(os.sep)[0]
        for f in files:
            if f.lower().endswith(video_extensions):
                tasks.append((os.path.join(root, f), gloss))

    total_videos = len(tasks)
    print(f"Tim thay {total_videos} video can trich xuat metadata.")
    if total_videos == 0:
        return []

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_video_metadata, path, gloss): (path, gloss) for path, gloss in tasks}
        for future in tqdm(concurrent.futures.as_completed(futures), total=total_videos, desc="Trich xuat metadata"):
            try:
                res = future.result()
                results.append(res)
            except Exception as e:
                path, gloss = futures[future]
                print(f"Loi xu ly {path}: {e}")

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Da luu metadata thanh cong tai: {output_json_path}")
    return results
