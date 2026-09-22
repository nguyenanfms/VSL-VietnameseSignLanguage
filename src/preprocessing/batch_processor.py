"""
Module ho tro tien xu ly hang loat (Batch Processing) su dung ProcessPoolExecutor da nhan CPU.
"""

import os
import glob
import concurrent.futures
from pathlib import Path
from typing import Union
from tqdm import tqdm
from .cropper import process_single_front_video


def _video_worker_task(args):
    video_path, output_root, theta, target_size = args
    try:
        parent_dir = os.path.basename(os.path.dirname(video_path))
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        class_output_dir = os.path.join(output_root, parent_dir)
        return process_single_front_video(
            video_id=video_id,
            src_video_path=video_path,
            output_root=class_output_dir,
            theta=theta,
            target_size=target_size,
        )
    except Exception as e:
        return f"Loi worker tai {video_path}: {str(e)}"


def process_video_batch(
    input_root: Union[str, Path],
    output_root: Union[str, Path],
    theta: int = 160,
    target_size: int = 224,
    max_workers: int = None,
) -> dict:
    """
    Quet toan bo input_root/*/*.mp4 va chay crop/resize song song da nhan.
    """
    input_root = str(input_root)
    output_root = str(output_root)

    all_videos = glob.glob(os.path.join(input_root, "*", "*.mp4"))
    if max_workers is None:
        max_workers = os.cpu_count() or 4

    print(f"Khoi dong xu ly da tien trinh CPU: {max_workers} cores")
    print(f"Tong so video can xu ly: {len(all_videos)}")

    if not all_videos:
        print("Khong tim thay video nao trong thu muc dau vao.")
        return {"success": 0, "skip": 0, "error": 0, "total": 0}

    stats = {"success": 0, "skip": 0, "error": 0, "total": len(all_videos)}
    worker_args = [(p, output_root, theta, target_size) for p in all_videos]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_video_worker_task, arg): arg[0] for arg in worker_args}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Dang xu ly video"):
            try:
                res = future.result()
                if "Thanh cong" in res:
                    stats["success"] += 1
                elif "Bo qua" in res:
                    stats["skip"] += 1
                else:
                    stats["error"] += 1
            except Exception:
                stats["error"] += 1

    print("=" * 50)
    print(f"Tong so video quet:     {stats['total']}")
    print(f"Da crop thanh cong:     {stats['success']}")
    print(f"Bi bo qua:              {stats['skip']}")
    print(f"Loi:                    {stats['error']}")
    print("=" * 50)

    return stats
