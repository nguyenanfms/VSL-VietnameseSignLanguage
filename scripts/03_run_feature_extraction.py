"""
Script CLI: Trich xuat 76 keypoints 3D va truc quan hoa skeleton.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.keypoints import batch_extract_keypoints
from src.features.visualizer import create_normalized_motion_video


def main():
    parser = argparse.ArgumentParser(description="Giai doan 3: Trich xuat 76 keypoints 3D va truc quan hoa")
    parser.add_argument("--input-dir", type=str, default="data/signer_splited/train",
                        help="Thu muc chua cac video da tien xu ly")
    parser.add_argument("--output-dir", type=str, default="data/keypoints",
                        help="Thu muc luu cac file .npy chua keypoints")
    parser.add_argument("--batch-idx", type=int, default=1, help="Chi so batch can chay (1-based)")
    parser.add_argument("--total-batches", type=int, default=4, help="Tong so batch can chia")
    parser.add_argument("--visualize-sample", type=str, default=None,
                        help="Duong dan file .npy de tao video animation skeleton truc quan")
    parser.add_argument("--vis-out", type=str, default="data/sample_skeleton.mp4",
                        help="Duong dan file video animation dau ra")

    args = parser.parse_args()

    if args.visualize_sample:
        print(f"Dang tao video truc quan hoa tu {args.visualize_sample} -> {args.vis_out}")
        create_normalized_motion_video(args.visualize_sample, args.vis_out)
        return

    print(f"Dang chay trich xuat 3D keypoints: batch {args.batch_idx}/{args.total_batches}...")
    batch_extract_keypoints(
        input_dir=args.input_dir,
        output_keypoints_dir=args.output_dir,
        batch_idx=args.batch_idx,
        total_batches=args.total_batches,
    )


if __name__ == "__main__":
    main()
