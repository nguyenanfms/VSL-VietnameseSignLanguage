"""
Script CLI: Thu thap, gop cac splits, phan loai video theo gloss va chia tap train/test theo signer.
"""

import argparse
from pathlib import Path
import sys

# Them root vao sys.path de import src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.merge_splits import merge_dataset
from src.data.categorize import categorize_videos_by_gloss
from src.data.split_signer import split_by_signer


def main():
    parser = argparse.ArgumentParser(description="Giai doan 1: Thu thap va to chuc du lieu VSL-400")
    parser.add_argument("--action", choices=["merge", "categorize", "split", "all"], default="all",
                        help="Thao tac can thuc hien: merge, categorize, split hoac all")
    parser.add_argument("--splits-root", type=str, default="data/raw_splits",
                        help="Thu muc chua cac split_1, split_2, ...")
    parser.add_argument("--merged-dir", type=str, default="data/merged",
                        help="Thu muc luu du lieu sau khi gop")
    parser.add_argument("--categorized-dir", type=str, default="data/categorized",
                        help="Thu muc luu video phan loai theo gloss")
    parser.add_argument("--split-output", type=str, default="data/signer_splited",
                        help="Thu muc luu ket qua chia train/test theo signer")
    parser.add_argument("--json-metadata", type=str, default="data/merged/front_view.json",
                        help="File JSON metadata goc dung de phan loai va chia signer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.action in ("merge", "all"):
        print("[Buoc 1.1] Gop cac phan doan splits...")
        merge_dataset(args.splits_root, args.merged_dir)

    if args.action in ("categorize", "all"):
        print("[Buoc 1.2] Phan loai video theo thu muc gloss...")
        src_front_videos = Path(args.merged_dir) / "front_view"
        if src_front_videos.exists() and Path(args.json_metadata).exists():
            categorize_videos_by_gloss(args.json_metadata, src_front_videos, args.categorized_dir)
        else:
            print("Khong tim thay du lieu gop de phan loai gloss. Bo qua buoc nay.")

    if args.action in ("split", "all"):
        print("[Buoc 1.3] Chia train/test theo signer ID...")
        src_front_videos = Path(args.merged_dir) / "front_view"
        if src_front_videos.exists() and Path(args.json_metadata).exists():
            split_by_signer(args.json_metadata, src_front_videos, args.split_output, seed=args.seed)
        else:
            print("Khong tim thay du lieu gop de chia signer. Bo qua buoc nay.")


if __name__ == "__main__":
    main()
