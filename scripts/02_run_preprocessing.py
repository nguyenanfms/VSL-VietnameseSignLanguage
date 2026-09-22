"""
Script CLI: Tien xu ly video TBL + Crop 224x224 va trich xuat JSON metadata sau xu ly.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.batch_processor import process_video_batch
from src.preprocessing.metadata import extract_metadata_after_preprocessing


def main():
    parser = argparse.ArgumentParser(description="Giai doan 2: Tien xu ly video VSL-400 (TBL & Spatial Crop)")
    parser.add_argument("--input-dir", type=str, default="data/categorized",
                        help="Thu muc chua cac video dau vao phan theo gloss")
    parser.add_argument("--output-dir", type=str, default="data/preprocessed_224",
                        help="Thu muc luu video sau khi crop va resize ve 224x224")
    parser.add_argument("--json-out", type=str, default="data/preprocessed_metadata.json",
                        help="Duong dan file JSON metadata sau khi tien xu ly")
    parser.add_argument("--theta", type=int, default=160, help="Nguong goc khuyu tay (TBL)")
    parser.add_argument("--target-size", type=int, default=224, help="Kich thuoc vuong video dau ra (pixel)")
    parser.add_argument("--workers", type=int, default=None, help="So luong workers CPU da tien trinh")

    args = parser.parse_args()

    print("[Buoc 2.1] Chay tien xu ly video da nhan CPU...")
    process_video_batch(
        input_root=args.input_dir,
        output_root=args.output_dir,
        theta=args.theta,
        target_size=args.target_size,
        max_workers=args.workers,
    )

    print("[Buoc 2.2] Trich xuat JSON metadata sau tien xu ly...")
    extract_metadata_after_preprocessing(
        preprocessed_dir=args.output_dir,
        output_json_path=args.json_out,
    )


if __name__ == "__main__":
    main()
