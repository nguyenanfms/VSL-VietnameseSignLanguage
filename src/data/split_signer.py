"""
Module chia tap du lieu Train / Test theo Signer ID (danh gia unseen-signer).
"""

import os
import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Union
from tqdm import tqdm


def train_signer_count(n_signers: int, train_ratio: float = 0.8) -> int:
    """Tinh so luong signer duoc phan bo cho tap train."""
    if n_signers <= 1:
        return n_signers
    n_train = round(n_signers * train_ratio)
    if n_train >= n_signers:
        n_train = n_signers - 1
    if n_train <= 0:
        n_train = 1
    return n_train


def link_or_copy(src: Path, dst: Path) -> None:
    """Uu tien tao hard link, neu that bai se sao chep file."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def split_by_signer(
    json_path: Union[str, Path],
    src_videos_dir: Union[str, Path],
    output_root: Union[str, Path],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> dict:
    """
    Chia du lieu theo Signer ID de dam bao moi signer chi xuat hien
    trong duy nhat mot tap (train hoac test).
    """
    json_path = Path(json_path)
    src_videos_dir = Path(src_videos_dir)
    output_root = Path(output_root)

    rows = json.loads(json_path.read_text(encoding="utf-8"))

    gloss_rows = defaultdict(list)
    all_signers = set()
    for r in rows:
        gloss_rows[r["gloss"]].append(r)
        all_signers.add(str(r["signer_id"]))

    random.seed(seed)
    signers_shuffled = sorted(all_signers)
    random.shuffle(signers_shuffled)

    n_train_g = train_signer_count(len(signers_shuffled), train_ratio=train_ratio)
    train_signers = set(signers_shuffled[:n_train_g])
    test_signers = set(signers_shuffled[n_train_g:])

    assert train_signers.isdisjoint(test_signers), "Xay ra trung lap signer giua train va test!"

    print(f"Tong so nguoi ky (signers): {len(all_signers)}")
    print(f"  Tap Train: {len(train_signers)} signers")
    print(f"  Tap Test:  {len(test_signers)} signers")

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    global_split_path = output_root / "global_signer_split.tsv"
    global_lines = ["signer_id\tsplit"]
    for s in sorted(train_signers):
        global_lines.append(f"{s}\ttrain")
    for s in sorted(test_signers):
        global_lines.append(f"{s}\ttest")
    global_split_path.write_text("\n".join(global_lines) + "\n", encoding="utf-8")

    train_total = 0
    test_total = 0

    for gloss in tqdm(sorted(gloss_rows.keys()), desc="Phan bo video"):
        items = gloss_rows[gloss]
        for r in items:
            vid = str(r["video_id"])
            sid = str(r["signer_id"])
            src = src_videos_dir / f"{vid}.mp4"

            if not src.is_file():
                continue

            if sid in train_signers:
                dst_dir = output_root / "train" / gloss
                link_or_copy(src, dst_dir / f"{vid}.mp4")
                train_total += 1
            elif sid in test_signers:
                dst_dir = output_root / "test" / gloss
                link_or_copy(src, dst_dir / f"{vid}.mp4")
                test_total += 1

    report = {
        "train_videos": train_total,
        "test_videos": test_total,
        "train_signers": len(train_signers),
        "test_signers": len(test_signers),
        "split_file": str(global_split_path),
    }
    print(f"Hoan tat chia tap du lieu: Train {train_total} video, Test {test_total} video")
    return report
