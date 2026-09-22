"""
Module ho tro gop nhieu phan doan (splits) thanh mot dataset thong nhat.
"""

import os
import re
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Union

VIEWS: Tuple[str, ...] = ("front_view", "left_view", "right_view")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def find_splits(root: Union[str, Path]) -> List[Path]:
    """Tim tat ca cac thu muc split_* trong thu muc goc."""
    root = Path(root)
    splits = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("split_")]
    splits.sort(key=lambda p: int(re.sub(r"[^0-9]", "", p.name) or 0))
    return splits


def copy_file(src: Union[str, Path], dst: Union[str, Path], mode: str = "copy", overwrite: bool = False) -> None:
    """Sao chep file theo che do chi dinh (copy, hardlink hoac symlink)."""
    dst = Path(dst)
    src = Path(src)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        return

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        try:
            if dst.exists():
                dst.unlink()
            os.link(src, dst)
        except Exception:
            shutil.copy2(src, dst)
    elif mode == "symlink":
        try:
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)
        except Exception:
            shutil.copy2(src, dst)


def merge_json_lists(json_paths: List[Union[str, Path]], stop_on_dup: bool = False) -> List[dict]:
    """Gop nhieu file JSON metadata thanh mot danh sach duy nhat."""
    merged = []
    seen_ids = set()

    for jp in json_paths:
        jp = Path(jp)
        if not jp.exists():
            continue
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Loi doc file JSON {jp}: {e}")
            continue

        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                vid = None
                for k in ("video_id", "id", "name", "videoid"):
                    if k in item:
                        vid = str(item[k])
                        break
                if vid is not None:
                    if stop_on_dup and vid in seen_ids:
                        raise RuntimeError(f"Phat hien trung lap video_id: {vid} tai {jp}")
                    if vid in seen_ids:
                        continue
                    seen_ids.add(vid)
                merged.append(item)

    return merged


def merge_dataset(
    root: Union[str, Path],
    out_dir: Union[str, Path],
    copy_mode: str = "copy",
    overwrite: bool = False,
    stop_on_dup: bool = False,
) -> None:
    """Gop tat ca cac thu muc split_* vao thu muc out_dir."""
    root = Path(root)
    out_dir = Path(out_dir)

    splits = find_splits(root)
    if not splits:
        print(f"Khong tim thay thu muc split_* nao trong {root}")
        return

    print(f"Tim thay {len(splits)} splits: {[s.name for s in splits]}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gop JSON metadata tung goc nhin (view)
    for view in VIEWS:
        json_paths = [s / f"{view}.json" for s in splits]
        merged_list = merge_json_lists(json_paths, stop_on_dup=stop_on_dup)
        out_json = out_dir / f"{view}.json"
        out_json.write_text(json.dumps(merged_list, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  Gop {view}: {len(merged_list)} muc metadata -> {out_json.name}")

    # Gop cac file video
    for view in VIEWS:
        out_view_dir = out_dir / view
        count = 0
        for s in splits:
            view_dir = s / view
            if not view_dir.exists():
                continue
            for src in view_dir.iterdir():
                if not src.is_file() or src.suffix.lower() not in VIDEO_EXTS:
                    continue
                dst = out_view_dir / src.name
                copy_file(src, dst, mode=copy_mode, overwrite=overwrite)
                count += 1
        print(f"  Gop video {view}: {count} files")

    print(f"Gop hoan tat tai: {out_dir}")
