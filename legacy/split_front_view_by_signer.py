"""

Split front_view videos into train/test by signer globally (~80% / ~20%).



Each signer_id appears in ONLY train or ONLY test across all glosses, so test

videos never reuse identities seen during training (true unseen-signer eval).

Uses hard links when possible (same volume), else copies.

"""

from __future__ import annotations



import json

import os

import random

import shutil

from collections import defaultdict

from pathlib import Path



BASE = Path(__file__).resolve().parent

JSON_PATH = BASE / "merged_7splits_hardlink" / "front_view.json"

SRC_VIDEOS = BASE / "merged_7splits_hardlink" / "front_view"

OUT_ROOT = BASE / "data_splited"

RNG_SEED = 42





def train_signer_count(n_signers: int) -> int:

    """~80% train, ~20% test; both sides non-empty when n_signers >= 2."""

    if n_signers <= 1:

        return n_signers

    n_train = round(n_signers * 0.8)

    if n_train >= n_signers:

        n_train = n_signers - 1

    if n_train <= 0:

        n_train = 1

    return n_train





def link_or_copy(src: Path, dst: Path) -> None:

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():

        dst.unlink()

    try:

        os.link(src, dst)

    except OSError:

        shutil.copy2(src, dst)





def main() -> None:

    if OUT_ROOT.exists():

        shutil.rmtree(OUT_ROOT)



    rows = json.loads(JSON_PATH.read_text(encoding="utf-8"))



    gloss_rows: dict[str, list[dict]] = defaultdict(list)

    all_signers: set[str] = set()

    for r in rows:

        gloss_rows[r["gloss"]].append(r)

        all_signers.add(str(r["signer_id"]))



    random.seed(RNG_SEED)

    signers_shuffled = sorted(all_signers)

    random.shuffle(signers_shuffled)



    n_train_g = train_signer_count(len(signers_shuffled))

    train_signers = set(signers_shuffled[:n_train_g])

    test_signers = set(signers_shuffled[n_train_g:])

    assert train_signers.isdisjoint(test_signers)

    assert train_signers | test_signers == all_signers



    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    global_split_path = OUT_ROOT / "global_signer_split.tsv"

    global_lines = ["signer_id\tsplit"]

    for s in sorted(train_signers):

        global_lines.append(f"{s}\ttrain")

    for s in sorted(test_signers):

        global_lines.append(f"{s}\ttest")

    global_split_path.write_text("\n".join(global_lines) + "\n", encoding="utf-8")



    train_total = test_total = 0

    meta_lines = [

        "gloss\tn_signers_in_gloss\tn_train_signers_hit\tn_test_signers_hit\ttrain_video_count\ttest_video_count"

    ]



    for gloss in sorted(gloss_rows.keys()):

        items = gloss_rows[gloss]

        gloss_signers = {str(r["signer_id"]) for r in items}

        hit_train = gloss_signers & train_signers

        hit_test = gloss_signers & test_signers



        train_dir = OUT_ROOT / "train" / gloss

        test_dir = OUT_ROOT / "test" / gloss



        n_vid_train = n_vid_test = 0

        for r in items:

            vid = str(r["video_id"])

            sid = str(r["signer_id"])

            src = SRC_VIDEOS / f"{vid}.mp4"

            if not src.is_file():

                raise FileNotFoundError(src)



            if sid in train_signers:

                link_or_copy(src, train_dir / f"{vid}.mp4")

                n_vid_train += 1

            elif sid in test_signers:

                link_or_copy(src, test_dir / f"{vid}.mp4")

                n_vid_test += 1

            else:

                raise RuntimeError(f"Signer {sid} not in global split")



        train_total += n_vid_train

        test_total += n_vid_test

        meta_lines.append(

            f"{gloss}\t{len(gloss_signers)}\t{len(hit_train)}\t{len(hit_test)}\t{n_vid_train}\t{n_vid_test}"

        )



    meta_path = OUT_ROOT / "split_manifest.tsv"

    meta_path.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")



    train_per_gloss_lines = ["gloss\tn_train_signers\tsigner_ids_sorted"]

    for gloss in sorted(gloss_rows.keys()):

        g_train = {

            str(r["signer_id"])

            for r in gloss_rows[gloss]

            if str(r["signer_id"]) in train_signers

        }

        sids = sorted(g_train, key=lambda x: int(x))

        train_per_gloss_lines.append(f'{gloss}\t{len(sids)}\t{",".join(sids)}')

    (OUT_ROOT / "train_signers_per_gloss.tsv").write_text(

        "\n".join(train_per_gloss_lines) + "\n",

        encoding="utf-8",

    )



    print(f"Global signers: {len(all_signers)} ({len(train_signers)} train, {len(test_signers)} test)")

    print(f"Done. Train videos: {train_total}, test videos: {test_total}")

    print(f"Signer list: {global_split_path}")

    print(f"Train signers per gloss: {OUT_ROOT / 'train_signers_per_gloss.tsv'}")

    print(f"Manifest: {meta_path}")





if __name__ == "__main__":

    main()


