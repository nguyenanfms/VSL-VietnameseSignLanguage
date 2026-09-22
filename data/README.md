# Huong Dan To Chuc Thu Muc Du Lieu (Data Directory)

Thu muc `data/` duoc thiet ke de luu tru du lieu trong suot qua trinh xu ly pipeline VSL-400. Cac tep video va ma tran `.npy` nang deu duoc cau hinh trong `.gitignore` de tranh day len repository Git.

---

## Cau Truc De Xuat

```
data/
├── raw_splits/                 # Chua cac thu muc split_1, split_2,... (tu dataset goc)
├── merged/                     # Ket qua sau khi gop: front_view/, left_view/, right_view/ va cac file .json
├── categorized/                # Video front_view duoc phan loai vao tung thu muc theo ten gloss
├── preprocessed_224/           # Video da qua TBL va crop ve 224x224
├── preprocessed_metadata.json  # Metadata thuc te sau tien xu ly
├── signer_splited/             # Du lieu chia train/test theo signer
│   ├── train/
│   ├── test/
│   └── global_signer_split.tsv
└── keypoints/                  # Ma tran 76 keypoints 3D .npy phan theo gloss
```

---

## Cac Buoc Tai Va Dat Du Lieu

1. Dat dataset VSL-400 goc vao `data/raw_splits/` hoac chi dinh duong dan tuy chon qua doi so dong lenh `--splits-root`.
2. Chay pipeline theo thu tu qua cac script trong `scripts/` hoac qua cac notebook trong `notebooks/`.
