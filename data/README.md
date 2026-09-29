# Cấu Trúc Quản Lý Dữ Liệu (Data Directory Structure)

## 1. TL;DR & Quy Ước Lưu Trữ

Thư mục `data/` tổ chức dữ liệu theo từng giai đoạn trong pipeline VSL. Toàn bộ file dung lượng lớn (`.mp4`, `.npy`, `.json` lớn) đều được tự động loại trừ bởi `.gitignore` nhằm bảo vệ git repository gọn nhẹ và sạch sẽ.

---

## 2. Directory Layout & Flow

```
data/
├── raw_splits/                 # Chứa các tập split_1, split_2,... (dataset gốc từ tác giả)
├── merged/                     # Tập hợp nhất các view và file JSON metadata gốc
├── categorized/                # Video front_view đã gom theo từng thư mục gloss
├── preprocessed_224/           # Video sạch sau TBL và crop chuẩn 224×224 px
├── preprocessed_metadata.json  # Metadata thực tế sau tiền xử lý (FPS, Resolution, Frames)
├── signer_splited/             # Dữ liệu phân tách theo Signer ID
│   ├── train/                  # Tập huấn luyện (~80% số signers)
│   ├── test/                   # Tập kiểm thử độc lập (~20% số signers - Unseen)
│   └── global_signer_split.tsv # Bảng ánh xạ Signer ID <-> Train/Test
└── keypoints/                  # Ma trận 76 keypoints 3D dạng .npy [T, 76, 3] phân theo gloss
```

---

## 3. Data Flow & Stage Verification

| Thư mục | Nguồn sinh ra | Định dạng | Mục đích sử dụng |
| :--- | :--- | :--- | :--- |
| `raw_splits/` | Giải nén ban đầu | MP4 (Gốc đa phân giải) | Dữ liệu thô ban đầu |
| `categorized/` | Bước 01 Collection | MP4 gom theo class | Chuẩn bị cho tiền xử lý |
| `preprocessed_224/` | Bước 02 Preprocessing | MP4 224×224 px | Video đã cắt tĩnh và chuẩn kích thước |
| `signer_splited/` | Bước 01/02 Splitting | MP4 chia Train/Test | Đảm bảo Unseen Signer evaluation |
| `keypoints/` | Bước 03 Feature Extraction | NumPy `.npy` `float32` | Huấn luyện trực tiếp trên mô hình GCN / Transformer |
