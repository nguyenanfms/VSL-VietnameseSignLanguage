# Hướng Dẫn Tổ Chức Thư Mục Dữ Liệu (Data Directory)

Thư mục `data/` được thiết kế để lưu trữ dữ liệu trong suốt quá trình xử lý pipeline VSL-400. Các tệp video và ma trận `.npy` dung lượng lớn đều được cấu hình trong `.gitignore` để tránh bị đẩy lên repository Git.

---

## Cấu Trúc Đề Xuất

```
data/
├── raw_splits/                 # Chứa các thư mục split_1, split_2,... (từ dataset gốc)
├── merged/                     # Kết quả sau khi gộp: front_view/, left_view/, right_view/ và các file .json
├── categorized/                # Video front_view được phân loại vào từng thư mục theo tên gloss
├── preprocessed_224/           # Video đã qua TBL và crop về kích thước 224x224
├── preprocessed_metadata.json  # Metadata thực tế sau tiền xử lý
├── signer_splited/             # Dữ liệu chia train/test theo signer
│   ├── train/
│   ├── test/
│   └── global_signer_split.tsv
└── keypoints/                  # Ma trận 76 keypoints 3D .npy phân theo gloss
```

---

## Các Bước Tải Và Đặt Dữ Liệu

1. Đặt dataset VSL-400 gốc vào `data/raw_splits/` hoặc chỉ định đường dẫn tùy chọn qua đối số dòng lệnh `--splits-root`.
2. Chạy pipeline theo thứ tự qua các script trong `scripts/` hoặc qua các notebook trong `notebooks/`.
