# Jupyter Notebooks: VSL Pipeline Execution Guide

## 1. TL;DR & Pipeline Overview

Bộ 3 Jupyter Notebooks tương tác trực quan hóa và kiểm thử từng giai đoạn trong pipeline nhận diện ngôn ngữ ký hiệu VSL-400: từ thu thập, gộp splits, làm sạch video bằng Temporal Boundary Localization (TBL) đến trích xuất 76 điểm keypoints 3D chuẩn hóa.

```
[01_data_collection.ipynb] ──> [02_data_cleaning_and_imputation.ipynb] ──> [03_exploratory_data_analysis.ipynb]
(Gộp splits & Chia Signer)        (TBL Elbow Angle & Crop 224x224)              (76 Keypoints 3D & EDA)
```

---

## 2. Notebook Execution & Technical Rationale

### 01_data_collection.ipynb
- **Nhiệm vụ:** Hợp nhất các tập phân đoạn (splits), phân loại video theo gloss và chia train/test theo `signer_id`.
- **Tech Rationale:** Sử dụng phân chia **Subject-Independent (Unseen Signer)** với tỷ lệ ~80/20 nhằm triệt tiêu hoàn toàn data leakage, bảo đảm mô hình học biểu diễn ký hiệu chứ không học vẹt đặc điểm hình thể của người ký.

### 02_data_cleaning_and_imputation.ipynb
- **Nhiệm vụ:** Định vị biên thời gian (TBL) loại bỏ khung hình tĩnh đầu/cuối, cắt vùng quan tâm (ROI) quanh cơ thể và resize về kích thước chuẩn `224×224`. Quét và xuất file JSON metadata thực tế của tập video sau xử lý.
- **Tech Rationale:** Dùng heuristic góc khuỷu tay ($\theta < 160^\circ$) từ MediaPipe Pose để lọc khung hình thay vì model action detection phức tạp; cắt khung hình động theo $3.6 \times$ khoảng cách hai vai giúp loại bỏ phông nền dư thừa mà không làm mất biên độ tay khi ký.

### 03_exploratory_data_analysis.ipynb
- **Nhiệm vụ:** Trích xuất 76 keypoints 3D (34 body + 42 hands), chuẩn hóa tọa độ không gian độc lập về khoảng `[-0.5, 0.5]`, render video animation skeleton 3 panel và thống kê phân bố dữ liệu (EDA).
- **Tech Rationale:** Chuẩn hóa tọa độ dựa trên Bounding Box của cơ thể (scale 1.6×) và bàn tay độc lập giúp mô hình bất biến trước khoảng cách camera và kích cỡ người ký.

---

## 3. Comparison & Stage Benchmark

| Giai đoạn | Dữ liệu đầu vào | Dữ liệu đầu ra | Mức độ tối ưu |
| :--- | :--- | :--- | :--- |
| **01. Collection** | Nhiều splits rời rạc, metadata phân mảnh | Cấu trúc phân loại theo gloss + tập Train/Test theo Signer | Quản lý tập trung, chống leakage |
| **02. Cleaning** | Video thô góc rộng (1280×720, chứa ~46% frame tĩnh) | Video chuẩn hóa 224×224 px + metadata JSON cập nhật | Giảm 10× dung lượng pixel, loại bỏ nhiễu tĩnh |
| **03. Feature & EDA** | Video sạch 224×224 px | Ma trận `.npy` `[T, 76, 3]` + Video skeleton 3 panel | Giảm >99% dung lượng so với video, sẵn sàng cho GCN/Transformer |

---

## 4. Quickstart Execution Guide

### Cài đặt môi trường

```bash
# Kích hoạt môi trường ảo từ thư mục gốc
cd notebooks
pip install -r requirements.txt
jupyter lab
```

### Thứ tự thực thi chuẩn

1. Chạy tuần tự từ cell đầu đến cuối trong `01_data_collection.ipynb`.
2. Kiểm tra các thư mục video đã phân loại trước khi sang `02_data_cleaning_and_imputation.ipynb`.
3. Chạy `02_data_cleaning_and_imputation.ipynb` với `ProcessPoolExecutor` để tiền xử lý hàng loạt video.
4. Chạy `03_exploratory_data_analysis.ipynb` theo từng batch (1→4) để xuất các tệp `.npy` phục vụ huấn luyện.
