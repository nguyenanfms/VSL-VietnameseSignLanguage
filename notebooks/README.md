# Vietnamese Sign Language Recognition — VSL-400

## Giới Thiệu

Dự án xây dựng pipeline xử lý dữ liệu cho hệ thống **Nhận Diện Ngôn Ngữ Ký Hiệu Việt Nam (VSL)** sử dụng dataset **VSL-400** (400 từ vựng ký hiệu). Pipeline bao gồm các Jupyter Notebook được thiết kế trực quan và chi tiết:

- **Thu thập & tổ chức dữ liệu** từ nhiều nguồn (gộp các splits, phân loại theo gloss và chia train/test theo signer).
- **Tiền xử lý video** bằng thuật toán TBL (Temporal Boundary Localization) và Spatial Crop.
- **Trích xuất metadata JSON** cho tập dữ liệu sau tiền xử lý.
- **Trích xuất 76 keypoints 3D** (body + hands) bằng MediaPipe Holistic và phân tích EDA.

---

## Cấu Trúc Thư Mục

```
project/
│
├── data/
│   ├── raw/                    ← Dữ liệu thô gốc (KHÔNG được sửa bằng tay)
│   └── processed/              ← Dữ liệu đã làm sạch (Thành phẩm)
│
├── notebooks/                  ← Thư mục Jupyter Notebooks
│   ├── 01_data_collection.ipynb              ← Thu thập & tổ chức dữ liệu
│   ├── 02_data_cleaning_and_imputation.ipynb ← Tiền xử lý video & Trích xuất JSON
│   ├── 03_exploratory_data_analysis.ipynb    ← Trích xuất keypoints & EDA
│   ├── requirements.txt                      ← Danh sách thư viện (pip)
│   └── README.md                             ← Hướng dẫn này
│
└── ...
```

---

## Yêu Cầu Hệ Thống

| Yêu cầu | Tối thiểu | Khuyến nghị |
|----------|-----------|-------------|
| **Python** | 3.9+ | 3.11+ |
| **RAM** | 8 GB | 16+ GB |
| **CPU** | 4 cores | 8+ cores |
| **Dung lượng ổ cứng** | 50 GB | 100+ GB |
| **GPU** | Không bắt buộc | - |

---

## Cài Đặt

### 1. Tạo môi trường ảo (khuyến nghị)

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt (Windows)
venv\Scripts\activate

# Kích hoạt (Linux/Mac)
source venv/bin/activate
```

### 2. Cài đặt thư viện

```bash
cd notebook
pip install -r requirements.txt
```

### 3. Khởi chạy Jupyter Notebook

```bash
jupyter notebook
```

---

## Hướng Dẫn Chạy Code (A-Z)

### Bước 1: Thu Thập & Tổ Chức Dữ Liệu

**File:** `01_data_collection.ipynb`

**Chức năng:**
- Gộp nhiều phân đoạn (splits) thành một dataset thống nhất.
- Phân loại video vào thư mục theo tên gloss (ký hiệu).
- Chia tập train/test theo signer ID (~80/20) để đảm bảo đánh giá khách quan (unseen-signer evaluation).

**Các biến cấu hình chính:**
```python
SPLITS_ROOT = "."          # Thư mục chứa các split_1, split_2, ...
MERGED_OUTPUT = "merged"   # Thư mục đầu ra của video gộp
```

**Đầu ra:**
| File/Thư mục | Mô tả |
|--------------|-------|
| `merged/front_view.json` | Metadata gốc đã gộp từ các splits |
| `data_splited/train/` | Video thô cho tập huấn luyện |
| `data_splited/test/` | Video thô cho tập kiểm thử |
| `data_splited/global_signer_split.tsv` | Bảng ánh xạ signer ↔ train/test |

---

### Bước 2: Tiền Xử Lý Video & Trích Xuất Metadata

**File:** `02_data_cleaning_and_imputation.ipynb`

**Chức năng:**
1. **Temporal Boundary Localization (TBL):** Xác định ranh giới thời gian hành động ký hiệu bằng góc khuỷu tay (MediaPipe Pose), loại bỏ tĩnh đầu/cuối.
2. **Spatial Crop & Resize:** Cắt vùng đầu-vai-eo và nén về kích thước chuẩn `224×224` pixel.
3. **Trích xuất Metadata JSON:** Quét các video đã tiền xử lý, tính toán các thông số thực tế (duration, num_frames, resolution `224x224`) và xuất ra file JSON mới.

**Các tham số quan trọng:**

| Tham số | Giá trị mặc định | Ý nghĩa |
|---------|-------------------|---------|
| `theta` | 160° | Ngưỡng góc khuỷu tay (< θ = active) |
| `t_min` | 0.67s | Thời lượng tối thiểu đoạn active |
| `max_gap` | 0.8s | Khoảng cách tối đa để gộp 2 đoạn |
| `padding` | 0.4s | Padding thêm trước/sau đoạn active |
| `target_size` | 224 | Kích thước đầu ra (pixel) |

**Đầu ra:**
| Thư mục / File | Mô tả |
|----------------|-------|
| `VSL_FULL_FRONT_CROPPED_TO224x224_V2/{gloss}/` | Video đã cắt và chuẩn hóa 224x224 |
| `preprocessed_vsl_metadata.json` | Metadata chi tiết của video đã tiền xử lý |

> **Lưu ý:** Phần xử lý video sử dụng đa nhân (ProcessPoolExecutor) để tối ưu hóa CPU. Với ~25,000 video trên CPU 8 nhân, thời gian chạy khoảng 4-8 giờ.

---

### Bước 3: Trích Xuất Keypoints 3D & Phân Tích EDA

**File:** `03_exploratory_data_analysis.ipynb`

**Chức năng:**
- Trích xuất **76 keypoints 3D** (34 body + 42 hands) bằng MediaPipe Holistic từ video đã sạch.
- Chuẩn hóa tọa độ theo bounding box cơ thể và bàn tay độc lập đưa về khoảng `[-0.5, 0.5]`.
- Tạo video animation trực quan hóa skeleton 3D đã chuẩn hóa.
- Phân tích thống kê EDA dataset (phân bố frames, glosses, ...).

**Cấu trúc Keypoints (76 điểm):**

| Nhóm | Số điểm | Chi tiết |
|------|---------|----------|
| **Body** | 34 | 33 MediaPipe Pose + 1 "neck" tổng hợp |
| **Left Hand** | 21 | Wrist + 5 ngón × 4 khớp |
| **Right Hand** | 21 | Wrist + 5 ngón × 4 khớp |

**Đầu ra:**
- `keypoints/{gloss}/{video_id}.npy` — Ma trận keypoints `[num_frames, 76, 3]`
- `test_vis/{video_id}_overlay.mp4` — Video overlay skeleton
- `test_vis/{video_id}_normalized_motion.mp4` — Video animation skeleton chuẩn hóa

---

## Sơ Đồ Pipeline Tổng Quan

```
                    ┌─────────────────────────┐
                    │   Dataset VSL-400 Gốc    │
                    │   (~25,000 video, HD)     │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │ 01_data_collection.ipynb│
                    │  Thu thập & tổ chức       │
                    │  • Gộp splits             │
                    │  • Phân loại theo gloss   │
                    │  • Chia train/test        │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │02_data_cleaning_and_    │
                    │imputation.ipynb         │
                    │  • TBL (góc khuỷu tay)    │
                    │  • Crop + Resize 224×224   │
                    │  • Trích xuất JSON sau    │
                    │    khi đã tiền xử lý      │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │03_exploratory_data_     │
                    │analysis.ipynb           │
                    │  • 76 keypoints 3D        │
                    │  • Chuẩn hóa tọa độ       │
                    │  • Trực quan hóa           │
                    │  • Phân tích thống kê      │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │  keypoints/*.npy          │
                    │  [frames, 76, 3]          │
                    │  → Sẵn sàng train model!  │
                    └─────────────────────────┘
```

---

## Thông Tin Kỹ Thuật

| Thuộc tính | Giá trị |
|------------|---------|
| Dataset | VSL-400 + VSL-UIT (472 glosses, ~26,673 videos) |
| Camera view | Front view (chỉ sử dụng góc chính diện) |
| Target resolution | 224×224 pixels |
| Keypoint model | MediaPipe Holistic (model_complexity=1) |
| Normalization | Body BBox 1.6× + Hand BBox independent |
| Train/Test split | 80/20 by signer ID (seed=42) |
| Output format | NumPy `.npy` (float32) |
| Parallel processing | ProcessPoolExecutor (all CPU cores) |

---

## Liên Hệ & Hỗ Trợ

Nếu gặp vấn đề khi chạy code, vui lòng kiểm tra:
1. Phiên bản Python >= 3.9
2. Các thư viện đã được cài đặt đầy đủ (`pip install -r requirements.txt`)
3. Đường dẫn dữ liệu đã được cấu hình đúng trong mỗi notebook
4. Đủ RAM (tối thiểu 8GB, khuyến nghị 16GB+)
