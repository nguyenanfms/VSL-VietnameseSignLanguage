# VSL-VietnameseSignLanguage: Pipeline Tien Xu Ly & Trich Xuat Dac Trung 3D Keypoints

He thong xu ly du lieu va trich xuat dac trung chuyen dong cho bai toan **Nhan Dien Ngon Ngu Ky Hieu Viet Nam (Vietnamese Sign Language - VSL)** dua tren dataset **VSL-400** va **VSL-UIT**.

Du an cung cap mot quy trinh khao sat toan dien tu du lieu video tho, qua dinh vi bien thoi gian (Temporal Boundary Localization - TBL), cat khung hinh (Spatial Crop & Resize 224x224), den trich xuat 76 toa do keypoints 3D chuan hoa su dung MediaPipe Holistic phuc vu huan luyen cac mo hinh Deep Learning (GCN, Transformer, LSTM).

---

## 1. Tinh Nang Noi Bat

- **Gop va to chuc du lieu**: Ho tro gop nhieu phan doan (splits) thanh mot kho luu tru tap trung va phan loai tu dong vao tung thu muc gloss (tu vung ky hieu).
- **Chia tap Train/Test theo Signer**: Chia tach du lieu theo dinh danh nguoi ky (Signer ID) de dam bao danh gia khach quan tren nhung nguoi ky chua tung xuat hien trong qua trinh huan luyen (Unseen-Signer Evaluation).
- **Temporal Boundary Localization (TBL)**: Loc bo cac khung hinh bat dong o dau va cuoi video dua tren thuat toan tinh goc khuyu tay (Elbow Angle) tu MediaPipe Pose (nguong goc < 160 do xac dinh trang thai ky hieu active).
- **Spatial Crop & Resize 224x224**: Tu dong xac dinh vung quan tam (Region of Interest - ROI) quanh phan dau, vai va eo dua tren khoang cach hai vai (shoulder width x 3.6), nén ve chuan 224x224 pixel.
- **Trich xuat 76 Keypoints 3D Chuan Hoa**: Su dung MediaPipe Holistic de trich xuat 34 diem co the (bao gom diem neck tong hop) va 42 diem ban tay (21 diem moi ban tay). Toa do duoc chuan hoa doc lap theo bounding box ve khoang [-0.5, 0.5].
- **Ho tro da hinh thuc su dung**: Cung cap day du ca 3 Jupyter Notebooks truc quan lan cac script CLI dong lenh ho tro da tien trinh (multiprocessing).

---

## 2. Cau Truc Thu Muc Du An

```
VSL-VietnameseSignLanguage/
│
├── .gitignore                      # Cau hinh loai tru du lieu video, npy, checkpoints
├── LICENSE                         # Giay phep ma nguon mo MIT
├── pyproject.toml                  # Cau hinh goi Python chuan
├── README.md                       # Tai lieu huong dan tong quan du an
├── requirements.txt                # Danh sach thu vien phu thuoc
│
├── notebooks/                      # Cac Jupyter Notebook tuong tac
│   ├── 01_data_collection.ipynb              # Thu thap, gop splits, phan loai gloss & chia signer
│   ├── 02_data_cleaning_and_imputation.ipynb # Tien xu ly TBL, Crop 224x224 & trich xuat JSON metadata
│   ├── 03_exploratory_data_analysis.ipynb    # Trich xuat 76 keypoints 3D & phan tich thong ke EDA
│   ├── requirements.txt                      # Dependencies danh rieng cho notebook
│   └── README.md                             # Huong dan su dung notebooks
│
├── src/                            # Ma nguon module hoa Python
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── merge_splits.py         # Module gop splits va hop nhat danh sach JSON
│   │   ├── categorize.py           # Module phan loai video theo gloss
│   │   └── split_signer.py         # Module chia train/test theo Signer ID
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── tbl.py                  # Thuat toan dinh vi bien thoi gian TBL
│   │   ├── cropper.py              # Thuat toan cat khong gian va resize ve 224x224
│   │   ├── metadata.py             # Trich xuat metadata sau tien xu ly
│   │   └── batch_processor.py      # Xu ly hang loat da nhan CPU (ProcessPoolExecutor)
│   └── features/
│       ├── __init__.py
│       ├── keypoints.py            # Trich xuat 76 diem landmarks bang MediaPipe Holistic
│       ├── normalizer.py           # Chuan hoa toa do co the (1.6x bbox) va ban tay
│       └── visualizer.py           # Tao video animation skeleton 3 panel
│
├── scripts/                        # Cac script CLI chay doc lap tu terminal
│   ├── 01_run_collection.py
│   ├── 02_run_preprocessing.py
│   └── 03_run_feature_extraction.py
│
├── data/                           # Thu muc chua du lieu (duoc bo qua trong git)
│   └── README.md                   # Huong dan to chuc du lieu
│
└── legacy/                         # Luu tru cac file code / notebook nguyen ban
    ├── extract_vsl_info.py
    ├── main_preprocessing.ipynb
    ├── merge_2_dataset.ipynb
    ├── merge_splits.py
    ├── split_front_view_by_signer.py
    ├── vsl400-keypoint-extract-new-1.ipynb
    └── worker_utils.py
```

---

## 3. So Do Quy Trinh Xu Ly (Pipeline Architecture)

```
                       [Video Goc VSL-400 / VSL-UIT]
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Giai doan 1: Thu Thap & To Chuc Du Lieu                 │
       │ - Gop cac split_1, split_2,...                          │
       │ - Phan loai video vao tung thu muc theo ten gloss       │
       │ - Chia train/test theo Signer ID (~80/20)               │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Giai doan 2: Tien Xu Ly Video (TBL & Spatial Crop)      │
       │ - Pass 1 (TBL): Danh gia goc khuyu tay < 160 do         │
       │ - Pass 2 (Crop): Cat vung dau-vai-eo (shoulder x 3.6)   │
       │ - Nén va xuat video chuan 224x224 px                    │
       │ - Trich xuat metadata JSON sau khi da tien xu ly        │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────┐
       │ Giai doan 3: Trich Xuat Dac Trung & Phan Tich EDA       │
       │ - MediaPipe Holistic: 34 Body + 42 Hand keypoints       │
       │ - Bounding Box Normalization ve khoang [-0.5, 0.5]      │
       │ - Xuat ma tran NumPy: [num_frames, 76, 3] (.npy)        │
       │ - Tao video animation truc quan hoa skeleton 3 panel    │
       │ - Thong ke phan bo frames, gloss va signers             │
       └────────────────────────────┬────────────────────────────┘
                                    │
                                    ▼
                      [File .npy San Sang Huan Luyen]
```

---

## 4. Yeu Cau He Thong & Cai Dat

### Yeu Cau He Thong

- **He dieu hanh**: Windows 10/11, Ubuntu 20.04+, macOS.
- **Python**: 3.9 tro len (khuyen nghi 3.10 hoac 3.11).
- **RAM**: Toi thieu 8 GB (khuyen nghi 16 GB+ khi xu ly da tien trinh).
- **CPU**: 4 nhan tro len (khuyen nghi 8+ nhan de toi uu thoi gian crop).

### Cac Buoc Cai Dat

1. Clone repository:
```bash
git clone https://github.com/nguyenanfms/VSL-VietnameseSignLanguage.git
cd VSL-VietnameseSignLanguage
```

2. Khoi tao moi truong ao (virtual environment):
```bash
# Tren Windows
python -m venv venv
venv\Scripts\activate

# Tren Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

3. Cai dat cac thu vien can thiet:
```bash
pip install -r requirements.txt
```

4. Cai dat package o che do development (tuy chon):
```bash
pip install -e .
```

---

## 5. Huong Dan Su Dung

### Cach 1: Su Dung Jupyter Notebooks

Khoi dong Jupyter Lab hoac Jupyter Notebook de thuc hien tung buoc truc quan:
```bash
jupyter lab notebooks/
```

- `01_data_collection.ipynb`: Thuc hien gop splits, phan chia gloss, chia train/test theo signer.
- `02_data_cleaning_and_imputation.ipynb`: Thuc thi TBL va spatial crop, sau do tu dong trich xuat metadata JSON thuc te cua tap video sach.
- `03_exploratory_data_analysis.ipynb`: Trich xuat 76 toa do 3D keypoints theo batch, tao video truc quan hoa 3 panel va phan tich thong ke dataset.

### Cach 2: Su Dung Cac Script CLI

Ban co the chay truc tiep pipeline tu dong lenh voi cac tham so tuy bien:

#### Buoc 1: Thu thap va to chuc du lieu
```bash
python scripts/01_run_collection.py --action all --splits-root data/raw_splits --merged-dir data/merged
```

#### Buoc 2: Tien xu ly video (TBL & Crop)
```bash
python scripts/02_run_preprocessing.py --input-dir data/categorized --output-dir data/preprocessed_224 --target-size 224
```

#### Buoc 3: Trich xuat keypoints 3D
```bash
# Chay batch so 1 tren tong so 4 batch
python scripts/03_run_feature_extraction.py --input-dir data/signer_splited/train --output-dir data/keypoints --batch-idx 1 --total-batches 4

# Tao video truc quan hoa skeleton mau tu file .npy da trich xuat
python scripts/03_run_feature_extraction.py --visualize-sample data/keypoints/Anh/sample.npy --vis-out data/sample_skeleton.mp4
```

---

## 6. Cau Truc Ma Tran Keypoints (76 Diem 3D)

Moi file `.npy` dai dien cho mot video duoc luu duoi dang mang NumPy 3 chieu voi shape: `[num_frames, 76, 3]`:

| Truc (Axis) | Kich thuoc | Y nghia |
|-------------|------------|---------|
| Axis 0 | `num_frames` | So luong khung hinh cua doan ky hieu sau TBL |
| Axis 1 | 76 | Danh muc 76 diem khac nhau tren co the va ban tay |
| Axis 2 | 3 | Toa do khong gian 3D (x, y, z) da duoc chuan hoa |

Phan bo 76 diem:
- **34 Body Landmarks**: 33 diem tu MediaPipe Pose + 1 diem `neck` tong hop (trung binh toa do hai vai).
- **21 Left Hand Landmarks**: Co tay va cac khop ngon tay trai.
- **21 Right Hand Landmarks**: Co tay va cac khop ngon tay phai.

---

## 7. Giay Phep (License)

Du an duoc phat hanh duoi giay phep [MIT License](LICENSE).
Tac gia: **Nguyen An** (nguyenanfms1401@gmail.com).
