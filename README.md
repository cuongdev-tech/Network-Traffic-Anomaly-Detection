# Ứng dụng học máy phát hiện tấn công mạng từ dữ liệu thu thập trên mạng

Pipeline phát hiện tấn công mạng: chuẩn bị dữ liệu flow, phân tích EDA, train các mô hình baseline và so sánh chỉ số. Bộ dữ liệu chính là **CIC-IDS2017**.

> Trước đó nhóm thử trên 2 file pcap NETRESEC nhưng chỉ ra 11 flow, quá ít để train. Phần đó được giữ làm phụ lục kỹ thuật: [docs/APPENDIX_NETRESEC.md](docs/APPENDIX_NETRESEC.md).

## Dữ liệu chính — CIC-IDS2017

Gộp 2 file Friday afternoon (DDoS + PortScan), sau khi bỏ NaN/inf còn **511.807 flow**, 3 lớp:

| Lớp | Số flow |
|---|---|
| BENIGN | 224.978 |
| PortScan | 158.804 |
| DDoS | 128.025 |

Hai CSV Friday afternoon (~147 MB tổng cộng) được lưu trong Git tại `data/raw/cicids2017/`; sau khi clone/pull không cần tải lại. Nguồn dữ liệu: [CIC-IDS2017 trên Hugging Face](https://huggingface.co/datasets/c01dsnap/CIC-IDS2017), do Canadian Institute for Cybersecurity phát hành. Khi sử dụng, vui lòng trích dẫn bài báo của Sharafaldin, Lashkari và Ghorbani (ICISSP 2018), “Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization”.

## Giai đoạn 1 — Chuẩn bị dữ liệu

- `src/prepare_cicids.py`: đọc 2 CSV, chuẩn hóa tên cột, bỏ cột định danh, thay inf/NaN, lấy mẫu stratified 60.000 dòng.
- Kết quả: `data/processed/cicids_60k.csv.gz` (~4,5 MB), 77 đặc trưng số.
- File này dùng chung cho EDA, train và cho các thành viên khác, tránh mỗi người xử lý một kiểu.

## Giai đoạn 2 — Phân tích EDA

`src/eda_cicids.py` xuất 5 hình vào `results/eda_cicids/`:

- Phân bố lớp, duration theo lớp, bytes/s theo lớp.
- Top 10 port đích của từng lớp, số cờ TCP trung bình.

Nhận xét chính:

- **PortScan**: duration trung vị 47 micro-giây, gần như 1 gói đi + 1 gói về, quét 1.000 port.
- **DDoS**: flow kéo dài (~1,9 giây), bytes/s thấp, chỉ nhắm 4 port.
- **BENIGN**: đa dạng port nhất (30.794 port).

## Giai đoạn 3 — 5 mô hình baseline: LR / Decision Tree / KNN / Random Forest / XGBoost

`src/train_baselines.py` đọc `cicids_60k.csv.gz`, chia train/test 70/30 (stratified, seed 42), train 5 mô hình học máy cổ điển và chạy thêm 5-fold cross-validation.

| Mô hình | Accuracy | Precision | Recall | F1 weighted | F1 5-fold CV |
|---|---|---|---|---|---|
| XGBoost | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.9998 ± 0.0001 |
| Random Forest | 0.9995 | 0.9995 | 0.9995 | 0.9995 | 0.9996 ± 0.0001 |
| Decision Tree | 0.9992 | 0.9992 | 0.9992 | 0.9992 | 0.9992 ± 0.0004 |
| KNN | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 0.9991 ± 0.0003 |
| Logistic Regression | 0.9966 | 0.9966 | 0.9966 | 0.9966 | 0.9961 ± 0.0007 |

Kết quả lưu ở `results/baselines_metrics.csv` (có cột `group = baseline`), `baselines_cv.csv`, `baselines_per_class.csv`, `cm_baseline_*.png`, `baselines_compare.png`.

Model CIC được lưu trong `models/cic/`; model UNSW-NB15 được lưu riêng trong `models/unsw_nb15/`.

## Giai đoạn 4 — Train riêng Random Forest / XGBoost (split 80/20)

Random Forest và XGBoost đã được gộp vào Giai đoạn 3 (cùng split 70/30 với các baseline còn lại). Script `src/train_random_forest_xgboost.py` giữ lại để train model riêng cho giao diện, chia train/test **80/20** (stratified, seed 42) nên số liệu dưới đây **không dùng để so sánh với baseline**:

- CIC-IDS2017: 60.000 flow, 77 đặc trưng, 3 lớp → `models/cic/random_forest_attack.joblib`, `models/cic/xgboost_attack.joblib`.
- UNSW-NB15: 60.000 dòng (mặc định), 34 đặc trưng, BENIGN/ATTACK → `models/unsw_nb15/random_forest_attack.joblib`, `models/unsw_nb15/xgboost_attack.joblib`.

```bash
python src/train_random_forest_xgboost.py --dataset cic
python src/train_random_forest_xgboost.py --dataset unsw
python src/train_random_forest_xgboost.py
# Dùng toàn bộ UNSW-NB15 thay vì lấy mẫu 60k:
python src/train_random_forest_xgboost.py --dataset unsw --sample 0
```

| Bộ dữ liệu | Mô hình | Accuracy | F1 weighted | F1 macro |
|---|---|---:|---:|---:|
| CIC-IDS2017 | Random Forest | 0.9995 | 0.9995 | 0.9995 |
| CIC-IDS2017 | XGBoost | 0.9998 | 0.9998 | 0.9998 |
| UNSW-NB15 | Random Forest | 0.9273 | 0.9279 | 0.9225 |
| UNSW-NB15 | XGBoost | 0.9299 | 0.9299 | 0.9241 |

Kết quả lưu ở `results/random_forest_xgboost_metrics.csv`, `results/random_forest_xgboost_per_class.csv`, `results/random_forest_xgboost_compare.png`, `cm_random_forest.png`, `cm_xgboost.png`; nhánh UNSW tương tự trong `results/unsw_nb15/`.

## Giai đoạn 5 — Mạng nơron MLP (CIC-IDS2017 + UNSW-NB15)

`src/train_mlp.py` huấn luyện MLP với 2 lớp ẩn `(128, 64)`, ReLU, Adam, early stopping, cùng split **80/20** (stratified, seed 42) như Random Forest và XGBoost. Đặc trưng số dùng `QuantileTransformer(output_distribution="normal")`; UNSW-NB15 giữ imputer và one-hot cho biến phân loại. Learning rate là `0.0003` cho CIC và `0.001` cho UNSW, chờ 30 epoch không cải thiện trước khi dừng, tối đa 400 epoch.

Cấu hình được chọn trên validation nằm trong tập train, gom các dòng trùng toàn bộ đặc trưng vào cùng nhóm để không xuất hiện ở cả hai phía của validation. Các thử nghiệm được lưu trong `results/mlp_validation_candidates.csv`; tập test 12.000 dòng của mỗi dataset được giữ nguyên để đối chiếu trước/sau trong `results/mlp_improvement.csv`.

- Model: `models/cic/mlp_attack.joblib`, `models/unsw_nb15/mlp_attack.joblib`.
- Kết quả: `results/mlp_metrics.csv`, `results/mlp_per_class.csv`, `results/mlp_meta.json`, `results/cm_mlp.png`; nhánh UNSW trong `results/unsw_nb15/`.
- So sánh 3 mô hình: `results/all_models_metrics.csv`, `results/all_models_compare.png` (và nhánh UNSW).

```bash
python src/train_mlp.py --dataset cic
python src/train_mlp.py --dataset unsw
python src/train_mlp.py
```

| Bộ dữ liệu | Mô hình | Accuracy | F1 weighted | F1 macro |
|---|---|---:|---:|---:|
| CIC-IDS2017 | MLP | 0.9996 | 0.9996 | 0.9996 |
| UNSW-NB15 | MLP | 0.9192 | 0.9194 | 0.9129 |

Accuracy trước/sau lần tối ưu này: CIC **99,9417% → 99,9583%**, UNSW **91,4167% → 91,9167%**. Đây là kết quả trên split hiện tại; test vẫn có các dòng trùng đặc trưng với train và nhánh UNSW vẫn gộp hai file rồi chia lại. Vì vậy chưa dùng các điểm này để khẳng định hiệu quả trên lưu lượng mạng mới hoặc so sánh trực tiếp với baseline dùng split chính thức.

## Giao diện desktop — PySide6

Thư mục `giao_dien/` cung cấp giao diện xem kết quả đã lưu, gồm bảng so sánh sáu mô hình, biểu đồ chỉ số, confusion matrix, chỉ số từng lớp và hình/bảng EDA cho CIC-IDS2017 và UNSW-NB15. Giao diện đọc trực tiếp các file trong `results/`, không cần nạp model hoặc chạy huấn luyện.

```bash
python -m pip install -r giao_dien/requirements.txt
python giao_dien/main.py
```

Chọn dataset rồi nhấn **Làm mới** sau khi chạy lại các script. Cột train/test cho biết cách chia dữ liệu của từng nhóm mô hình; các mô hình dùng split khác nhau cần được đánh giá lại trên cùng tập test trước khi kết luận mô hình tốt hơn.

## Chạy lại

```bash
pip install -r requirements.txt
python src/prepare_cicids.py
python src/eda_cicids.py
python src/train_baselines.py --cv
python src/train_unsw_baselines.py
python src/compare_baselines.py
python src/train_classifier.py
python src/train_random_forest_xgboost.py
python src/train_mlp.py
```

Notebook: `notebooks/04_baselines.ipynb`, `05_eda_cicids.ipynb`.

## Ghi chú cho thành viên khác

MLP dùng file `data/processed/cicids_60k.csv.gz` với split **80/20**, trong khi 5 baseline ở Giai đoạn 3 dùng split 70/30. Muốn so sánh công bằng thì MLP phải chạy lại với `TEST_SIZE = 0.3` và cùng tập test:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("data/processed/cicids_60k.csv.gz")
X = data.drop(columns=["Label"]).select_dtypes(include=["number"])
y = LabelEncoder().fit_transform(data["Label"])
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
```

## Baseline bổ sung — UNSW-NB15

Năm baseline Logistic Regression, Decision Tree, KNN, Random Forest và XGBoost cũng chạy trên [UNSW-NB15 của Kaggle](https://www.kaggle.com/datasets/dhoogla/unswnb15), với nhãn nhị phân `label` (BENIGN/ATTACK). Hai file Parquet `UNSW_NB15_training-set.parquet` và `UNSW_NB15_testing-set.parquet` được lưu trong Git tại `data/raw/unsw_nb15/`; sau khi clone/pull không cần tải lại. Giữ nguyên split train/test chính thức; mặc định lấy mẫu phân tầng tổng cộng 60.000 dòng để KNN có thời gian chạy hợp lý.

```bash
python src/train_unsw_baselines.py
# Dùng toàn bộ split chính thức thay vì lấy mẫu:
python src/train_unsw_baselines.py --sample 0
```

Lần chạy hiện tại dùng 42.000 dòng train và 18.000 dòng test, với 34 đặc trưng sau khi bỏ cột nhãn/metadata. Kết quả:

| Mô hình | Accuracy | F1 weighted | F1 macro |
|---|---:|---:|---:|
| Decision Tree | 0.8853 | 0.8848 | 0.8833 |
| XGBoost | 0.8680 | 0.8651 | 0.8623 |
| Random Forest | 0.8649 | 0.8618 | 0.8588 |
| KNN | 0.8323 | 0.8274 | 0.8234 |
| Logistic Regression | 0.8016 | 0.7993 | 0.7958 |

Kết quả và biểu đồ lưu ở `results/unsw_nb15/`; model lưu riêng ở `models/unsw_nb15/`, không ghi đè model CIC. File Parquet cần thư viện `pyarrow`, đã khai báo trong `requirements.txt`. Dataset Kaggle có giấy phép CC BY-NC-SA 4.0; cần ghi công Laurens D'Hooge và các tác giả gốc Nour Moustafa, Jill Slay, đồng thời tuân thủ điều khoản phi thương mại và chia sẻ tương tự. Tham khảo bài báo gốc “UNSW-NB15: a comprehensive data set for network intrusion detection systems” (Moustafa & Slay, 2015).

EDA riêng cho UNSW-NB15 dùng cả hai split chính thức và lưu bảng thống kê/biểu đồ tại `results/eda_unsw_nb15/`. Báo cáo EDA cũng kiểm tra feature trùng giữa hai split trong `split_feature_overlap.csv`; các bản ghi test trùng đặc trưng với train có thể làm kết quả đánh giá lạc quan hơn:

```bash
python src/eda_unsw_nb15.py
```
