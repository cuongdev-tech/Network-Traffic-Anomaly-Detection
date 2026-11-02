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

## Giai đoạn 3 — Baseline Logistic Regression / Decision Tree / KNN

`src/train_baselines.py` đọc `cicids_60k.csv.gz`, chia train/test 70/30 (stratified, seed 42), train 3 mô hình và chạy thêm 5-fold cross-validation.

| Mô hình | Accuracy | Precision | Recall | F1 weighted | F1 5-fold CV |
|---|---|---|---|---|---|
| Decision Tree | 0.9992 | 0.9992 | 0.9992 | 0.9992 | 0.9992 ± 0.0004 |
| KNN | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 0.9991 ± 0.0003 |
| Logistic Regression | 0.9966 | 0.9966 | 0.9966 | 0.9966 | 0.9961 ± 0.0007 |

Kết quả lưu ở `results/baselines_metrics.csv`, `baselines_cv.csv`, `baselines_per_class.csv`, `cm_*.png`, `baselines_compare.png`.

Random Forest cũng dùng chung `data/processed/cicids_60k.csv.gz` và split stratified 70/30 (seed 42). Chạy `python src/train_classifier.py`; model được lưu ở `models/cic/rf_attack.joblib`.

Model CIC được lưu trong `models/cic/`; model UNSW-NB15 được lưu riêng trong `models/unsw_nb15/`.

## Chạy lại

```bash
pip install -r requirements.txt
python src/prepare_cicids.py
python src/eda_cicids.py
python src/train_baselines.py --cv
python src/train_classifier.py
```

Notebook: `notebooks/04_baselines.ipynb`, `05_eda_cicids.ipynb`.

## Ghi chú cho thành viên khác

XGBoost / MLP có thể dùng file `data/processed/cicids_60k.csv.gz` và cùng split:

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

Ba baseline Logistic Regression, Decision Tree và KNN cũng có thể chạy trên [UNSW-NB15 của Kaggle](https://www.kaggle.com/datasets/dhoogla/unswnb15), với nhãn nhị phân `label` (BENIGN/ATTACK). Hai file Parquet `UNSW_NB15_training-set.parquet` và `UNSW_NB15_testing-set.parquet` được lưu trong Git tại `data/raw/unsw_nb15/`; sau khi clone/pull không cần tải lại. Giữ nguyên split train/test chính thức; mặc định lấy mẫu phân tầng tổng cộng 60.000 dòng để KNN có thời gian chạy hợp lý.

```bash
python src/train_unsw_baselines.py
# Dùng toàn bộ split chính thức thay vì lấy mẫu:
python src/train_unsw_baselines.py --sample 0
```

Lần chạy hiện tại dùng 42.000 dòng train và 18.000 dòng test, với 34 đặc trưng sau khi bỏ cột nhãn/metadata. Kết quả:

| Mô hình | Accuracy | F1 weighted | F1 macro |
|---|---:|---:|---:|
| Decision Tree | 0.8853 | 0.8848 | 0.8833 |
| KNN | 0.8323 | 0.8274 | 0.8234 |
| Logistic Regression | 0.8016 | 0.7993 | 0.7958 |

Kết quả và biểu đồ lưu ở `results/unsw_nb15/`; ba model lưu riêng ở `models/unsw_nb15/`, không ghi đè model CIC. File Parquet cần thư viện `pyarrow`, đã khai báo trong `requirements.txt`. Dataset Kaggle có giấy phép CC BY-NC-SA 4.0; cần ghi công Laurens D'Hooge và các tác giả gốc Nour Moustafa, Jill Slay, đồng thời tuân thủ điều khoản phi thương mại và chia sẻ tương tự. Tham khảo bài báo gốc “UNSW-NB15: a comprehensive data set for network intrusion detection systems” (Moustafa & Slay, 2015).

EDA riêng cho UNSW-NB15 dùng cả hai split chính thức và lưu bảng thống kê/biểu đồ tại `results/eda_unsw_nb15/`. Báo cáo EDA cũng kiểm tra feature trùng giữa hai split trong `split_feature_overlap.csv`; các bản ghi test trùng đặc trưng với train có thể làm kết quả đánh giá lạc quan hơn:

```bash
python src/eda_unsw_nb15.py
```
