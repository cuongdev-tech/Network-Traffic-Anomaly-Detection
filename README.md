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

Random Forest cũng dùng chung `data/processed/cicids_60k.csv.gz`. Chạy `python src/train_classifier.py` để train riêng Random Forest; model được lưu ở `models/cic/random_forest_attack.joblib`.

Model CIC được lưu trong `models/cic/`; model UNSW-NB15 được lưu riêng trong `models/unsw_nb15/`.

## Giai đoạn 4 — Random Forest và XGBoost (CIC-IDS2017 + UNSW-NB15)

`src/train_random_forest_xgboost.py` huấn luyện Random Forest (300 cây) và XGBoost trên từng bộ dữ liệu, chia train/test **80/20** (stratified, seed 42). Trọng số mỗi bộ được lưu riêng theo dataset:

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

## Mô hình dùng trong bản nộp

Mỗi bộ dữ liệu có năm baseline gồm Logistic Regression, Decision Tree, KNN, Random Forest và XGBoost. Mô hình học sâu chính là **TabNet cho CIC-IDS2017** và **TabR cho UNSW-NB15**.

| Dataset | Dòng dùng | Train / test | Đặc trưng gốc | Lớp |
|---|---:|---:|---:|---|
| CIC-IDS2017 | 511.807 | 409.445 / 102.362 | 77 | BENIGN, DDoS, PortScan |
| UNSW-NB15 | 184.673, sau khi giảm Normal còn 20.000 | 147.738 / 36.935 | 34 | Normal + 9 loại tấn công |

Bảng chính thức của từng bộ nằm ở `results/all_models_metrics.csv` và `results/unsw_nb15/all_models_metrics.csv`. Các kết quả TabNet và TabM trên UNSW được lưu trong các tệp riêng để đối chiếu. Các script baseline 60.000 dòng và UNSW nhị phân thuộc luồng thực nghiệm cũ. `src/train_full_unsw.py --all` tái lập thử nghiệm UNSW với TabNet. Dùng mục TabR bên dưới để tái lập bảng chính thức của UNSW.

Các phép đo thực hiện offline, dùng split 80/20 và có thể có dòng test trùng đặc trưng với train. Đánh giá TabR đầy đủ hiện có trên UNSW-NB15.

## Thử nghiệm TabM

TabM là mô hình học sâu cho dữ liệu bảng. Chạy từ thư mục gốc trên nhánh `main`:

```bash
python -m pip install -r requirements.txt
python -m src.train_tabm --dataset unsw
# Thử cùng giao thức trên CIC-IDS2017:
python -m src.train_tabm --dataset cic
```

Script dùng đúng cách chia 80/20 (seed 42) của bảng baseline hiện tại. Tiền xử lý, các ngưỡng embedding và hai cấu hình trọng số lớp được fit hoặc chọn bên trong tập train. Mặc định chọn checkpoint theo Accuracy trên validation. Dùng `--selection-metric f1_macro` nếu macro F1 là chỉ số chính đã thống nhất trước khi chạy. Tập test chỉ dùng để đánh giá cấu hình đã chọn. Lần thử này giữ split lịch sử nên train và test vẫn có các dòng trùng đặc trưng.

Kết quả gồm `tabm_metrics.csv`, `tabm_per_class.csv`, `tabm_meta.json` và `cm_tabm.png` trong thư mục kết quả của từng bộ dữ liệu. File `tabm_bundle.joblib` chứa cả model và bộ tiền xử lý. Các tệp này lưu kết quả thử nghiệm TabM.

Trên 36.935 mẫu test UNSW-NB15, TabM đạt Accuracy 79,3827% và macro F1 50,5684%. XGBoost đạt Accuracy cao nhất trong các baseline (79,8944%), còn Decision Tree đạt macro F1 cao nhất (57,5830%). Bảng thử nghiệm TabM được giữ để đối chiếu với TabNet và các baseline.

## TabR trên UNSW-NB15

```bash
python -m pip install -r requirements.txt
python -m src.train_tabr --pilot --max-epochs 2
python -m src.train_tabr --max-epochs 20
```

Pilot dùng 20.000 dòng huấn luyện và 3.000 dòng validation để đo thời gian chạy. Lượt đầy đủ dùng cùng split 80/20 và 10 nhãn với các baseline UNSW hiện có. Biến thể CPU sử dụng FAISS IVF-flat với 24 mẫu tham chiếu, làm mới danh sách láng giềng của train theo từng giai đoạn rồi giữ cố định để giảm thời gian chạy. Các truy vấn validation và test chỉ tìm trên phần train bên trong. Checkpoint được chọn theo Accuracy trên validation, rồi đánh giá trên tập test 36.935 dòng.

Kết quả lưu ở `results/unsw_nb15/tabr_metrics.csv`, `tabr_per_class.csv`, `tabr_meta.json` và `cm_tabr.png`. Checkpoint chứa trọng số và bộ tiền xử lý. Giao diện tự dựng chỉ mục tham chiếu khi chọn TabR, từ hai file Parquet trong `data/raw/unsw_nb15/` và đúng phần train bên trong đã dùng để đánh giá. Split lịch sử vẫn có các dòng trùng đặc trưng giữa train và test.

TabR đạt Accuracy **80,4142%** và macro F1 **61,1442%** trên toàn bộ test UNSW-NB15. Hai mốc baseline tương ứng là **79,8944%** của XGBoost và **57,5830%** của Decision Tree. Kiểm tra bổ sung giữ nguyên mô hình đã fit và chỉ tính trên 14.045 dòng test có bộ đặc trưng vắng mặt trong toàn bộ tập train. Trên phần này, Accuracy của TabR là **82,3994%** và của XGBoost là **81,1890%**.

## Giao diện desktop — PySide6

Thư mục `giao_dien/` cung cấp giao diện xem kết quả đã lưu, gồm bảng so sánh các mô hình, biểu đồ chỉ số, confusion matrix, chỉ số từng lớp và hình/bảng EDA cho CIC-IDS2017 và UNSW-NB15. Mục **Dự đoán dữ liệu mới** cho chọn CSV/CSV.GZ và model đã lưu, kiểm tra đặc trưng, dự đoán và xuất toàn bộ kết quả CSV. Demo với TabR chọn UNSW-NB15 và `giao_dien/mau_du_doan_unsw.csv`. Demo với TabNet chọn CIC-IDS2017 và `giao_dien/mau_du_doan_cic.csv`. Hai file mẫu dùng để kiểm tra luồng dự đoán và xuất CSV.

```bash
python -m pip install -r giao_dien/requirements.txt
python giao_dien/main.py
```

Kết quả chính trên giao diện và cột `prediction_class` trong CSV là **Lớp dự đoán**: UNSW có **10 lớp**, gồm **Bình thường (Normal)** và **Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms**. CIC có **Bình thường, DDoS, PortScan**. Cột `prediction` trong CSV giữ nhãn gốc model để đối chiếu; các cột cảnh báo 0/1 chỉ được giữ bổ sung trong CSV để tương thích, không thay thế kết quả phân loại 10 lớp.

Chọn dataset rồi nhấn **Làm mới** sau khi chạy lại các script. Cột train/test cho biết cách chia dữ liệu của từng nhóm mô hình; các mô hình dùng split khác nhau cần được đánh giá lại trên cùng tập test trước khi kết luận mô hình tốt hơn.

## Chạy lại

```bash
python -m pip install -r requirements.txt -r giao_dien/requirements.txt
python src/train_full_cic.py --all
python -m src.train_tabr --max-epochs 20
python giao_dien/main.py
```

## Ghi chú cho thành viên khác

Random Forest / XGBoost / TabNet dùng file `data/processed/cicids_60k.csv.gz` và cùng split 80/20 dưới đây. Các script cũ train_baselines.py vẫn dùng split 70/30; luồng train_full_* mới dùng 80/20 cho tất cả model:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("data/processed/cicids_60k.csv.gz")
X = data.drop(columns=["Label"]).select_dtypes(include=["number"])
y = LabelEncoder().fit_transform(data["Label"])
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
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
