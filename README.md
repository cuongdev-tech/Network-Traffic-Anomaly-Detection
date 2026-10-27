# Ứng dụng học máy phát hiện tấn công mạng từ dữ liệu thu thập trên mạng

Mình làm một pipeline phát hiện tấn công mạng đơn giản: đọc file pcap thô bằng Scapy, gom thành flow, trích đặc trưng rồi train thử mô hình Random Forest. Dữ liệu mẫu lấy từ kho NETRESEC.

## Giai đoạn 1 — Biến pcap thành bảng dữ liệu

- Tải 2 file mẫu từ NETRESEC về `data/raw/`: `hao123` (202 gói) và `id1cn` (155 gói).
- Đọc từng gói tin bằng Scapy: IP nguồn/đích, port, giao thức, cờ TCP, size, timestamp.
- Gom các gói cùng 5-tuple thành 1 flow, rồi gom tiếp theo IP nguồn trong cửa sổ 60 giây cho góc nhìn host.
- Trích đặc trưng flow (`duration`, `n_pkts`, `bytes`, `rst`, `syn_no_ack`...) và host (`conn_per_sec`, `rst_ratio`, `port_entropy`...).
- Kết quả lưu ở `data/processed/flows.csv` (11 flows) và `hosts.csv`.

## Giai đoạn 2 — Gán nhãn và train thử

- File pcap chưa có nhãn nên mình gán tay bằng rule đơn giản: flow nào ngắn mà dính nhiều cờ RST thì đánh dấu suspicious (1), còn lại là normal (0). Được 6 suspicious / 5 normal.
- Vẽ EDA trong notebook 03: nhóm suspicious thường ngắn dưới 6 giây, cờ RST cao hẳn, còn nhóm normal thì flow dài và ổn định hơn.
- Train thử Random Forest (100 cây) trên tập nhỏ này, test 4 mẫu đạt 1.0. Nói trước là tập có 11 mẫu nên con số này chỉ để kiểm tra pipeline chạy thông suốt, chưa có ý nghĩa thống kê.

## Giai đoạn 3 — Baseline trên bộ dữ liệu thứ hai (CIC-IDS2017)

- Tải 2 file Friday afternoon của CIC-IDS2017 (`DDoS`, `PortScan`) từ Hugging Face, gộp được 570k flow, 3 lớp `BENIGN / DDoS / PortScan`.
- Lấy mẫu stratified 60k dòng, 77 đặc trưng số, chia train/test 70/30.
- Train 3 mô hình baseline: Logistic Regression, Decision Tree, KNN.

| Mô hình | Accuracy | Precision | Recall | F1 (weighted) | F1 5-fold CV |
|---|---|---|---|---|---|
| Decision Tree | 0.9992 | 0.9992 | 0.9992 | 0.9992 | 0.9992 ± 0.0004 |
| KNN | 0.9990 | 0.9990 | 0.9990 | 0.9990 | 0.9991 ± 0.0003 |
| Logistic Regression | 0.9966 | 0.9966 | 0.9966 | 0.9966 | 0.9961 ± 0.0007 |

Kết quả lưu ở `results/baselines_metrics.csv`, `results/baselines_cv.csv`, `results/baselines_per_class.csv` và `results/cm_*.png`.

EDA trên CIC-IDS2017 (`src/eda_cicids.py`): PortScan có duration trung vị 47 micro-giây và quét 1,000 port; DDoS flow kéo dài ~1.9 giây; BENIGN đa dạng port nhất. Hình lưu ở `results/eda_cicids/`.

### Tải dữ liệu CIC-IDS2017

```bash
# Tải 2 file vào data/raw/cicids2017/
curl -L -o data/raw/cicids2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv \
  https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv
curl -L -o data/raw/cicids2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv \
  https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
```

## Chạy lại thế nào

```bash
pip install -r requirements.txt
python src/explore_pcap.py
python src/pcap_to_flows.py
python src/label_flows.py
python src/train_classifier.py
python src/train_baselines.py --sample 60000
python src/eda_cicids.py
```

Notebook để xem từng bước: `notebooks/01_explore_pcap.ipynb`, `02_flow_features.ipynb`, `03_eda_classification.ipynb`, `04_baselines.ipynb`, `05_eda_cicids.ipynb`.
