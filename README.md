# Ứng dụng học máy phát hiện tấn công mạng từ dữ liệu thu thập trên mạng

Mình làm một pipeline phát hiện tấn công mạng đơn giản: đọc file pcap thô bằng Scapy, gom thành flow, trích đặc trưng rồi train thử mô hình Random Forest. Dữ liệu mẫu lấy từ kho NETRESEC.

## Đã làm tới đâu

**Giai đoạn 1 — Biến pcap thành bảng dữ liệu**

- Tải 2 file mẫu từ NETRESEC về `data/raw/`: `hao123` (202 gói) và `id1cn` (155 gói).
- Đọc từng gói tin bằng Scapy: IP nguồn/đích, port, giao thức, cờ TCP, size, timestamp.
- Gom các gói cùng 5-tuple thành 1 flow, rồi gom tiếp theo IP nguồn trong cửa sổ 60 giây cho góc nhìn host.
- Trích đặc trưng flow (`duration`, `n_pkts`, `bytes`, `rst`, `syn_no_ack`...) và host (`conn_per_sec`, `rst_ratio`, `port_entropy`...).
- Kết quả lưu ở `data/processed/flows.csv` (11 flows) và `hosts.csv`.

**Giai đoạn 2 — Gán nhãn và train thử**

- File pcap chưa có nhãn nên mình gán tay bằng rule đơn giản: flow nào ngắn mà dính nhiều cờ RST thì đánh dấu suspicious (1), còn lại là normal (0). Được 6 suspicious / 5 normal.
- Vẽ EDA trong notebook 03: nhóm suspicious thường ngắn dưới 6 giây, cờ RST cao hẳn, còn nhóm normal thì flow dài và ổn định hơn.
- Train thử Random Forest (100 cây) trên tập nhỏ này, test 4 mẫu đạt 1.0. Nói trước là tập có 11 mẫu nên con số này chỉ để kiểm tra pipeline chạy thông suốt, chưa có ý nghĩa thống kê.

## Chạy lại thế nào

```bash
pip install scapy pandas scikit-learn matplotlib seaborn dpkt jupyter
python src/explore_pcap.py
python src/pcap_to_flows.py
python src/label_flows.py
python src/train_classifier.py
```

Notebook để xem từng bước: `notebooks/01_explore_pcap.ipynb`, `02_flow_features.ipynb`, `03_eda_classification.ipynb`.

## Hướng tiếp theo

Lấy thêm bộ dữ liệu chuẩn có nhãn sẵn như CIC-IDS2017 hoặc CTU-13 để train cho ra hồn, rồi làm thêm phát hiện beaconing/botnet và bản chạy realtime trên Kali.
