# Phụ lục — Pipeline pcap trên mẫu NETRESEC

Phần này giữ lại để minh chứng kỹ năng đọc pcap và trích đặc trưng, **không dùng để train trong báo cáo chính** vì chỉ có 11 flow.

## Dữ liệu

- `data/raw/attack-injection-hao123.pcap` — 176 KB, 202 gói
- `data/raw/attack-injection-id1cn.pcap` — 74 KB, 155 gói
- Nguồn: <https://www.netresec.com/?page=pcapfiles>

## Kết quả

- `data/processed/flows.csv` — 11 flow, 24 cột
- `data/processed/hosts.csv` — 2 dòng host/60 giây
- `data/processed/labeled_flows.csv` — thêm nhãn tay 6 suspicious / 5 normal

## Code

| File | Việc |
|---|---|
| `src/explore_pcap.py` | Đọc pcap bằng Scapy, in thông tin từng gói |
| `src/pcap_to_flows.py` | Gom 5-tuple thành flow, trích feature flow/host |
| `src/label_flows.py` | Gán nhãn rule-based cho 11 flow |
| `src/train_classifier.py` | Random Forest hiện train trên CIC-IDS2017; không còn train trên 11 flow |

Notebook: `notebooks/01_explore_pcap.ipynb`, `02_flow_features.ipynb`, `03_eda_classification.ipynb`.

## Vì sao không dùng để train

357 gói tin gom lại chỉ còn 11 flow. Tập này quá nhỏ, train 7 / test 4 nên kết quả 100% chỉ là overfitting, không có ý nghĩa thống kê. Vì vậy toàn bộ EDA và mô hình trong báo cáo chính chuyển sang **CIC-IDS2017** (511.807 flow). Kết quả Random Forest cũ được lưu trong notebook chỉ mang tính lịch sử; script hiện tại train trên bộ CIC đã xử lý.
