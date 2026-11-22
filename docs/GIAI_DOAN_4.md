# Giai đoạn 4 theo đề tài 2.17

Chương trình giám sát chạy bằng Scapy. Chọn một giao diện lab để bắt gói trực tiếp hoặc nạp một PCAP đã lưu. Chương trình gom IPv4 TCP, UDP và ICMP hai chiều, chốt flow khi không có gói mới trong 5 giây hoặc khi đạt 60 giây. Có thể đổi hai thời hạn bằng tham số dòng lệnh. Log chỉ chứa metadata và kết quả phát hiện. [Tài liệu Scapy](https://scapy.readthedocs.io/en/stable/usage.html#asynchronous-sniffing)

Gaussian chấm điểm bất thường. Random Forest phân loại flow thành bình thường, Rbot hoặc NSIS.ay. Hai mô hình dùng cùng năm trường gồm thời lượng, số gói và ba cờ giao thức. Hệ thống ghi một cảnh báo flow khi Gaussian vượt ngưỡng hoặc Random Forest dự đoán một trong hai lớp botnet. Phân tích đồ thị dùng các cặp nguồn–đích từ tất cả flow đã chốt trong 900 giây gần nhất và chạy tối đa một lần mỗi 30 giây khi có dữ liệu mới. Cảnh báo đồ thị là ứng viên cấu trúc dựa trên kết nối định kỳ.

Các trường được chọn để có thể tạo từ gói tin và đối chiếu với NetFlow CTU-13. Quy tắc tách flow của Argus và bộ giám sát có thể khác nhau. Mô hình TabR của Giai đoạn 3 yêu cầu bộ đặc trưng khác và tiếp tục được dùng trong giao diện CSV.

## Chạy trên Kali Linux

Mở terminal tại thư mục gốc dự án. Lệnh bắt gói cần quyền đọc giao diện đã chọn. Các lệnh dưới đây dùng môi trường Python 3.11 trở lên. Nếu dùng bộ lọc BPF, hệ thống cần libpcap.

~~~bash
sudo apt install python3-venv libpcap-dev
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-monitor.txt
ip -br link
sudo .venv/bin/python -m src.stage4_monitor --interface lo --seconds 45 \
  --bpf 'ip and tcp port 8765' --output results/stage4/kali_loopback
~~~

Để tạo lưu lượng kiểm tra lành tính trên chính Kali, mở một terminal khác để chạy máy chủ HTTP chỉ lắng nghe trên loopback. Chạy máy chủ từ thư mục tạm rỗng hoặc truyền một thư mục rỗng để phục vụ.

~~~bash
sample_dir=$(mktemp -d)
python3 -m http.server 8765 --bind 127.0.0.1 --directory "$sample_dir"
~~~

Khi monitor báo đã bắt đầu lắng nghe, mở một terminal khác và gửi một yêu cầu HTTP tới máy chủ thử nghiệm.

~~~bash
curl http://127.0.0.1:8765/
~~~

Với giao diện mạng lab khác, thay **lo** bằng tên giao diện được cấp cho lab và chọn bộ lọc phù hợp. Kết thúc phiên, **events.jsonl** chứa thời gian UTC, đầu mút, lớp dự đoán, điểm Gaussian và thông tin đồ thị. **summary.json** chứa số gói, số flow, số cảnh báo, số gói rơi trong hàng đợi ứng dụng và độ trễ. Mỗi lần chạy chọn thư mục đầu ra mới để lưu lại log trước.

## Chạy với PCAP ngoại tuyến

~~~bash
python -m src.stage4_monitor --pcap data/raw/attack-injection-hao123.pcap --output results/stage4/demo_01
python -m src.stage4_monitor --pcap data/raw/attack-injection-id1cn.pcap --output results/stage4/demo_02
~~~

## Kết quả kiểm tra

| Nguồn | Số gói | Flow | Cảnh báo flow | Cảnh báo đồ thị | Độ trễ p95 sau gói cuối (ms) |
|---|---:|---:|---:|---:|---:|
| PCAP NETRESEC hao123 | 202 | 2 | 1 | 0 | — |
| PCAP NETRESEC id1cn | 155 | 9 | 7 | 0 | — |
| HTTP loopback Windows | 39 | 3 | 0 | 0 | 2171,57 |
| HTTP loopback Kali | 72 | 3 | 0 | 0 | 2082,05 |

Lần bắt gói trên Windows/Npcap dùng ba yêu cầu HTTP loopback và idle 2,00 giây. Cả ba flow lành tính đều có quyết định và không phát cảnh báo. Số gói rơi trong hàng đợi bằng 0. Lần chạy mẫu chỉ đo đường xử lý và độ trễ trong điều kiện loopback.

Trên Kali GNU/Linux Rolling trong VMware, bộ giám sát bắt trên lo với ba yêu cầu HTTP nội bộ và idle 2 giây. Chương trình nhận 72 gói, ghi 3 quyết định, 0 cảnh báo flow và rơi 0 gói trong hàng đợi. Log gốc, kết quả tính độ trễ và phiên bản hệ điều hành nằm ở **results/stage4/demo_live_kali**. Độ chính xác và độ trễ trên các tình huống tấn công trực tiếp chưa được đo.

Thời gian CPU trong tóm tắt đo mỗi lô chấm điểm flow. Với nguồn sống, độ trễ từ gói cuối tới quyết định gồm thời gian chờ chốt flow. Với PCAP, thời gian gốc chỉ dùng để sắp xếp gói và gom flow nên trường độ trễ live để trống. Các capture NETRESEC có ít flow và không được dùng để ước lượng độ chính xác.

## Replay NetFlow có nhãn trên Kali

Dùng các bản ghi TCP, UDP và ICMP có nhãn từ CTU-11 (Rbot) và CTU-12 (NSIS.ay). Trong mỗi fold, mô hình được huấn luyện trên các nhóm máy nguồn khác nhóm được chấm. Mỗi flow được chấm đúng một lần. Giữ thứ tự kết thúc gốc trong mỗi capture và fold. Chương trình đưa tối đa 256 flow đã hoàn tất vào bộ chấm điểm mỗi 100 ms, theo nhịp thử nghiệm cố định.

~~~bash
python -m src.stage4_flow_replay --output results/stage4/replay_lan_moi
~~~

| Kịch bản | Flow | TP | FP | FN | TN | Accuracy | Recall | FPR | p95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Rbot | 10.882 | 8.147 | 2.118 | 17 | 600 | 80,38% | 99,79% | 77,92% | 45,49 |
| NSIS.ay | 9.790 | 1.924 | 588 | 241 | 7.037 | 91,53% | 88,87% | 7,71% | 42,32 |

TP là số flow có nhãn botnet được cảnh báo, FP là số flow bình thường bị cảnh báo, FN là số flow botnet bị bỏ sót và TN là số flow bình thường được bỏ qua. p95 là thời gian từ lịch đưa lô vào đến khi có quyết định, gồm trễ lịch và thời gian chấm điểm. Flow trong cùng lô có chung thời điểm quyết định.

Thí nghiệm chạy trong tiến trình trên Kali, dùng NetFlow ghi sẵn. Số đo bắt đầu sau khi flow đã hoàn tất và chưa bao gồm thời gian bắt gói hay chờ chốt flow. Phần HTTP loopback ở trên ghi độ trễ bắt gói trực tiếp. Các kịch bản tấn công trực tiếp qua giao diện mạng vẫn chưa được thực hiện.

Dữ liệu và bằng chứng nằm ở **data/processed/stage4_replay_ctu13.csv.gz** và **results/stage4/demo_replay_kali**. **environment.json** lưu mã SHA-256 của dữ liệu đầu vào, phiên bản môi trường và phân nhóm máy nguồn. **decisions.csv.gz** chứa nhãn và cảnh báo theo từng flow. [Nguồn CTU-13](https://www.stratosphereips.org/datasets-ctu13/)

## Huấn luyện lại

~~~bash
python -m src.train_monitor
~~~

Dữ liệu **data/processed/stage4_ctu13.parquet** giữ thời lượng, số gói, giao thức và nhóm máy nguồn thay thế. Có thể dựng lại bằng **python -m src.train_monitor --source-dir <thu-muc-ctu13>** khi có hai tệp gốc cùng SHA-256 được ghi trong **data/processed/ctu13_stage3/source.json**. Nguồn CTU-11 và CTU-12 dùng nhãn flow thuộc Rbot hoặc NSIS.ay theo capture. Nhóm nguồn có mặt trong một fold duy nhất. Cấu hình triển khai được huấn luyện lại sau khi đánh giá ba fold theo máy nguồn. [Nguồn CTU-13](https://www.stratosphereips.org/datasets-ctu13/)

Đánh giá ngoại tuyến có 20.672 flow và 40 nhóm máy nguồn. Random Forest đạt Accuracy 85,94% và Macro F1 85,02%. Quy tắc cảnh báo kết hợp phát hiện 10.071/10.329 flow có nhãn botnet và báo nhầm 2.706/10.343 flow bình thường (FPR 26,16%). Đây là số đo trên NetFlow CTU-13, có thể khác trên dữ liệu do Scapy gom trực tiếp.
