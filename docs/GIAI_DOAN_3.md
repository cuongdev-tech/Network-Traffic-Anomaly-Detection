# Giai đoạn 3 của đề tài 2.17

## Cách chạy

Chạy từ thư mục gốc của dự án. Môi trường đã kiểm tra dùng Python 3.11 trên Windows.

~~~powershell
python -m pip install -r requirements.txt
python -m src.evaluate_stage3 --dataset cic
python -m src.evaluate_stage3 --dataset unsw
python -m src.evaluate_botnet_graph
python -m src.evaluate_ctu13_graph
~~~

Phân tích một CSV flow khác bằng bộ phát hiện đồ thị:

~~~powershell
python -m src.botnet_graph --input data/processed/flows.csv --output results/graph_demo
~~~

CSV nhận ba cột bắt buộc **src**, **dst**, **start**. Mỗi dòng là một lần bắt đầu kết nối và **start** tính theo giây. Nếu có nhiều bản ghi capture, cột **file** phân tách chúng. Kết quả **edges.csv** ghi số mốc bắt đầu khác nhau, trung bình và độ lệch chuẩn của khoảng cách giữa các mốc, hệ số biến thiên và cờ định kỳ. **components.csv** ghi dạng đồ thị, số nút/cạnh và thông tin nút trung tâm theo cửa sổ.

## Gaussian và phân loại tấn công

- Tập test của mỗi bộ có cùng chỉ mục với thực nghiệm TabR trước đó. Hai mô hình giám sát được nạp từ checkpoint hiện có và phải tái tạo Accuracy đa lớp đã lưu. Để so sánh phát hiện nhị phân, gộp các nhãn tấn công từ quyết định đa lớp thành **Attack**. Điểm xếp hạng ROC-AUC của bộ phân loại là 1 trừ xác suất của lớp bình thường.
- Gaussian độc lập theo đặc trưng số dùng phép biến đổi signed log1p, điền trung vị và chặn dưới phương sai ở 0,0001. Tham số được ước lượng chỉ từ dòng bình thường trong train bên trong. Ngưỡng lấy phân vị 99% từ tập validation bình thường, sau đó áp dụng bất biến cho test. Phân phối và ngưỡng chưa được tinh chỉnh bằng kết quả test.
- Accuracy và Macro F1 ở đây tính trên **hai nhãn**. Chỉ số báo động giả (FPR) là tỷ lệ dòng bình thường bị gán nhầm là tấn công. Bảng đa lớp đã chốt nằm ở hai tệp **all_models_metrics.csv** và giữ nguyên giao thức cũ.
- Các dòng trùng đặc trưng giữa tập train và test có thể ảnh hưởng cả ba phương pháp. Kết quả là thực nghiệm offline trên giao thức đã chốt, chưa chứng minh hiệu quả khi diễn biến lưu lượng thay đổi.

| Bộ dữ liệu | Mô hình | Accuracy | Macro F1 nhị phân | Recall tấn công | FPR | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| CIC-IDS2017 | Gaussian | 43,515% | 30,325% | 0,005% | 1,013% | 59,548% |
| CIC-IDS2017 | XGBoost | 99,996% | 99,996% | 99,995% | 0,002% | 99,998% |
| CIC-IDS2017 | TabR | 99,979% | 99,979% | 99,979% | 0,020% | 99,998% |
| UNSW-NB15 | Gaussian | 34,100% | 33,041% | 26,173% | 0,625% | 76,798% |
| UNSW-NB15 | XGBoost | 96,762% | 90,536% | 99,736% | 27,725% | 98,692% |
| UNSW-NB15 | TabR | 96,716% | 90,642% | 99,405% | 25,425% | 98,833% |

Với ngưỡng đã đặt, Gaussian bắt được rất ít dòng tấn công. Nhiều flow tấn công vẫn nằm trong vùng điểm phổ biến của lưu lượng bình thường, nhất là trên CIC. Hai mô hình có giám sát đạt F1 nhị phân cao hơn, nhưng báo động giả trên lớp Normal của UNSW còn đáng kể. Kết quả ROC-AUC và độ chính xác nhị phân không thay thế bảng phân loại đa lớp.

Khi nạp bộ tiền xử lý XGBoost UNSW, scikit-learn báo khác phiên bản tuần tự hóa. Chương trình kiểm tra Accuracy đa lớp sau khi nạp và đã tái tạo được kết quả lưu trước khi gộp nhãn nhị phân.

## Beaconing và đồ thị botnet

Mỗi cửa sổ 900 giây được tính riêng trong từng capture. Cạnh có hướng là cặp nguồn–đích đã quan sát. Cạnh được đánh dấu định kỳ nếu có ít nhất 5 mốc kết nối khác nhau và độ lệch chuẩn của các khoảng cách chia cho khoảng cách trung bình không quá 0,15. Với đồ thị vô hướng dựng từ các cạnh, ứng viên tập trung có tối thiểu 4 nút, một nút kề ít nhất 80% các nút còn lại và số cạnh không vượt quá 1,2 lần số cạnh của hình sao cùng kích thước. Ứng viên P2P có tối thiểu 4 nút, ít nhất 80% nút thuộc 2-core và bậc cao nhất không quá hai lần bậc trung vị. 2-core là phần đồ thị mà mỗi nút còn ít nhất hai hàng xóm trong chính phần đó.

| Phương pháp | Accuracy phát hiện | Precision | Recall | F1 | Báo giả | Bỏ sót | Đúng loại đồ thị |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chỉ dạng đồ thị | 70,00% | 62,50% | 100,00% | 76,92% | 3 | 0 | 70,00% |
| Chỉ tính định kỳ | 80,00% | 71,43% | 100,00% | 83,33% | 2 | 0 | — |
| Đồ thị từ các cặp định kỳ | 80,00% | 80,00% | 80,00% | 80,00% | 1 | 1 | 80,00% |

Bảng này tính trên **10 ca tổng hợp** gồm 2 dạng tập trung, 3 dạng P2P, 5 ca đối chứng. Ca dịch vụ hợp lệ định kỳ có đồ thị hình sao bị cảnh báo. Ca P2P thưa thiếu kết nối định kỳ trên một cạnh bị bỏ sót. Cả hai hiện diện trong bộ thử để minh họa giới hạn của tín hiệu chỉ dựa trên metadata. Dữ liệu tổng hợp chỉ là các dòng bảng có nhãn, không sinh lưu lượng mạng. Bộ thử nhỏ và do nhóm thiết kế, không ước lượng được độ chính xác thực tế.

Với **11 flow trích từ hai PCAP NETRESEC** của dự án, chương trình phân tích được 4 cạnh có hướng trong các cửa sổ, không thấy cạnh đạt ngưỡng định kỳ. Hai capture này không có nhãn botnet topology nên kết quả chỉ xác minh rằng bộ phân tích chạy được trên dữ liệu thu thập. [Phụ lục NETRESEC](APPENDIX_NETRESEC.md)

## Kiểm chứng trên CTU-13

Hai phiên NetFlow hai chiều được dùng nguyên số dòng: CTU-11 (Rbot, có lưu lượng IRC) và CTU-12 (NSIS.ay, P2P). Hai tệp gốc được lấy qua bản sao lưu công khai. Tệp **data/processed/ctu13_stage3/flows.parquet** giữ thời điểm tương đối, định danh máy thay thế và nhãn nguồn. Tệp **source.json** ghi nguồn, số dòng và SHA-256 để tái lập từ tệp gốc. [CTU-13 do Stratosphere Laboratory công bố](https://www.stratosphereips.org/datasets-ctu13/) · [Bài báo mô tả CTU-13](https://doi.org/10.1016/j.cose.2014.05.011)

~~~powershell
python -m src.evaluate_ctu13_graph
# Tái tạo Parquet nếu có hai tệp .binetflow gốc
python -m src.evaluate_ctu13_graph --source-dir <thu-muc-chua-hai-tep>
~~~

Mỗi mẫu đánh giá là một cặp nguồn–đích có hướng trong một cửa sổ 900 giây. Cặp có ít nhất một flow **Botnet** là dương tính. Cặp có **Normal** và không có **Botnet** là âm tính. Cặp chỉ có **Background** vẫn góp vào đồ thị nhưng được loại khỏi mẫu chấm. Nhãn chỉ dùng sau khi tạo đồ thị. Các ngưỡng nhận dạng hình dạng và định kỳ giữ như phép thử tổng hợp.

CTU-11 có 107.251 flow và 78 cạnh có nhãn, trong đó 16 cạnh dương tính. CTU-12 có 325.471 flow và 2.282 cạnh có nhãn, trong đó 1.725 cạnh dương tính. Số cạnh Background chưa chấm tương ứng là 47.607 và 124.521.

| Kịch bản | Phương pháp | Phát hiện đúng | Báo giả | Bỏ sót | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| CTU-11 | Chỉ dạng đồ thị | 2 | 1 | 14 | 66,67% | 12,50% | 21,05% |
| CTU-11 | Đồ thị từ các cặp định kỳ | 0 | 0 | 16 | 0,00% | 0,00% | 0,00% |
| CTU-11 | Chỉ tính định kỳ | 0 | 0 | 16 | 0,00% | 0,00% | 0,00% |
| CTU-12 | Chỉ dạng đồ thị | 0 | 0 | 1725 | 0,00% | 0,00% | 0,00% |
| CTU-12 | Đồ thị từ các cặp định kỳ | 0 | 0 | 1725 | 0,00% | 0,00% | 0,00% |
| CTU-12 | Chỉ tính định kỳ | 0 | 9 | 1725 | 0,00% | 0,00% | 0,00% |

Quy tắc chỉ xét hình dạng gắn đúng 2 cạnh Rbot vào ứng viên đồ thị tập trung. Ở CTU-12, các quy tắc chưa bắt được cạnh dương tính. Phép thử tổng hợp 8/10 vì vậy chỉ phản ánh các hình dạng đã thiết kế.

### Phân loại có giám sát bằng đặc trưng đồ thị

| Kịch bản | Phương pháp | Phát hiện đúng | Báo giả | Bỏ sót | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| CTU-11 | Logistic Regression | 10 | 22 | 6 | 31,25% | 62,50% | 41,67% |
| CTU-11 | Random Forest | 10 | 9 | 6 | 52,63% | 62,50% | 57,14% |
| CTU-12 | Logistic Regression | 1714 | 13 | 11 | 99,25% | 99,36% | 99,30% |
| CTU-12 | Random Forest | 1635 | 104 | 90 | 94,02% | 94,78% | 94,40% |

Random Forest có F1 tốt nhất trên 78 cạnh đã chấm của CTU-11 với 57,14%. Logistic Regression phát hiện 1.714/1.725 cạnh dương tính trên CTU-12, đạt Recall 99,36%. Số cạnh dương tính từng fold của CTU-12 lần lượt là 1.087, 629 và 9. Ở fold cuối, Logistic Regression bắt 5/9 cạnh. F1 tổng hợp chịu ảnh hưởng lớn từ các máy nguồn có nhiều cạnh.

Hai thuật toán bổ sung học từ 10 đặc trưng thời gian và đồ thị, gồm bậc vào/ra, hoạt động và tỷ lệ kết nối ở mỗi đầu mút, cặp đảo chiều và khoảng thời gian giữa lần kết nối. Chia ba fold theo máy nguồn trong từng capture. Mọi cạnh của một máy nguồn nằm cùng fold, và mã máy không được đưa vào đặc trưng. Đồ thị của từng cửa sổ được dựng từ toàn bộ metadata không nhãn, nên đây là phép kiểm tra trong cùng capture với ngữ cảnh đồ thị hiện có. Phép thử đo phát hiện cạnh botnet, còn nhãn gốc thiếu xác nhận hình dạng cho từng thành phần. Mô tả IRC/P2P là tham chiếu cấp kịch bản.

Kết quả lưu trong **results/stage3/ctu13/**. Bảng **edge_metrics.csv** ghi cả cảnh báo trên các cạnh Background chưa chấm. **graph_model_folds.csv** ghi số máy và cạnh ở từng fold.

## Vai trò và cách kết hợp

| Hướng phát hiện | Đơn vị phân tích | Thông tin trả về | Giới hạn chính |
|---|---|---|---|
| Gaussian | Flow | Điểm bất thường, cờ theo ngưỡng | Phụ thuộc phân phối của lưu lượng bình thường và giả định độc lập |
| XGBoost và TabR | Flow | Nhãn cụ thể đã học và xác suất | Phụ thuộc nhãn và độ đại diện của dữ liệu huấn luyện |
| Beaconing và đồ thị | Cặp host và thành phần liên thông trong cửa sổ | Dấu hiệu kết nối định kỳ, dạng tập trung/P2P | Dịch vụ hợp lệ cũng có thể định kỳ hoặc có cùng cấu trúc |

Khi có dữ liệu chứa cả đặc trưng flow, đầu mút kết nối và dấu thời gian, có thể dùng điểm Gaussian để ưu tiên kiểm tra, đưa flow qua bộ phân lớp, rồi tích lũy sự kiện gắn với host theo cửa sổ để xem đồ thị. Khi giám sát thật, các quyết định lọc sớm cần kiểm tra riêng vì chúng có thể làm mất kết nối cần cho đồ thị. Các bảng sử dụng đơn vị đánh giá khác nhau nên không dùng Accuracy của đồ thị để xếp hạng trực tiếp với Gaussian hoặc TabR. Bắt gói trên Kali và đánh giá độ trễ live thuộc Giai đoạn 4.

Tệp kết quả trong **results/stage3** gồm bảng phát hiện nhị phân, metadata Gaussian, tỷ lệ cảnh báo theo nhãn, kết quả từng ca đồ thị và hình minh họa.
