# Giao diện desktop PBL ATTT

Từ thư mục gốc dự án:

```powershell
python -m pip install -r giao_dien/requirements.txt
python giao_dien/main.py
```

Hoặc chạy `python -m giao_dien.main`. Giao diện tự tìm thư mục `results` theo vị trí mã nguồn, không phụ thuộc thư mục đang mở terminal.

- **So sánh mô hình:** bảng Accuracy, Precision/Recall weighted, F1 weighted/macro và thời gian. Bộ lọc **Tất cả** hiện các mô hình có kết quả; **Baseline** hiện Logistic Regression, Decision Tree, KNN, Random Forest và XGBoost; **Học sâu** hiện TabNet trên CIC-IDS2017 và TabR trên UNSW-NB15. Bộ lọc áp dụng cho cả bảng và biểu đồ trên hai dataset.
- **Chi tiết mô hình:** confusion matrix, chỉ số từng lớp, thông tin lần train.
- **EDA dữ liệu:** chọn biểu đồ và bảng thống kê đã xuất cho CIC-IDS2017 hoặc UNSW-NB15.
- **Làm mới kết quả:** đọc lại file sau khi train/EDA xong. Nút mở hình hiển thị ảnh ở kích thước gốc bằng trình xem ảnh của máy.

Mục **Dự đoán dữ liệu mới**: chọn dataset bên trái, model đã lưu và CSV/CSV.GZ, rồi nhấn **Dự đoán**. Chương trình chạy nền, hiển thị 500 dòng đầu và cho xuất toàn bộ kết quả CSV. Cột đầu **Lớp dự đoán** (`prediction_class` trong CSV) hiển thị **Bình thường** hoặc tên loại tấn công cụ thể. UNSW phân loại **10 lớp**: **Bình thường (Normal), Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms**. CIC có **Bình thường (BENIGN), DDoS, PortScan**. Mỗi dòng trả về một lớp; phần tổng hợp đếm riêng từng lớp xuất hiện trong kết quả.

Nhãn được giải mã bằng thông tin lưu cùng model. CSV giữ thêm nhãn gốc ở cột `prediction`, xác suất, tên dataset/model và dữ liệu gốc. Các cột `prediction_code` (0/1), `prediction_status`, `attack_type` được giữ trong CSV để tương thích; giao diện ưu tiên kết quả phân loại cụ thể. Nhãn thật nếu có không dùng để dự đoán; xác suất không phải accuracy. Một file mẫu nhỏ có thể chỉ được dự đoán vào một số lớp, không nhất thiết đủ 10 lớp.

CSV phải có đủ đặc trưng, đúng đơn vị và cách tính như dữ liệu huấn luyện. Cột dư không đưa vào model. File thiếu cột, trùng tên cột, giá trị thiếu/không hợp lệ hoặc NaN/inf bị từ chối. CIC cần 77 đặc trưng số; UNSW cần 34 đặc trưng gồm proto/service/state. Chưa hỗ trợ nhập PCAP hoặc bắt gói realtime. Chỉ nạp model từ thư mục models của dự án, không nạp joblib/pickle do người dùng nhập.

Các trang báo cáo đọc CSV/JSON/PNG hiện có; giao diện không tự huấn luyện. Các model train_full_* mới dùng split 80/20; bảng hiển thị train/test và danh sách lớp từ metadata. File thiếu hoặc lỗi được hiển thị ngay tại phần liên quan.

Kiểm tra: `python -m unittest giao_dien.test_data giao_dien.test_gui`. Kiểm tra Qt chạy offscreen, không bật cửa sổ; nếu chưa cài thư viện giao diện thì phần kiểm tra Qt được bỏ qua.

Thiết kế: Framed sheet, seed 281715824; khung xanh xám nhạt, nền nội dung trắng, teal trên dữ liệu. Dùng Segoe UI trên Windows; nếu Qt không nhận font hệ thống, ứng dụng nạp DejaVu Sans có sẵn trong Matplotlib để hiển thị tiếng Việt. Không cần tải font. Các biểu đồ và con số lấy từ kết quả thật của dự án.

Nạp model mới: hỗ trợ `classes` và bundle cũ `labels`; áp dụng `scaler` hoặc `preprocessor` đã fit trước khi dự đoán. TabNet đọc `tabnet_attack.zip` trên CPU. TabR là lựa chọn mặc định trên UNSW-NB15. Để demo, chọn `mau_du_doan_unsw.csv`; chương trình nạp `tabr_checkpoint.joblib` và dựng chỉ mục từ đúng phần train trong hai file Parquet gốc ở `data/raw/unsw_nb15/`. Với UNSW ML cũ thiếu preprocessor, dùng `preprocessor_bundle.joblib` có cùng features/classes từ lần train chung; thiếu hoặc không khớp thì báo lỗi, không tự fit hoặc điền cột giả. Không chạy src/train_unsw_tabnet.py nhị phân cũ để ghi đè TabNet 10 lớp.
