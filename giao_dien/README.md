# Giao diện desktop PBL ATTT

Từ thư mục gốc dự án:

```powershell
python -m pip install -r giao_dien/requirements.txt
python giao_dien/main.py
```

Hoặc chạy `python -m giao_dien.main`. Giao diện tự tìm thư mục `results` theo vị trí mã nguồn, không phụ thuộc thư mục đang mở terminal.

- **So sánh mô hình:** bảng Accuracy, Precision/Recall weighted, F1 weighted/macro và thời gian; biểu đồ đổi chỉ số, lọc nhóm mô hình.
- **Chi tiết mô hình:** confusion matrix, chỉ số từng lớp, thông tin lần train.
- **EDA dữ liệu:** chọn biểu đồ và bảng thống kê đã xuất cho CIC-IDS2017 hoặc UNSW-NB15.
- **Làm mới kết quả:** đọc lại file sau khi train/EDA xong. Nút mở hình hiển thị ảnh ở kích thước gốc bằng trình xem ảnh của máy.

Giao diện chỉ đọc CSV/JSON/PNG hiện có, không tải model joblib hoặc tự chạy huấn luyện. Baseline và RF/XGBoost/MLP đang dùng tập test khác nhau; cột Train/test và thông báo trên màn hình nêu rõ giới hạn so sánh. File thiếu hoặc lỗi được hiển thị ngay tại phần liên quan.

Kiểm tra: `python -m unittest giao_dien.test_data giao_dien.test_gui`. Kiểm tra Qt chạy offscreen, không bật cửa sổ; nếu chưa cài thư viện giao diện thì phần kiểm tra Qt được bỏ qua.

Thiết kế: Framed sheet, seed 281715824; khung xanh xám nhạt, nền nội dung trắng, teal trên dữ liệu. Dùng Segoe UI trên Windows; nếu Qt không nhận font hệ thống, ứng dụng nạp DejaVu Sans có sẵn trong Matplotlib để hiển thị tiếng Việt. Không cần tải font. Các biểu đồ và con số lấy từ kết quả thật của dự án.
