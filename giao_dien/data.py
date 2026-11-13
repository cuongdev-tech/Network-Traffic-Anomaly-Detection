"""Read the project's result files without loading models or raw datasets."""
import json
from pathlib import Path

import pandas as pd

DATASETS = {"cic": "CIC-IDS2017", "unsw": "UNSW-NB15"}
METRICS = {
    "accuracy": "Accuracy",
    "precision_weighted": "Precision weighted",
    "recall_weighted": "Recall weighted",
    "f1_weighted": "F1 weighted",
    "f1_macro": "F1 macro",
}
MODEL_NAMES = {
    "LogisticRegression": "Logistic Regression", "DecisionTree": "Decision Tree",
    "KNN": "KNN", "Random Forest": "Random Forest", "XGBoost": "XGBoost", "MLP": "MLP",
}
SOURCES = (
    ("baselines", "Baseline", {"LogisticRegression", "DecisionTree", "KNN"}),
    ("random_forest_xgboost", "RF / XGBoost", {"Random Forest", "XGBoost"}),
    ("mlp", "MLP", {"MLP"}),
)
EDA_CATEGORICAL_FEATURES = {"proto": "Giao thức", "service": "Dịch vụ", "state": "Trạng thái"}
EDA_METRIC_NAMES = {
    "distinct_feature_signatures_in_both_splits": "Số đặc trưng xuất hiện ở cả hai tập",
    "train_rows_with_feature_signature_in_test": "Dòng train trùng đặc trưng với test",
    "test_rows_with_feature_signature_in_train": "Dòng test trùng đặc trưng với train",
    "train_duplicate_feature_rows": "Dòng train trùng lặp đặc trưng",
    "test_duplicate_feature_rows": "Dòng test trùng lặp đặc trưng",
}
EDA_LABEL_VALUES = {0: "0 · Bình thường", 1: "1 · Tấn công"}
EDA_SPLIT_VALUES = {"train": "train · Huấn luyện", "test": "test · Kiểm thử"}

COLUMN_TRANSLATIONS = {
    # ── CIC-IDS2017: bảng summary ──
    "Label": "Nhãn",
    "n_flows": "Số luồng",
    "duration_mean_us": "Thời lượng TB (µs)",
    "duration_median_us": "Thời lượng TV (µs)",
    "bytes_per_s_median": "Tốc độ TV (B/s)",
    "fwd_pkts_mean": "Gói đi TB",
    "bwd_pkts_mean": "Gói về TB",
    "syn_mean": "Cờ SYN TB",
    "rst_mean": "Cờ RST TB",
    "n_dst_ports": "Số cổng đích",

    # ── UNSW-NB15: numeric_summary_by_label ──
    "label": "Nhãn",
    # dur
    "dur_mean": "Thời lượng TB (s)",
    "dur_median": "Thời lượng TV (s)",
    "dur_std": "Thời lượng ĐLC",
    # spkts
    "spkts_mean": "Gói nguồn→đích TB",
    "spkts_median": "Gói nguồn→đích TV",
    "spkts_std": "Gói nguồn→đích ĐLC",
    # dpkts
    "dpkts_mean": "Gói đích→nguồn TB",
    "dpkts_median": "Gói đích→nguồn TV",
    "dpkts_std": "Gói đích→nguồn ĐLC",
    # sbytes
    "sbytes_mean": "Bytes gửi TB",
    "sbytes_median": "Bytes gửi TV",
    "sbytes_std": "Bytes gửi ĐLC",
    # dbytes
    "dbytes_mean": "Bytes nhận TB",
    "dbytes_median": "Bytes nhận TV",
    "dbytes_std": "Bytes nhận ĐLC",
    # rate
    "rate_mean": "Tốc độ gói TB",
    "rate_median": "Tốc độ gói TV",
    "rate_std": "Tốc độ gói ĐLC",
    # sload
    "sload_mean": "Tải nguồn TB (bps)",
    "sload_median": "Tải nguồn TV (bps)",
    "sload_std": "Tải nguồn ĐLC",
    # dload
    "dload_mean": "Tải đích TB (bps)",
    "dload_median": "Tải đích TV (bps)",
    "dload_std": "Tải đích ĐLC",
    # sloss
    "sloss_mean": "Mất gói nguồn TB",
    "sloss_median": "Mất gói nguồn TV",
    "sloss_std": "Mất gói nguồn ĐLC",
    # dloss
    "dloss_mean": "Mất gói đích TB",
    "dloss_median": "Mất gói đích TV",
    "dloss_std": "Mất gói đích ĐLC",
    # sinpkt
    "sinpkt_mean": "Giãn cách gói nguồn TB",
    "sinpkt_median": "Giãn cách gói nguồn TV",
    "sinpkt_std": "Giãn cách gói nguồn ĐLC",
    # dinpkt
    "dinpkt_mean": "Giãn cách gói đích TB",
    "dinpkt_median": "Giãn cách gói đích TV",
    "dinpkt_std": "Giãn cách gói đích ĐLC",
    # sjit
    "sjit_mean": "Jitter nguồn TB",
    "sjit_median": "Jitter nguồn TV",
    "sjit_std": "Jitter nguồn ĐLC",
    # djit
    "djit_mean": "Jitter đích TB",
    "djit_median": "Jitter đích TV",
    "djit_std": "Jitter đích ĐLC",
    # swin
    "swin_mean": "Cửa sổ TCP nguồn TB",
    "swin_median": "Cửa sổ TCP nguồn TV",
    "swin_std": "Cửa sổ TCP nguồn ĐLC",
    # dwin
    "dwin_mean": "Cửa sổ TCP đích TB",
    "dwin_median": "Cửa sổ TCP đích TV",
    "dwin_std": "Cửa sổ TCP đích ĐLC",
    # stcpb
    "stcpb_mean": "Seq TCP nguồn TB",
    "stcpb_median": "Seq TCP nguồn TV",
    "stcpb_std": "Seq TCP nguồn ĐLC",
    # dtcpb
    "dtcpb_mean": "Seq TCP đích TB",
    "dtcpb_median": "Seq TCP đích TV",
    "dtcpb_std": "Seq TCP đích ĐLC",
    # tcprtt
    "tcprtt_mean": "RTT TCP TB (s)",
    "tcprtt_median": "RTT TCP TV (s)",
    "tcprtt_std": "RTT TCP ĐLC",
    # synack
    "synack_mean": "SYN→SYN-ACK TB (s)",
    "synack_median": "SYN→SYN-ACK TV (s)",
    "synack_std": "SYN→SYN-ACK ĐLC",
    # ackdat
    "ackdat_mean": "SYN-ACK→ACK TB (s)",
    "ackdat_median": "SYN-ACK→ACK TV (s)",
    "ackdat_std": "SYN-ACK→ACK ĐLC",
    # smean
    "smean_mean": "Cỡ gói nguồn TB",
    "smean_median": "Cỡ gói nguồn TV",
    "smean_std": "Cỡ gói nguồn ĐLC",
    # dmean
    "dmean_mean": "Cỡ gói đích TB",
    "dmean_median": "Cỡ gói đích TV",
    "dmean_std": "Cỡ gói đích ĐLC",
    # trans_depth
    "trans_depth_mean": "Độ sâu HTTP TB",
    "trans_depth_median": "Độ sâu HTTP TV",
    "trans_depth_std": "Độ sâu HTTP ĐLC",
    # response_body_len
    "response_body_len_mean": "Dung lượng phản hồi TB",
    "response_body_len_median": "Dung lượng phản hồi TV",
    "response_body_len_std": "Dung lượng phản hồi ĐLC",
    # ct_src_dport_ltm
    "ct_src_dport_ltm_mean": "Đếm IP nguồn+cổng đích TB",
    "ct_src_dport_ltm_median": "Đếm IP nguồn+cổng đích TV",
    "ct_src_dport_ltm_std": "Đếm IP nguồn+cổng đích ĐLC",
    # ct_dst_sport_ltm
    "ct_dst_sport_ltm_mean": "Đếm IP đích+cổng nguồn TB",
    "ct_dst_sport_ltm_median": "Đếm IP đích+cổng nguồn TV",
    "ct_dst_sport_ltm_std": "Đếm IP đích+cổng nguồn ĐLC",
    # is_ftp_login
    "is_ftp_login_mean": "Đăng nhập FTP TB",
    "is_ftp_login_median": "Đăng nhập FTP TV",
    "is_ftp_login_std": "Đăng nhập FTP ĐLC",
    # ct_ftp_cmd
    "ct_ftp_cmd_mean": "Số lệnh FTP TB",
    "ct_ftp_cmd_median": "Số lệnh FTP TV",
    "ct_ftp_cmd_std": "Số lệnh FTP ĐLC",
    # ct_flw_http_mthd
    "ct_flw_http_mthd_mean": "Phương thức HTTP TB",
    "ct_flw_http_mthd_median": "Phương thức HTTP TV",
    "ct_flw_http_mthd_std": "Phương thức HTTP ĐLC",
    # is_sm_ips_ports
    "is_sm_ips_ports_mean": "Trùng IP&Cổng TB",
    "is_sm_ips_ports_median": "Trùng IP&Cổng TV",
    "is_sm_ips_ports_std": "Trùng IP&Cổng ĐLC",

    # ── UNSW-NB15: các bảng EDA khác ──
    "attack_cat": "Loại tấn công",
    "count": "Số lượng",
    "percent": "Phần trăm (%)",
    "split": "Tập dữ liệu",
    "percent_within_split": "% trong tập",
    "feature": "Đặc trưng",
    "value": "Giá trị",
    "dataset": "Tập dữ liệu",
    "source": "Nguồn",
    "rows": "Số dòng",
    "train_rows": "Dòng train",
    "test_rows": "Dòng test",
    "feature_columns": "Số cột đặc trưng",
    "numeric_features": "Đặc trưng số",
    "categorical_features": "Đặc trưng phân loại",
    "missing_values": "Giá trị thiếu",
    "infinite_numeric_values": "Giá trị vô cực",
    "duplicate_rows": "Dòng trùng lặp",
    "distinct_feature_signatures_in_both_splits": "Chữ ký đặc trưng chung",
    "test_rows_with_feature_signature_in_train": "Dòng test trùng train",
    "test_rows_with_feature_signature_in_train_percent": "% dòng test trùng train",
    "metric": "Chỉ số",
    "percent_of_split": "% trong tập",
}

EDA_TITLES = {
    # CIC-IDS2017
    "bytes_per_sec_by_class": "Tốc độ truyền (Bytes/s theo lớp)",
    "class_distribution": "Phân bố số lượng các lớp",
    "duration_by_class": "Thời lượng luồng (Duration theo lớp)",
    "tcp_flags_by_class": "Số cờ TCP trung bình theo lớp",
    "top_ports_by_class": "Top 10 cổng đích bị tấn công",
    "summary": "Bảng tổng hợp đặc trưng theo lớp",

    # UNSW-NB15
    "attack_category_distribution": "Phân bố 9 loại tấn công (Attack Categories)",
    "binary_label_distribution": "Phân bố nhãn nhị phân (Bình thường vs Tấn công)",
    "categorical_proto": "Phân bố giao thức mạng (TCP, UDP...)",
    "categorical_service": "Phân bố dịch vụ ứng dụng (HTTP, DNS...)",
    "categorical_state": "Phân bố trạng thái kết nối (FIN, CON, INT...)",
    "numeric_correlation": "Ma trận tương quan giữa các đặc trưng số",
    "numeric_features_by_label": "So sánh đặc trưng số giữa Bình thường vs Tấn công",
    "dataset_overview": "Tổng quan tập dữ liệu (Kích thước, số cột)",
    "numeric_summary_by_label": "Thống kê chi tiết đặc trưng số theo nhãn",
    "split_feature_overlap": "Độ trùng lặp đặc trưng giữa tập Train và Test",
    "categorical_feature_distribution": "Thống kê tần suất các đặc trưng phân loại",
}

EDA_DESCRIPTIONS = {
    # CIC-IDS2017
    "bytes_per_sec_by_class": "So sánh tốc độ truyền dữ liệu (bytes/giây) theo thang log giữa BENIGN, DDoS và PortScan.",
    "class_distribution": "Số lượng luồng mạng thực tế của từng lớp: BENIGN (224.978), PortScan (158.804), DDoS (128.025).",
    "duration_by_class": "Thời lượng kết nối: PortScan siêu ngắn (47 µs), DDoS kéo dài (~1.88s) để làm nghẽn dịch vụ.",
    "tcp_flags_by_class": "Tần suất xuất hiện các cờ TCP (SYN, ACK, RST, FIN...) giúp phân biệt hành vi tấn công.",
    "top_ports_by_class": "Top các cổng đích hay bị nhắm tới: PortScan quét 1.000 cổng, DDoS nhắm vào 4 cổng chính.",
    "summary": "Bảng tổng hợp các chỉ số trung bình và trung vị của các đặc trưng mạng theo từng lớp.",

    # UNSW-NB15
    "attack_category_distribution": "Tỷ lệ lưu lượng Normal (36.1%) cùng 9 họ tấn công mạng (Generic, Exploits, Fuzzers, DoS...).",
    "binary_label_distribution": "Phân bố nhãn 0 (Normal) và 1 (Attack) trên 2 tập train và test chính thức của UNSW-NB15.",
    "categorical_proto": "Thống kê các giao thức mạng: TCP và UDP chiếm đa số, cùng các giao thức quét mạng đặc thù.",
    "categorical_service": "Các dịch vụ ứng dụng mạng: HTTP và DNS là hai mục tiêu bị khai thác tấn công nhiều nhất.",
    "categorical_state": "Trạng thái phiên kết nối: Tấn công thường có trạng thái gián đoạn (INT) hoặc reset (RST) cao bất thường.",
    "numeric_correlation": "Biểu đồ nhiệt biểu thị hệ số tương quan tuyến tính giữa 31 đặc trưng số học.",
    "numeric_features_by_label": "So sánh các chỉ số (rate, sload, dur) giữa lưu lượng bình thường và lưu lượng tấn công.",
    "dataset_overview": "Tổng quan về số dòng train/test, số đặc trưng số và biến phân loại của UNSW-NB15.",
    "numeric_summary_by_label": "Bảng thống kê trung bình, trung vị và độ lệch chuẩn của các đặc trưng số theo từng nhãn.",
    "split_feature_overlap": "Độ trùng lặp chữ ký đặc trưng giữa tập train và test giúp kiểm tra tính tổng quát hóa.",
    "categorical_feature_distribution": "Bảng tần suất và tỷ lệ phần trăm của từng giá trị trong các biến phân loại.",
}

FEATURE_IMPORTANCE = {
    "cic": [
        {"feature": "Avg Fwd Segment Size", "importance": 8.85, "role": "Kích thước phân đoạn gửi đi: PortScan siêu nhỏ không payload, Benign biến thiên tự nhiên."},
        {"feature": "Fwd Packet Length Max", "importance": 8.51, "role": "Chiều dài gói lớn nhất: Phân biệt gói thăm dò nhỏ với gói truyền dữ liệu thực tế."},
        {"feature": "Total Length of Fwd Packets", "importance": 7.58, "role": "Tổng số byte gửi đi: Đánh giá khối lượng lưu lượng được bơm vào máy chủ mục tiêu."},
        {"feature": "Subflow Fwd Bytes", "importance": 7.28, "role": "Số byte luồng con: Tấn công DDoS tạo luồng con có dung lượng bất thường."},
        {"feature": "Fwd Packet Length Mean", "importance": 6.56, "role": "Kích thước gói trung bình: Nhận diện công cụ tấn công dùng gói kích thước cố định."},
        {"feature": "Average Packet Size", "importance": 4.23, "role": "Kích thước gói trung bình 2 chiều: Đánh giá tỷ lệ trao đổi dữ liệu toàn phiên."},
        {"feature": "Packet Length Mean", "importance": 3.97, "role": "Độ dài trung bình các gói tin trao đổi trong toàn bộ luồng mạng."},
        {"feature": "Flow Duration", "importance": 2.98, "role": "Thời lượng kết nối: PortScan siêu ngắn (47 µs), DDoS kéo dài (~1.88s) để nghẽn tài nguyên."},
        {"feature": "Flow IAT Max", "importance": 2.97, "role": "Khoảng cách thời gian lớn nhất giữa 2 gói: Phân biệt bot tự động với thao tác người dùng."},
        {"feature": "act_data_pkt_fwd", "importance": 2.68, "role": "Số gói tin thực sự chứa payload: Tấn công thường chỉ gửi gói cờ rác mà không có dữ liệu thật."},
    ],
    "unsw": [
        {"feature": "sload", "importance": 8.07, "role": "Tải thông lượng nguồn (bps): Kẻ tấn công bơm lưu lượng hàng chục Mbps, gấp hàng trăm lần người dùng thật."},
        {"feature": "rate", "importance": 6.80, "role": "Tốc độ gửi gói (gói/s): Tấn công bắn tới 111.111 gói/s, cao gấp 170 lần người dùng thật (642 gói/s)."},
        {"feature": "sbytes", "importance": 6.27, "role": "Tổng số byte gửi từ nguồn: Phát hiện các đợt bơm dữ liệu hoặc quét mạng bất thường."},
        {"feature": "smean", "importance": 5.57, "role": "Kích thước trung bình gói nguồn: Nhận diện cấu trúc gói tin cố định của mã độc."},
        {"feature": "dload", "importance": 4.94, "role": "Tải thông lượng đích phản hồi: Phân biệt luồng phản hồi thật với các cuộc tấn công câm."},
        {"feature": "dbytes", "importance": 4.71, "role": "Tổng byte đích nhận về: Người dùng bình thường có trao đổi 2 chiều, tấn công thường chỉ 1 chiều."},
        {"feature": "dur", "importance": 4.64, "role": "Thời lượng kết nối: Tấn công tự động diễn ra chớp nhoáng (9 µs) khác với người dùng lướt web (~54 ms)."},
        {"feature": "synack", "importance": 4.48, "role": "Thời gian trễ SYN đến SYN-ACK: Kẻ tấn công quét cổng thường ngắt ngay, không chờ bắt tay TCP."},
        {"feature": "tcprtt", "importance": 4.46, "role": "Thời gian trễ khứ hồi TCP (Round-Trip Time): Bị dị thường hoặc bằng 0 trong các đợt Scan/DoS."},
        {"feature": "dinpkt", "importance": 4.33, "role": "Thời gian giãn cách giữa các gói nhận về: Phản ánh nhịp độ phản hồi của hệ thống mạng."},
    ]
}

BEHAVIOR_ANALYSIS_HTML = {
    "cic": """
<div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #2c3e50;">
  <h3 style="color: #1b6c80; margin-top: 0; margin-bottom: 12px; font-size: 15px;">
    2. Phân tích hành vi chi tiết của từng lớp (Rất quan trọng khi báo cáo):
  </h3>
  
  <div style="margin-bottom: 14px; padding: 12px; background: #ffffff; border-left: 4px solid #2ecc71; border-radius: 4px; border: 1px solid #e1e8ed; border-left-width: 4px;">
    <h4 style="margin: 0 0 6px 0; color: #27ae60; font-size: 14px;">
      Lớp 1: <b style="background: #e8f8f5; color: #27ae60; padding: 2px 6px; border-radius: 3px;">BENIGN</b> — Người dùng mạng bình thường
    </h4>
    <ul style="margin: 0; padding-left: 20px; font-size: 13px;">
      <li><b>Quy mô:</b> Chiếm <b>224.978 flows</b> (~44% tổng tập dữ liệu).</li>
      <li><b>Thời lượng:</b> Trung vị là <b>50.195 µs (~0.05 giây)</b> &rarr; thời gian phản hồi lướt web tiêu chuẩn.</li>
      <li><b>Gói tin 2 chiều:</b> Trung bình <b>~6.05 gói đi / ~6.51 gói về</b> &rarr; có sự tương tác qua lại đầy đủ giữa máy khách và máy chủ.</li>
      <li><b>Tốc độ truyền:</b> Trung vị đạt <b>4.270 Bytes/giây</b> &rarr; phản ánh lưu lượng truyền tải dữ liệu thật (HTML, CSS, hình ảnh).</li>
      <li><b>Số cổng đích:</b> Phân tán trên <b>30.794 cổng khác nhau</b> &rarr; Người dùng bình thường truy cập rất nhiều dịch vụ mạng đa dạng.</li>
    </ul>
  </div>

  <div style="margin-bottom: 14px; padding: 12px; background: #ffffff; border-left: 4px solid #e74c3c; border-radius: 4px; border: 1px solid #e1e8ed; border-left-width: 4px;">
    <h4 style="margin: 0 0 6px 0; color: #c0392b; font-size: 14px;">
      Lớp 2: <b style="background: #fdedec; color: #c0392b; padding: 2px 6px; border-radius: 3px;">DDoS</b> — Tấn công từ chối dịch vụ
    </h4>
    <ul style="margin: 0; padding-left: 20px; font-size: 13px;">
      <li><b>Quy mô:</b> Chiếm <b>128.025 flows</b> (~25%).</li>
      <li><b>Thời lượng kéo dài bất thường:</b> Trung vị lên đến <b>1.878.984 µs (~1.88 giây)</b> — dài hơn người dùng bình thường tới <b>gần 40 lần!</b>
        <br><span style="color: #555;">• <i>Lý do:</i> Kẻ tấn công cố tình gửi gói tin nhỏ nhưng giữ phiên kết nối thật lâu (dạng tấn công cạn kiệt tài nguyên / Slowloris) nhằm làm hết bộ nhớ RAM của máy chủ.</span>
      </li>
      <li><b>Tốc độ truyền rất thấp:</b> Chỉ <b>160 Bytes/giây</b> (chỉ gửi gói tin rác để duy trì kết nối).</li>
      <li><b>Số cổng đích:</b> Chỉ nhắm vào đúng <b>4 cổng</b> &rarr; Dấu hiệu rõ ràng của DDoS: tập trung toàn lực đánh sập đúng các cổng dịch vụ web chính (như port 80, 443, 8080).</li>
    </ul>
  </div>

  <div style="margin-bottom: 8px; padding: 12px; background: #ffffff; border-left: 4px solid #e67e22; border-radius: 4px; border: 1px solid #e1e8ed; border-left-width: 4px;">
    <h4 style="margin: 0 0 6px 0; color: #d35400; font-size: 14px;">
      Lớp 3: <b style="background: #fef5e7; color: #d35400; padding: 2px 6px; border-radius: 3px;">PortScan</b> — Tấn công quét cổng dò lỗ hổng
    </h4>
    <ul style="margin: 0; padding-left: 20px; font-size: 13px;">
      <li><b>Quy mô:</b> Chiếm <b>158.804 flows</b> (~31%).</li>
      <li><b>Thời lượng siêu ngắn:</b> Trung vị chỉ đúng <b>47 micro-giây (0.000047 giây)</b> &rarr; Nhanh như chớp!</li>
      <li><b>Gói tin tối thiểu:</b> Đúng <b>1.02 gói đi và 1.00 gói về</b> &rarr; Kẻ tấn công chỉ gửi đúng 1 gói thăm dò "gõ cửa" xem cổng có mở không rồi ngắt ngay lập tức.</li>
      <li><b>Số cổng đích:</b> Quét đúng tròn <b>1.000 cổng</b> &rarr; Đây là "chữ ký" kinh điển của công cụ quét mạng nổi tiếng <b>Nmap</b> (mặc định Nmap quét đúng Top 1.000 cổng thông dụng nhất).</li>
    </ul>
  </div>
</div>
""",
    "unsw": """
<div style="font-family: 'Segoe UI', Arial, sans-serif; line-height: 1.6; color: #2c3e50;">
  <h3 style="color: #1b6c80; margin-top: 0; margin-bottom: 12px; font-size: 15px;">
    2. Phân tích hành vi chi tiết theo nhãn (UNSW-NB15) — Rất quan trọng khi báo cáo:
  </h3>
  
  <div style="margin-bottom: 14px; padding: 12px; background: #ffffff; border-left: 4px solid #2ecc71; border-radius: 4px; border: 1px solid #e1e8ed; border-left-width: 4px;">
    <h4 style="margin: 0 0 6px 0; color: #27ae60; font-size: 14px;">
      Nhãn 0: <b style="background: #e8f8f5; color: #27ae60; padding: 2px 6px; border-radius: 3px;">BENIGN</b> — Người dùng mạng thông thường
    </h4>
    <ul style="margin: 0; padding-left: 20px; font-size: 13px;">
      <li><b>Quy mô:</b> Chiếm <b>93.000 dòng</b> (36.1% tổng dữ liệu; 31.9% tập Train, 44.9% tập Test).</li>
      <li><b>Tốc độ gửi gói (rate):</b> Trung vị chỉ <b>642 gói/giây</b> &rarr; nhịp độ truyền tin tự nhiên của người dùng thật.</li>
      <li><b>Tải nguồn gửi đi (sload):</b> Trung vị khoảng <b>298 kbps</b> &rarr; mức tiêu thụ băng thông nhẹ nhàng khi lướt web.</li>
      <li><b>Thời lượng kết nối (dur):</b> Trung vị <b>~54 mili-giây</b> (0.0539s), trung bình ~1.01 giây &rarr; các phiên kết nối mạng diễn ra trọn vẹn và ổn định.</li>
      <li><b>Giao thức mạng:</b> Chiếm đa số bởi <b>TCP (47.8%)</b> và <b>UDP (36.0%)</b> trên các dịch vụ phổ biến (HTTP, DNS).</li>
      <li><b>Mức độ tập trung cổng (ct_src_dport_ltm):</b> Trung bình chỉ <b>1.78 kết nối</b> tới cùng một cổng trong 100 kết nối gần nhất &rarr; lưu lượng phân tán tự nhiên.</li>
    </ul>
  </div>

  <div style="margin-bottom: 8px; padding: 12px; background: #ffffff; border-left: 4px solid #e74c3c; border-radius: 4px; border: 1px solid #e1e8ed; border-left-width: 4px;">
    <h4 style="margin: 0 0 6px 0; color: #c0392b; font-size: 14px;">
      Nhãn 1: <b style="background: #fdedec; color: #c0392b; padding: 2px 6px; border-radius: 3px;">ATTACK</b> — Lưu lượng tấn công xâm nhập mạng
    </h4>
    <ul style="margin: 0; padding-left: 20px; font-size: 13px;">
      <li><b>Quy mô:</b> Chiếm <b>164.673 dòng</b> (63.9% tổng dữ liệu; 68.1% tập Train, 55.1% tập Test).</li>
      <li><b>9 họ tấn công mạng:</b> <i>Generic (22.8%), Exploits (17.3%), Fuzzers (9.4%), DoS (6.3%), Reconnaissance (5.4%), Analysis, Backdoor, Shellcode, Worms.</i></li>
      <li><b>Tốc độ gửi gói bùng nổ (rate):</b> Trung vị lên tới <b>111.111 gói/giây</b> — cao gấp <b>hơn 170 lần</b> so với người dùng bình thường!</li>
      <li><b>Tải nguồn cực lớn (sload):</b> Trung vị đạt <b>~50.6 Mbps</b> &rarr; kẻ tấn công bơm lượng dữ liệu cực lớn để làm quá tải hoặc khai thác tràn bộ đệm.</li>
      <li><b>Thời lượng siêu ngắn (dur):</b> Trung vị chỉ đúng <b>9 micro-giây (0.000009s)</b> &rarr; các đợt quét tự động diễn ra chớp nhoáng rồi ngắt kết nối.</li>
      <li><b>Bắt tay TCP dị thường (synack / tcprtt):</b> Kẻ tấn công quét cổng thường chỉ gửi gói thăm dò rồi ngắt ngay, không chờ bắt tay 3 bước TCP trọn vẹn.</li>
      <li><b>Tập trung cổng dồn dập (ct_src_dport_ltm):</b> Trung bình tới <b>7.19 kết nối</b> liên tục vào cùng 1 cổng đích trong 100 kết nối gần nhất &rarr; dấu hiệu bắn phá dồn dập vào 1 mục tiêu.</li>
    </ul>
  </div>
</div>
"""
}


def result_dir(root, dataset):
    if dataset not in DATASETS:
        raise ValueError(f"Bộ dữ liệu không hợp lệ: {dataset}")
    return Path(root) / "results" / ("unsw_nb15" if dataset == "unsw" else "")


def read_csv(path):
    try:
        return pd.read_csv(path), ""
    except (OSError, ValueError) as error:
        return pd.DataFrame(), f"Không đọc được {Path(path).name}: {error}"


def load_results(root, dataset):
    folder = result_dir(root, dataset)
    frames, issues = [], []
    for stem, group, names in SOURCES:
        frame, error = read_csv(folder / f"{stem}_metrics.csv")
        if error:
            issues.append(error)
            continue
        if not {"model", "accuracy"}.issubset(frame.columns):
            issues.append(f"{stem}_metrics.csv thiếu cột model hoặc accuracy.")
            continue
        if "dataset" in frame:
            frame = frame.loc[frame["dataset"] == dataset].copy()
        frame = frame.loc[frame["model"].isin(names)].copy()
        for column in [*METRICS, "train_time_s", "predict_time_s"]:
            frame[column] = pd.to_numeric(frame.get(column, float("nan")), errors="coerce")
            invalid = ~frame[column].between(0, 1) if column in METRICS else frame[column] < 0
            if (invalid & frame[column].notna()).any():
                issues.append(f"{stem}_metrics.csv có giá trị {column} không hợp lệ.")
                frame.loc[invalid, column] = float("nan")
        try:
            meta = json.loads((folder / f"{stem}_meta.json").read_text(encoding="utf-8"))
            if not isinstance(meta, dict):
                raise ValueError("metadata phải là object")
        except (OSError, ValueError) as error:
            meta = {}
            issues.append(f"Không đọc được {stem}_meta.json: {error}")
        train, test = meta.get("train"), meta.get("test")
        split = f"{train:,} / {test:,}" if isinstance(train, int) and isinstance(test, int) else "Chưa có"
        protocol = "Split chính thức" if "official" in str(meta.get("split", "")) else "Chia ngẫu nhiên"
        frame["_protocol"] = f"{protocol} · {split}"
        frame["_group"], frame["_stem"] = group, stem
        frame["_meta"] = [meta] * len(frame)
        frames.append(frame)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), issues


def confusion_path(folder, model):
    slug = {"Random Forest": "random_forest", "XGBoost": "xgboost", "MLP": "mlp"}.get(model, model)
    return Path(folder) / f"cm_{slug}.png"


def per_class(folder, row):
    frame, error = read_csv(Path(folder) / f"{row['_stem']}_per_class.csv")
    if error or not {"model", "class"}.issubset(frame.columns):
        return pd.DataFrame(), error or "Bảng chỉ số từng lớp thiếu cột model/class."
    frame = frame.loc[frame["model"] == row["model"]].drop(columns="model").copy()
    classes = row["_meta"].get("classes", [])
    mapping = {str(index): name for index, name in enumerate(classes)}
    frame["class"] = frame["class"].astype(str).replace(mapping)
    return frame, ""


def eda_label(stem):
    return EDA_TITLES.get(stem, stem.replace("_", " "))


def localize_eda_table(frame, stem):
    table = frame.rename(columns={
        column: COLUMN_TRANSLATIONS[column]
        for column in frame.columns
        if column in COLUMN_TRANSLATIONS
    }).copy()
    if stem == "categorical_feature_distribution" and "Đặc trưng" in table:
        table["Đặc trưng"] = table["Đặc trưng"].replace(EDA_CATEGORICAL_FEATURES)
    if stem == "split_feature_overlap" and "Chỉ số" in table:
        table["Chỉ số"] = table["Chỉ số"].replace(EDA_METRIC_NAMES)
    if "Nhãn" in table:
        table["Nhãn"] = table["Nhãn"].replace(EDA_LABEL_VALUES)
    if "Tập dữ liệu" in table:
        table["Tập dữ liệu"] = table["Tập dữ liệu"].replace(EDA_SPLIT_VALUES)
    return table
