"""Run with python giao_dien/main.py or python -m giao_dien.main."""
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from PySide6.QtCore import Qt, QUrl, QThread, Signal
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QHeaderView, QLabel, QListWidget, QMainWindow,
    QPushButton, QSplitter, QStackedWidget, QTabWidget, QTableWidget, QFileDialog,
    QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import get_data_path
from matplotlib.ticker import PercentFormatter

from giao_dien.data import (
    BEHAVIOR_ANALYSIS_HTML, DATASETS, EDA_DESCRIPTIONS, FEATURE_IMPORTANCE,
    METRICS, MODEL_NAMES, confusion_path, eda_label, load_results,
    localize_eda_table, per_class, read_csv, result_dir,
)
from giao_dien.prediction import model_paths, predict_file

ROOT = Path(__file__).resolve().parent.parent
ACCENT = "#1b6c80"
STYLE = """
QWidget { color: #131416; font-family: 'Segoe UI'; font-size: 14px; }
QMainWindow, QWidget#frame { background: #f3f7f9; }
QFrame#sheet { background: white; border: 1px solid #e8eaec; border-radius: 16px; }
QLabel#title { font-size: 22px; font-weight: 600; }
QLabel#brand { font-size: 21px; font-weight: 600; }
QLabel#section { font-size: 16px; font-weight: 600; }
QLabel#muted { color: #5d5f60; font-size: 12px; }
QLabel#notice { background: #f4f6f8; padding: 12px; border-radius: 6px; color: #42484b; }
QLabel#error { color: #92400e; font-size: 12px; }
QListWidget { background: transparent; border: none; outline: none; }
QListWidget::item { padding: 12px; border-radius: 6px; margin-bottom: 4px; }
QListWidget::item:selected { background: #e6ebee; color: #131416; }
QListWidget::item:hover { background: #ebeff1; }
QComboBox, QPushButton { min-height: 32px; padding: 2px 10px; border: 1px solid #cdced1;
    border-radius: 6px; background: white; }
QComboBox::drop-down { border: none; width: 24px; }
QPushButton:hover { background: #f0f3f4; border-color: #858b8e; }
QPushButton:pressed { background: #e6ebee; }
QPushButton:focus, QComboBox:focus, QTableWidget:focus { border: 1px solid #1b6c80; }
QPushButton:disabled { color: #74797c; background: #f4f6f8; }
QTableWidget { border: 1px solid #e8eaec; border-radius: 6px; background: white;
    selection-background-color: #e6edef; selection-color: #131416; gridline-color: #f0f1f2; }
QHeaderView::section { background: white; color: #5d5f60; border: none; padding: 10px 8px;
    font-size: 12px; }
QTableWidget::item { padding: 6px 8px; border-bottom: 1px solid #f0f1f2; }
QGraphicsView { background: white; border: 1px solid #e8eaec; border-radius: 6px; }
QSplitter::handle { background: #f4f6f8; }
"""


def label(text, role="", wrap=False):
    widget = QLabel(text)
    widget.setObjectName(role)
    widget.setWordWrap(wrap)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    return widget


class NumberItem(QTableWidgetItem):
    def __lt__(self, other):
        return self.data(Qt.ItemDataRole.UserRole) < other.data(Qt.ItemDataRole.UserRole)


def fill_table(table, frame, percentages=()):
    table.setSortingEnabled(False)
    table.clear()
    table.setRowCount(len(frame))
    table.setColumnCount(len(frame.columns))
    table.setHorizontalHeaderLabels([str(column) for column in frame.columns])
    for column, name in enumerate(frame.columns):
        numeric = pd.api.types.is_numeric_dtype(frame[name])
        for row, value in enumerate(frame[name]):
            if pd.isna(value):
                item = NumberItem("—") if numeric else QTableWidgetItem("—")
                item.setData(Qt.ItemDataRole.UserRole, float("-inf"))
            elif numeric:
                text = f"{value * 100:.4f}%" if name in percentages else (
                    f"{value:,.0f}" if name in {"support", "Số mẫu"} else f"{value:,.4f}".rstrip("0").rstrip("."))
                item = NumberItem(text)
                item.setData(Qt.ItemDataRole.UserRole, float(value))
            else:
                item = QTableWidgetItem(str(value))
            item.setToolTip(item.text())
            if numeric:
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            table.setItem(row, column, item)
        header = table.horizontalHeaderItem(column)
        header.setTextAlignment((Qt.AlignmentFlag.AlignRight if numeric else Qt.AlignmentFlag.AlignLeft)
                                | Qt.AlignmentFlag.AlignVCenter)
    table.resizeColumnsToContents()
    if len(frame.columns):
        table.horizontalHeader().setMinimumSectionSize(85)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
    table.setSortingEnabled(True)


def table_widget():
    table = QTableWidget()
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setShowGrid(False)
    table.verticalHeader().hide()
    table.verticalHeader().setDefaultSectionSize(42)
    return table


class ImageView(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.picture = QGraphicsPixmapItem()
        self.scene().addItem(self.picture)
        self.path = None
        self.setMinimumSize(320, 240)
        self.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

    def show_path(self, path):
        self.path = Path(path) if path else None
        self.scene().clear()
        self.picture = QGraphicsPixmapItem()
        pixmap = QPixmap(str(path)) if path else QPixmap()
        if pixmap.isNull():
            self.path = None
            self.scene().addText("Chưa có hình kết quả." if not path else f"Không đọc được hình: {Path(path).name}")
        else:
            self.picture.setPixmap(pixmap)
            self.scene().addItem(self.picture)
        self.scene().setSceneRect(self.scene().itemsBoundingRect())
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class PredictionWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, root, dataset, model, path, parent):
        super().__init__(parent)
        self.arguments = root, dataset, model, path

    def run(self):
        try:
            self.succeeded.emit(predict_file(*self.arguments))
        except Exception as error:
            self.failed.emit(str(error))


class Dashboard(QMainWindow):
    def __init__(self, root=ROOT):
        super().__init__()
        self.root, self.results = Path(root), pd.DataFrame()
        self.setWindowTitle("PBL ATTT — Kết quả học máy")
        self.resize(1440, 900)
        self.setMinimumSize(1024, 740)
        frame = QWidget(objectName="frame")
        outer = QHBoxLayout(frame)
        outer.setContentsMargins(0, 0, 8, 8)
        outer.setSpacing(0)
        self.setCentralWidget(frame)
        sidebar = QWidget()
        sidebar.setFixedWidth(228)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(24, 28, 24, 24)
        side.setSpacing(16)
        side.addWidget(label("PBL ATTT", "brand"))
        side.addWidget(label("Phát hiện tấn công mạng", "muted"))
        side.addSpacing(24)
        side.addWidget(label("Bộ dữ liệu", "muted"))
        self.dataset = QComboBox()
        for key, name in DATASETS.items():
            self.dataset.addItem(name, key)
        self.dataset.setAccessibleName("Bộ dữ liệu")
        side.addWidget(self.dataset)
        side.addSpacing(12)
        self.navigation = QListWidget()
        self.navigation.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.navigation.addItems(["So sánh mô hình", "Chi tiết mô hình", "EDA dữ liệu", "Phân tích đặc trưng", "Dự đoán dữ liệu mới"])
        self.navigation.setCurrentRow(0)
        side.addWidget(self.navigation)
        side.addWidget(label("Kết quả lưu trên máy\nCIC-IDS2017 / UNSW-NB15", "muted", True))
        outer.addWidget(sidebar)
        sheet = QFrame(objectName="sheet")
        layout = QVBoxLayout(sheet)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(16)
        title_row = QHBoxLayout()
        self.title = label("So sánh mô hình", "title")
        title_row.addWidget(self.title)
        title_row.addStretch()
        refresh = QPushButton("Làm mới kết quả")
        refresh.clicked.connect(self.refresh)
        title_row.addWidget(refresh)
        layout.addLayout(title_row)
        self.freshness = label("", "muted")
        layout.addWidget(self.freshness)
        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)
        self.build_comparison()
        self.build_details()
        self.build_eda()
        self.build_features()
        self.build_prediction()
        outer.addWidget(sheet, 1)
        self.navigation.currentRowChanged.connect(self.change_page)
        self.dataset.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def change_page(self, index):
        self.pages.setCurrentIndex(index)
        self.title.setText(self.navigation.item(index).text())

    def build_prediction(self):
        self.prediction_result = pd.DataFrame()
        self.prediction_path = None
        self.prediction_worker = None
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        layout.addWidget(label(
            "Nhập CSV/CSV.GZ có đúng các đặc trưng và đơn vị như dữ liệu huấn luyện. "
            "Cột Lớp dự đoán trả về một lớp cụ thể: CIC có Bình thường/DDoS/PortScan; "
            "UNSW có 10 lớp gồm Bình thường (Normal) và 9 loại tấn công. "
            "Cột nhãn (nếu có) không dùng để dự đoán. PCAP cần trích đặc trưng trước.",
            "notice", True))
        controls = QHBoxLayout()
        controls.addWidget(label("Model"))
        self.prediction_model = QComboBox()
        self.prediction_model.setAccessibleName("Model dự đoán")
        controls.addWidget(self.prediction_model, 1)
        self.pick_prediction = QPushButton("Chọn file dữ liệu…")
        self.pick_prediction.clicked.connect(self.choose_prediction_file)
        controls.addWidget(self.pick_prediction)
        self.run_prediction = QPushButton("Dự đoán")
        self.run_prediction.clicked.connect(self.start_prediction)
        controls.addWidget(self.run_prediction)
        layout.addLayout(controls)
        self.prediction_file_label = label("Chưa chọn file dữ liệu.", "muted", True)
        layout.addWidget(self.prediction_file_label)
        self.prediction_status = label("Chọn bộ dữ liệu ở bên trái, model và file để bắt đầu.", "notice", True)
        layout.addWidget(self.prediction_status)
        self.prediction_table = table_widget()
        layout.addWidget(self.prediction_table, 1)
        layout.addWidget(label("Xác suất là điểm do model trả về, không phải độ chính xác đã được kiểm chứng.", "muted", True))
        self.export_prediction = QPushButton("Xuất toàn bộ kết quả CSV…")
        self.export_prediction.setEnabled(False)
        self.export_prediction.clicked.connect(self.save_prediction)
        layout.addWidget(self.export_prediction)
        self.prediction_model.currentIndexChanged.connect(self.clear_prediction)
        self.pages.addWidget(page)

    def clear_prediction(self):
        self.prediction_result = pd.DataFrame()
        self.export_prediction.setEnabled(False)
        fill_table(self.prediction_table, pd.DataFrame())
        self.run_prediction.setEnabled(bool(self.prediction_path and self.prediction_model.count()))
        self.prediction_status.setText("Sẵn sàng dự đoán." if self.prediction_path else "Chọn file CSV/CSV.GZ để bắt đầu.")

    def choose_prediction_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn dữ liệu mạng", str(self.root / "data"),
                                            "Dữ liệu CSV (*.csv *.csv.gz)")
        if path:
            self.prediction_path = Path(path)
            self.prediction_file_label.setText(str(self.prediction_path))
            self.clear_prediction()

    def start_prediction(self):
        if not self.prediction_path or not self.prediction_model.currentData():
            return
        self.clear_prediction()
        for control in (self.dataset, self.prediction_model, self.pick_prediction, self.run_prediction):
            control.setEnabled(False)
        self.prediction_status.setText("Đang kiểm tra dữ liệu và chạy model…")
        self.prediction_worker = PredictionWorker(
            self.root, self.dataset.currentData(), self.prediction_model.currentData(),
            self.prediction_path, self)
        self.prediction_worker.succeeded.connect(self.show_prediction)
        self.prediction_worker.failed.connect(self.prediction_status.setText)
        self.prediction_worker.finished.connect(self.finish_prediction)
        self.prediction_worker.start()

    def show_prediction(self, result):
        self.prediction_result = result
        preview = result.head(500).drop(columns=["prediction", "prediction_code", "prediction_status", "attack_type"]).rename(columns={
            "prediction_class": "Lớp dự đoán",
            "prediction_probability": "Xác suất model", "prediction_dataset": "Dataset",
            "prediction_model": "Model"})
        fill_table(self.prediction_table, preview, percentages=("Xác suất model",))
        if "Xác suất model" in preview:
            self.prediction_table.setColumnWidth(preview.columns.get_loc("Xác suất model"), 160)
        self.export_prediction.setEnabled(True)
        counts = "; ".join(f"{name}: {count:,}" for name, count in result.prediction_class.value_counts().items())
        self.prediction_status.setText(f"Đã dự đoán {len(result):,} dòng · {counts}. "
                                       f"Hiển thị {len(preview):,} dòng đầu; xuất CSV để xem toàn bộ. "
                                       + result.attrs.get("model_warning", ""))

    def finish_prediction(self):
        for control in (self.dataset, self.prediction_model, self.pick_prediction):
            control.setEnabled(True)
        self.run_prediction.setEnabled(bool(self.prediction_path and self.prediction_model.count()))
        self.prediction_worker.deleteLater()
        self.prediction_worker = None

    def save_prediction(self):
        if self.prediction_result.empty:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Xuất kết quả", str(self.root / "predictions.csv"), "CSV (*.csv)")
        if path:
            if Path(path).resolve() == self.prediction_path.resolve():
                self.prediction_status.setText("Chọn file xuất khác để không ghi đè dữ liệu đầu vào.")
                return
            try:
                self.prediction_result.to_csv(path, index=False, encoding="utf-8-sig")
                self.prediction_status.setText(f"Đã lưu {len(self.prediction_result):,} dòng vào {path}")
            except OSError as error:
                self.prediction_status.setText(f"Không lưu được kết quả: {error}")

    def closeEvent(self, event):
        if self.prediction_worker is not None and self.prediction_worker.isRunning():
            self.prediction_status.setText("Đang dự đoán; hãy đợi hoàn tất trước khi đóng cửa sổ.")
            event.ignore()
        else:
            super().closeEvent(event)

    def build_comparison(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        self.notice = label("", "notice", True)
        layout.addWidget(self.notice)
        controls = QHBoxLayout()
        controls.addWidget(label("Chỉ số"))
        self.metric_choice = QComboBox()
        for key, name in METRICS.items():
            self.metric_choice.addItem(name, key)
        controls.addWidget(self.metric_choice)
        controls.addSpacing(12)
        controls.addWidget(label("Nhóm mô hình"))
        self.group = QComboBox()
        self.group.addItems(["Tất cả", "Baseline (5 mô hình ML)", "TabNet"])
        controls.addWidget(self.group)
        controls.addStretch()
        layout.addLayout(controls)
        self.figure = Figure(figsize=(9, 3.0), dpi=100, layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(230)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(label("Bảng kết quả", "section"))
        self.comparison = table_widget()
        self.comparison.setMinimumHeight(270)
        layout.addWidget(self.comparison, 1)
        self.errors = label("", "error", True)
        self.errors.setMaximumHeight(70)
        layout.addWidget(self.errors)
        self.metric_choice.currentIndexChanged.connect(self.update_comparison)
        self.group.currentIndexChanged.connect(self.update_comparison)
        self.pages.addWidget(page)

    def build_details(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.model_choice = QComboBox()
        layout.addWidget(self.model_choice)
        self.model_summary = label("", "notice", True)
        layout.addWidget(self.model_summary)
        panes = QSplitter()
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 12, 0)
        image_layout.addWidget(label("Confusion matrix", "section"))
        self.confusion = ImageView()
        image_layout.addWidget(self.confusion, 1)
        open_image = QPushButton("Mở hình kích thước gốc")
        open_image.clicked.connect(lambda: self.open_image(self.confusion))
        image_layout.addWidget(open_image)
        panes.addWidget(image_panel)
        class_panel = QWidget()
        class_layout = QVBoxLayout(class_panel)
        class_layout.setContentsMargins(12, 0, 0, 0)
        class_layout.addWidget(label("Chỉ số theo từng lớp", "section"))
        self.classes = table_widget()
        class_layout.addWidget(self.classes)
        self.class_error = label("", "error", True)
        class_layout.addWidget(self.class_error)
        class_layout.addWidget(label("Thông tin lần huấn luyện", "section"))
        self.metadata = table_widget()
        class_layout.addWidget(self.metadata)
        panes.addWidget(class_panel)
        panes.setSizes([540, 540])
        layout.addWidget(panes, 1)
        self.model_choice.currentIndexChanged.connect(self.update_details)
        self.pages.addWidget(page)

    def build_eda(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.eda_summary = label("", "notice", True)
        layout.addWidget(self.eda_summary)
        panes = QSplitter()
        for kind in ("image", "table"):
            panel = QWidget()
            box = QVBoxLayout(panel)
            box.setContentsMargins(0 if kind == "image" else 12, 0, 12 if kind == "image" else 0, 0)
            box.addWidget(label("Biểu đồ EDA" if kind == "image" else "Thống kê dữ liệu", "section"))
            choice = QComboBox()
            box.addWidget(choice)
            if kind == "image":
                self.eda_image_choice, self.eda_image = choice, ImageView()
                self.eda_image_desc = label("", "notice", True)
                box.addWidget(self.eda_image_desc)
                box.addWidget(self.eda_image, 1)
                button = QPushButton("Mở hình kích thước gốc")
                button.clicked.connect(lambda: self.open_image(self.eda_image))
                box.addWidget(button)
            else:
                self.eda_table_choice, self.eda_table = choice, table_widget()
                self.eda_table_desc = label("", "notice", True)
                box.addWidget(self.eda_table_desc)
                box.addWidget(self.eda_table, 1)
                self.eda_error = label("", "error", True)
                box.addWidget(self.eda_error)
            panes.addWidget(panel)
        panes.setSizes([650, 430])
        layout.addWidget(panes, 1)
        self.eda_image_choice.currentIndexChanged.connect(self.update_eda_image)
        self.eda_table_choice.currentIndexChanged.connect(self.update_eda_table)
        self.pages.addWidget(page)

    def build_features(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.features_summary = label("", "notice", True)
        layout.addWidget(self.features_summary)

        panes = QSplitter()

        # Cột trái: Tab chứa Biểu đồ và Bảng tỷ trọng đặc trưng
        left_panel = QWidget()
        left_box = QVBoxLayout(left_panel)
        left_box.setContentsMargins(0, 0, 8, 0)
        left_box.addWidget(label("Độ quan trọng của đặc trưng (Feature Importance)", "section"))
        
        self.features_tab = QTabWidget()
        
        # Tab 1: Biểu đồ cột ngang
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(4, 4, 4, 4)
        self.features_figure = Figure(figsize=(5.5, 4.8), dpi=100)
        self.features_canvas = FigureCanvasQTAgg(self.features_figure)
        chart_layout.addWidget(self.features_canvas)
        self.features_tab.addTab(chart_widget, "Biểu đồ cột")

        # Tab 2: Bảng số liệu Top 10
        table_widget_panel = QWidget()
        table_layout = QVBoxLayout(table_widget_panel)
        table_layout.setContentsMargins(4, 4, 4, 4)
        self.features_table = table_widget()
        table_layout.addWidget(self.features_table)
        self.features_tab.addTab(table_widget_panel, "Bảng Top 10")

        left_box.addWidget(self.features_tab, 1)
        panes.addWidget(left_panel)

        # Cột phải: Phân tích hành vi chi tiết theo từng lớp (dạng báo cáo ATTT)
        right_panel = QWidget()
        right_box = QVBoxLayout(right_panel)
        right_box.setContentsMargins(8, 0, 0, 0)
        right_box.addWidget(label("Phân tích hành vi chi tiết theo từng lớp (Báo cáo ATTT)", "section"))
        
        self.features_text = QTextBrowser()
        self.features_text.setOpenExternalLinks(True)
        self.features_text.setStyleSheet("""
            QTextBrowser {
                background-color: #ffffff;
                border: 1px solid #d1d8e0;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        right_box.addWidget(self.features_text, 1)
        panes.addWidget(right_panel)

        panes.setSizes([460, 620])
        layout.addWidget(panes, 1)
        self.pages.addWidget(page)

    def refresh(self):
        dataset = self.dataset.currentData()
        self.folder = result_dir(self.root, dataset)
        self.results, issues = load_results(self.root, dataset)
        self.errors.setText("\n".join(issues))
        self.errors.setToolTip("\n".join(issues))
        files = list(self.folder.glob("*_metrics.csv"))
        latest = max((path.stat().st_mtime for path in files), default=None)
        stamp = datetime.fromtimestamp(latest).strftime("%d/%m/%Y %H:%M") if latest else "Chưa có kết quả"
        self.freshness.setText(f"{DATASETS[dataset]}   ·   Cập nhật file: {stamp}")
        count = len(self.results)
        self.notice.setText(f"{count} mô hình đã có kết quả. 5 mô hình Baseline và TabNet (Deep Learning dạng bảng) "
                            "được đánh giá trên cùng điều kiện phân chia 80% train / 20% test.")
        self.update_comparison()
        self.model_choice.blockSignals(True)
        self.model_choice.clear()
        if not self.results.empty:
            for index, row in self.results.iterrows():
                self.model_choice.addItem(MODEL_NAMES.get(row["model"], row["model"]), index)
        self.model_choice.blockSignals(False)
        self.update_details()
        eda = self.root / "results" / ("eda_cicids" if dataset == "cic" else "eda_unsw_nb15")
        for choice, extension in ((self.eda_image_choice, "png"), (self.eda_table_choice, "csv")):
            choice.blockSignals(True)
            choice.clear()
            for path in sorted(eda.glob(f"*.{extension}")):
                choice.addItem(eda_label(path.stem), path)
                choice.setItemData(choice.count() - 1, path.name, Qt.ItemDataRole.ToolTipRole)
            choice.blockSignals(False)
        self.eda_summary.setText(f"{DATASETS[dataset]} · Phân bố nhãn, đặc trưng và chất lượng dữ liệu. "
                                 "Tên đặc trưng và bảng thống kê hiển thị bằng tiếng Việt; "
                                 "EDA dùng dữ liệu gốc, các lần train mặc định lấy mẫu 60.000 dòng.")
        self.update_eda_image()
        self.update_eda_table()
        self.update_features()
        if self.prediction_worker is None:
            self.prediction_model.blockSignals(True)
            self.prediction_model.clear()
            for name in model_paths(self.root, dataset):
                self.prediction_model.addItem(MODEL_NAMES[name], name)
            self.prediction_model.blockSignals(False)
            self.clear_prediction()

    def update_comparison(self):
        frame = self.results.copy()
        if not frame.empty and self.group.currentIndex():
            is_tabnet = frame["model"] == "TabNet"
            frame = frame.loc[~is_tabnet if self.group.currentIndex() == 1 else is_tabnet]
        metric = self.metric_choice.currentData()
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if not frame.empty:
            frame = frame.sort_values(metric, ascending=True, na_position="first")
            plotted = frame.loc[frame[metric].notna()]
            names = [MODEL_NAMES.get(name, name) for name in plotted["model"]]
            bars = ax.barh(names, plotted[metric] * 100, height=.52, color=ACCENT)
            ax.bar_label(bars, labels=[f"{value * 100:.4f}%" for value in plotted[metric]], padding=7, fontsize=10)
            ax.set_xlim(0, 112)
            ax.set_xticks([0, 25, 50, 75, 100])
            ax.xaxis.set_major_formatter(PercentFormatter())
            columns = ["model", *METRICS, "train_time_s", "predict_time_s", "_protocol"]
            short_metrics = {"accuracy": "Accuracy", "precision_weighted": "Precision (w)",
                             "recall_weighted": "Recall (w)", "f1_weighted": "F1 (w)", "f1_macro": "F1 macro"}
            display = frame[columns].sort_values(metric, ascending=False).rename(columns={
                "model": "Mô hình", **short_metrics, "train_time_s": "Train (s)",
                "predict_time_s": "Dự đoán (s)", "_protocol": "Train / test"})
            display["Mô hình"] = display["Mô hình"].replace(MODEL_NAMES)
            fill_table(self.comparison, display, percentages=short_metrics.values())
            self.comparison.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        else:
            ax.text(.5, .5, "Chưa có kết quả cho nhóm mô hình này.", ha="center", va="center", transform=ax.transAxes)
            fill_table(self.comparison, pd.DataFrame())
        ax.set_title(self.metric_choice.currentText(), loc="left", fontsize=12, pad=12)
        ax.tick_params(axis="both", length=0, labelsize=10, colors="#5d5f60")
        ax.xaxis.grid(True, color="#eef0f1")
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        self.canvas.draw_idle()

    def update_details(self):
        index = self.model_choice.currentData()
        if index is None:
            self.model_summary.setText("Chưa có kết quả mô hình. Hãy chạy script huấn luyện rồi làm mới.")
            self.confusion.show_path(None)
            fill_table(self.classes, pd.DataFrame())
            fill_table(self.metadata, pd.DataFrame())
            self.class_error.clear()
            return
        row = self.results.loc[index]
        accuracy = f"{row['accuracy'] * 100:.4f}%" if pd.notna(row["accuracy"]) else "—"
        self.model_summary.setText(f"{MODEL_NAMES.get(row['model'], row['model'])} · Accuracy {accuracy} · {row['_protocol']}")
        self.confusion.show_path(confusion_path(self.folder, row["model"]))
        report, error = per_class(self.folder, row)
        report = report.rename(columns={"class": "Lớp", "support": "Số mẫu"})
        fill_table(self.classes, report, percentages=("precision", "recall", "f1-score"))
        for column in range(1, self.classes.columnCount()):
            self.classes.setColumnWidth(column, max(105, self.classes.columnWidth(column)))
        self.class_error.setText(error)
        meta = row["_meta"]
        values = {"Dataset": DATASETS[self.dataset.currentData()], "Nhóm": row["_group"],
                  "Train / test": row["_protocol"], "Đặc trưng": meta.get("features", "—"),
                  "Nhãn": ", ".join(map(str, meta.get("classes", []))) or "—",
                  "Seed chia dữ liệu": meta.get("split_seed", 42 if meta else "—")}
        if "hidden_layer_sizes" in meta:
            values["Lớp ẩn"] = str(meta["hidden_layer_sizes"])
        for key, name in (("numeric_transform", "Biến đổi dữ liệu số"),
                          ("learning_rate_init", "Learning rate ban đầu"), ("epochs", "Số vòng train")):
            if key in meta:
                values[name] = str(meta[key])
        fill_table(self.metadata, pd.DataFrame({"Thông tin": values.keys(), "Giá trị": map(str, values.values())}))
        self.metadata.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)

    def update_eda_image(self):
        path = self.eda_image_choice.currentData()
        self.eda_image.show_path(path)
        if path:
            desc = EDA_DESCRIPTIONS.get(Path(path).stem, "")
            self.eda_image_desc.setText(desc)
            self.eda_image_desc.setVisible(bool(desc))
        else:
            self.eda_image_desc.clear()
            self.eda_image_desc.setVisible(False)

    def update_eda_table(self):
        path = self.eda_table_choice.currentData()
        frame, error = read_csv(path) if path else (pd.DataFrame(), "Chưa có bảng EDA. Chạy script EDA rồi làm mới.")
        if not frame.empty:
            frame = localize_eda_table(frame, Path(path).stem)
        fill_table(self.eda_table, frame)
        if not frame.empty:
            self.eda_table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.ResizeToContents)
        self.eda_error.setText(error)
        if path and not error:
            desc = EDA_DESCRIPTIONS.get(Path(path).stem, "")
            self.eda_table_desc.setText(desc)
            self.eda_table_desc.setVisible(bool(desc))
        else:
            self.eda_table_desc.clear()
            self.eda_table_desc.setVisible(False)

    def update_features(self):
        dataset = self.dataset.currentData()
        items = FEATURE_IMPORTANCE.get(dataset, [])
        if not items:
            return

        dataset_name = DATASETS.get(dataset, dataset)
        if dataset == "cic":
            summary_text = (
                f"{dataset_name} · Mô hình nhận diện tấn công chủ yếu dựa vào Kích thước gói tin (~24%) "
                "và Tổng dung lượng byte gửi đi (~15%), kết hợp với Thời lượng luồng kết nối và khoảng cách gửi gói (IAT)."
            )
        else:
            summary_text = (
                f"{dataset_name} · Kẻ tấn công bị phát hiện chủ yếu qua Tải thông lượng nguồn sload (8.07%), "
                "Tốc độ gửi gói rate (6.80%), Tổng số byte sbytes (6.27%) và Độ trễ bắt tay TCP synack/tcprtt (~8.9%)."
            )
        self.features_summary.setText(summary_text)

        self.features_figure.clear()
        ax = self.features_figure.add_subplot(111)
        names = [item["feature"] for item in reversed(items)]
        vals = [item["importance"] for item in reversed(items)]

        bars = ax.barh(names, vals, color=ACCENT, height=0.55)
        ax.set_xlabel("Tỷ trọng quyết định (%)", fontsize=10)
        ax.set_title(f"Top 10 đặc trưng quyết định ({dataset_name})", fontsize=11, fontweight="bold", pad=10)
        ax.grid(axis="x", linestyle="--", alpha=0.5)

        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.15, bar.get_y() + bar.get_height() / 2, f"{width:.2f}%",
                    va="center", ha="left", fontsize=9, color="#2c3e50")

        ax.set_xlim(0, max(vals) * 1.3)
        self.features_figure.tight_layout()
        self.features_canvas.draw()

        table_df = pd.DataFrame([
            {
                "Đặc trưng": item["feature"],
                "Tỷ trọng": f"{item['importance']:.2f}%",
                "Vai trò nhận diện tấn công": item["role"],
            }
            for item in items
        ])
        fill_table(self.features_table, table_df)
        self.features_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.features_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.features_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)

        # Cập nhật văn bản phân tích hành vi chi tiết theo lớp
        self.features_text.setHtml(BEHAVIOR_ANALYSIS_HTML.get(dataset, ""))

    @staticmethod
    def open_image(view):
        if view.path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(view.path.resolve())))


def configure_app(app):
    app.setStyle("Fusion")
    family = "Segoe UI"
    if family not in QFontDatabase.families():
        QFontDatabase.addApplicationFont(str(Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"))
        family = "DejaVu Sans"
    app.setFont(QFont(family, 10))
    app.setStyleSheet(STYLE.replace("Segoe UI", family))


def main():
    app = QApplication(sys.argv)
    configure_app(app)
    window = Dashboard()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
