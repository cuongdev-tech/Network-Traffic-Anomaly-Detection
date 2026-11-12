"""Run with python giao_dien/main.py or python -m giao_dien.main."""
import sys
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFrame, QGraphicsPixmapItem, QGraphicsScene,
    QGraphicsView, QHBoxLayout, QHeaderView, QLabel, QListWidget, QMainWindow,
    QPushButton, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import get_data_path
from matplotlib.ticker import PercentFormatter

from giao_dien.data import (
    DATASETS, METRICS, MODEL_NAMES, confusion_path, load_results, per_class,
    read_csv, result_dir,
)

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
        self.navigation.addItems(["So sánh mô hình", "Chi tiết mô hình", "EDA dữ liệu"])
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
        outer.addWidget(sheet, 1)
        self.navigation.currentRowChanged.connect(self.change_page)
        self.dataset.currentIndexChanged.connect(self.refresh)
        self.refresh()

    def change_page(self, index):
        self.pages.setCurrentIndex(index)
        self.title.setText(self.navigation.item(index).text())

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
        self.group.addItems(["Tất cả", "Baseline", "RF / XGBoost và MLP"])
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
                box.addWidget(self.eda_image, 1)
                button = QPushButton("Mở hình kích thước gốc")
                button.clicked.connect(lambda: self.open_image(self.eda_image))
                box.addWidget(button)
            else:
                self.eda_table_choice, self.eda_table = choice, table_widget()
                box.addWidget(self.eda_table, 1)
                self.eda_error = label("", "error", True)
                box.addWidget(self.eda_error)
            panes.addWidget(panel)
        panes.setSizes([650, 430])
        layout.addWidget(panes, 1)
        self.eda_image_choice.currentIndexChanged.connect(self.update_eda_image)
        self.eda_table_choice.currentIndexChanged.connect(self.update_eda_table)
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
        self.notice.setText(f"{count} mô hình đã có kết quả. Baseline và RF/XGBoost/MLP dùng tập test khác nhau; "
                            "điểm số giữa hai nhóm chưa phải phép so sánh cùng điều kiện.")
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
                choice.addItem(path.stem.replace("_", " "), path)
            choice.blockSignals(False)
        self.eda_summary.setText(f"{DATASETS[dataset]} · Phân bố nhãn, đặc trưng và chất lượng dữ liệu. "
                                 "EDA dùng dữ liệu gốc; các lần train mặc định lấy mẫu 60.000 dòng.")
        self.update_eda_image()
        self.update_eda_table()

    def update_comparison(self):
        frame = self.results.copy()
        if not frame.empty and self.group.currentIndex():
            mask = frame["_group"] == "Baseline"
            frame = frame.loc[mask if self.group.currentIndex() == 1 else ~mask]
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
        self.eda_image.show_path(self.eda_image_choice.currentData())

    def update_eda_table(self):
        path = self.eda_table_choice.currentData()
        frame, error = read_csv(path) if path else (pd.DataFrame(), "Chưa có bảng EDA. Chạy script EDA rồi làm mới.")
        fill_table(self.eda_table, frame)
        self.eda_error.setText(error)

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
