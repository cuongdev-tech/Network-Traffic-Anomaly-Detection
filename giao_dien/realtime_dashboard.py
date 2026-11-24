
import json
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QAbstractItemView, QFileDialog, QHeaderView,
    QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.cic_feature_extractor import extract_cic_features
from src.unsw_feature_extractor import extract_unsw_features


ROOT = Path(__file__).resolve().parents[1]

MODELS = (
    "LogisticRegression",
    "DecisionTree",
    "KNN",
    "Random Forest",
    "XGBoost",
    "TabR",
)


def read_new_events(path, offset, limit=100):
    events, invalid = [], 0

    with path.open("rb") as file:
        file.seek(offset)

        for _ in range(limit):
            line = file.readline()

            if not line or not line.endswith(b"\n"):
                break

            offset = file.tell()

            try:
                event = json.loads(line)

                if not isinstance(event, dict):
                    raise ValueError("Not a JSON object")

                events.append(event)

            except (ValueError, UnicodeError):
                invalid += 1

    return events, offset, invalid


def make_table(headers):
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(headers)

    table.setEditTriggers(
        QAbstractItemView.EditTrigger.NoEditTriggers
    )

    table.setSelectionBehavior(
        QAbstractItemView.SelectionBehavior.SelectRows
    )

    table.setSelectionMode(
        QAbstractItemView.SelectionMode.SingleSelection
    )

    table.verticalHeader().hide()

    table.horizontalHeader().setStretchLastSection(True)

    return table


def set_cell(table, row, column, value):
    item = QTableWidgetItem(
        str(value if value is not None else "—")
    )

    item.setToolTip(item.text())
    table.setItem(row, column, item)


def plain_label(text=""):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)

    return label


class RealtimeDashboard(QWidget):

    def __init__(self, root=ROOT):
        super().__init__()

        self.root = Path(root)
        self.path = None
        self.offset = 0
        self.events = []

        self.flows = 0
        self.alerts = 0
        self.graph_alerts = 0
        self.packets = 0
        self.invalid = 0

        layout = QVBoxLayout(self)

        title = plain_label(
            "REAL-TIME NETWORK INTRUSION DETECTION"
        )

        title.setStyleSheet(
            "font-size: 22px; font-weight: bold;"
            "color: #1b6c80;"
        )

        layout.addWidget(title)

        controls = QHBoxLayout()

        choose = QPushButton("Chọn events.jsonl")
        choose.clicked.connect(self.choose_log)

        controls.addWidget(choose)

        self.status = plain_label(
            "Chưa chọn phiên giám sát"
        )

        controls.addWidget(self.status, 1)
        layout.addLayout(controls)

        # Monitoring counters.
        self.metrics = plain_label()

        self.metrics.setStyleSheet(
            "background: #eaf3f5;"
            "padding: 12px;"
            "border-radius: 6px;"
            "font-weight: bold;"
        )

        layout.addWidget(self.metrics)

        layout.addWidget(
            plain_label("Lưu lượng mạng thời gian thực")
        )

        self.flow_table = make_table([
            "UTC",
            "IP nguồn",
            "IP đích",
            "Giao thức",
            "Số gói",
            "CTU-13",
            "Gaussian",
            "Trễ (ms)",
        ])

        self.flow_table.itemSelectionChanged.connect(
            self.show_selected
        )

        layout.addWidget(self.flow_table, 2)

        self.detail = plain_label(
            "Chọn một flow để xem chi tiết."
        )

        layout.addWidget(self.detail)

        layout.addWidget(
            plain_label("So sánh 6 thuật toán AI")
        )

        self.model_table = make_table([
            "Model",
            "CIC-IDS2017",
            "UNSW-NB15",
        ])

        self.model_table.setRowCount(len(MODELS))

        self.model_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )

        layout.addWidget(self.model_table, 2)

        layout.addWidget(plain_label(
            "Confidence không phải độ chính xác. "
            "Chưa có nhãn thực tế thì không thể "
            "kết luận mô hình đúng hay sai."
        ))

        self.update_metrics()
        self.show_models({})

        # Read new logs every 500 ms.
        self.timer = QTimer(self)

        self.timer.timeout.connect(self.poll)
        self.timer.start(500)

    def choose_log(self):

        name, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn log IDS",
            str(self.root / "results" / "stage4"),
            "JSONL (*.jsonl)",
        )

        if name:
            self.path = Path(name)
            self.reset()
            self.poll()

    def reset(self):

        self.offset = 0
        self.events.clear()

        self.flows = 0
        self.alerts = 0
        self.graph_alerts = 0
        self.packets = 0
        self.invalid = 0

        self.flow_table.setRowCount(0)

        self.detail.setText(
            "Chọn một flow để xem chi tiết."
        )

        self.show_models({})
        self.update_metrics()

    def update_metrics(self):

        self.metrics.setText(
            f"Flow: {self.flows}  |  "
            f"Gói trong flow: {self.packets}  |  "
            f"Cảnh báo flow: {self.alerts}  |  "
            f"Cảnh báo đồ thị: {self.graph_alerts}  |  "
            f"Dòng lỗi: {self.invalid}"
        )

    def poll(self):

        if self.path is None:
            return

        try:
            if self.path.stat().st_size < self.offset:
                self.reset()

            events, self.offset, invalid = read_new_events(
                self.path,
                self.offset,
            )

            self.invalid += invalid

            self.status.setText(
                f"Đang đọc: "
                f"{self.path.parent.name}/{self.path.name}"
            )

            self.flow_table.blockSignals(True)

            try:
                added = False

                for event in events:

                    kind = event.get("event")

                    if kind == "graph_alert":
                        self.graph_alerts += 1

                    elif kind in (
                        "flow_decision",
                        "flow_alert",
                    ):
                        self.add_flow(event)
                        added = True

            finally:
                self.flow_table.blockSignals(False)

            if added:
                self.flow_table.selectRow(
                    self.flow_table.rowCount() - 1
                )

                self.flow_table.scrollToBottom()
                self.show_selected()

            self.update_metrics()

        except (OSError, ValueError, TypeError) as exc:
            self.status.setText(
                f"Lỗi đọc log: {exc}"
            )

    def add_flow(self, event):

        self.flows += 1

        count = event.get("packets", 0)

        if type(count) is int and count >= 0:
            self.packets += count

        self.alerts += int(
            event["event"] == "flow_alert"
        )

        # Keep only 500 recent flows in memory.
        if len(self.events) >= 500:
            self.events.pop(0)
            self.flow_table.removeRow(0)

        self.events.append(event)

        row = self.flow_table.rowCount()
        self.flow_table.insertRow(row)

        values = (
            event.get("logged_utc"),
            event.get("src"),
            event.get("dst"),
            event.get("protocol"),
            event.get("packets"),
            event.get("label"),
            event.get("gaussian_flag"),
            event.get("post_last_packet_ms"),
        )

        for column, value in enumerate(values):

            set_cell(
                self.flow_table,
                row,
                column,
                value,
            )

    def show_selected(self):

        row = self.flow_table.currentRow()

        if not 0 <= row < len(self.events):
            return

        event = self.events[row]

        cic = extract_cic_features(event)
        unsw = extract_unsw_features(event)

        self.detail.setText(
            f"CTU-13: {event.get('label')} | "
            f"Gaussian: {event.get('gaussian_score')}\n"
            f"Đặc trưng ứng viên CIC: "
            f"{cic['available_count']}/77 | "
            f"UNSW: {unsw['available_count']}/34 | "
            "Chưa xác minh tương thích"
        )

        self.show_models(
            event.get("dual_inference", {})
        )

    def show_models(self, dual):

        for row, model in enumerate(MODELS):

            set_cell(
                self.model_table,
                row,
                0,
                model,
            )

            for column, dataset in enumerate(
                ("cic", "unsw"), 1
            ):

                branch = dual.get(dataset, {})

                result = next(
                    (
                        r for r in branch.get("models", [])
                        if r.get("model") == model
                    ),
                    None,
                )

                text = (
                    "Chưa đủ đặc trưng / chưa suy luận"
                )

                if result:

                    if (
                        result.get("status") == "OK"
                        and len(
                            result.get("predictions", [])
                        ) == 1
                    ):

                        pred = result["predictions"][0]

                        text = (
                            f"{pred['label']} "
                            f"({pred['confidence']:.1%})"
                        )

                    else:
                        text = result.get(
                            "error",
                            result.get("status", "ERROR"),
                        )

                set_cell(
                    self.model_table,
                    row,
                    column,
                    text,
                )


if __name__ == "__main__":

    from PySide6.QtWidgets import QApplication

    app = QApplication([])

    window = RealtimeDashboard()
    window.setWindowTitle("Real-time IDS")
    window.resize(1100, 720)
    window.show()

    raise SystemExit(app.exec())
