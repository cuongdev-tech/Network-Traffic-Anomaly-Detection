
"""Real-time IDS dashboard with a separately privileged capture helper."""
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import sys
from uuid import uuid4

from PySide6.QtCore import Qt, QProcess, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QHeaderView, QHBoxLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from src.cic_feature_extractor import extract_cic_features
from src.unsw_feature_extractor import extract_unsw_features

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("LogisticRegression", "DecisionTree", "KNN", "Random Forest", "XGBoost", "TabR")
STATUS = {
    "NOT_ROUTED": "Không được chọn", "FEATURES_UNAVAILABLE": "Chưa đủ đặc trưng",
    "UNVERIFIED": "Chưa kiểm định tương thích", "INVALID_FLOW": "Flow không hợp lệ",
    "REFERENCE_INVALID": "Đặc trưng tham chiếu không hợp lệ",
    "PROVIDER_ERROR": "Lỗi bộ trích xuất", "REFERENCE_SCHEMA_OK": "Đã khớp schema tham chiếu",
    "VERSION_MISMATCH": "Lệch phiên bản thư viện", "ERROR": "Lỗi suy luận",
    "UNAVAILABLE": "Chưa suy luận", "INCOMPLETE": "Chưa đủ kết quả",
    "DISAGREEMENT": "Các mô hình bất đồng", "CONFLICT": "Hai nhánh bất đồng",
    "ATTACK_CANDIDATE": "Nghi ngờ tấn công", "NORMAL_CANDIDATE": "Ứng viên bình thường",
    "router_not_trained": "Router chưa được huấn luyện",
    "low_routing_score": "Router chưa đủ tin cậy", "router_requested_both": "Router chọn cả hai",
    "router_selected": "Router chọn nhánh",
}


def translate(value):
    return STATUS.get(value, str(value or "—"))


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
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    table.verticalHeader().hide()
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def set_cell(table, row, column, value, tooltip=None):
    item = QTableWidgetItem(str(value if value is not None else "—"))
    item.setToolTip(str(tooltip) if tooltip else item.text())
    table.setItem(row, column, item)


def plain_label(text=""):
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


class RealtimeDashboard(QWidget):
    session_finished = Signal()

    def __init__(self, root=ROOT):
        super().__init__()
        self.root = Path(root).resolve()
        self.path = self.stop_file = None
        self.offset = 0
        self.events = []
        self.flows = self.alerts = self.graph_alerts = self.packets = self.invalid = 0
        self.process = QProcess(self)
        self.process.setWorkingDirectory(str(self.root))
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_process_output)
        self.process.errorOccurred.connect(self.process_error)
        self.process.finished.connect(self.process_finished)
        self.output_tail = ""
        self.stopping = self.close_requested = False
        self.phase = "idle"

        layout = QVBoxLayout(self)
        title = plain_label("REAL-TIME NETWORK INTRUSION DETECTION")
        title.setStyleSheet("font-size: 22px; font-weight: bold; color: #1b6c80;")
        layout.addWidget(title)

        controls = QHBoxLayout()
        controls.addWidget(plain_label("Interface"))
        self.interface = QComboBox()
        self.interface.setEditable(False)
        try:
            names = [name for _, name in socket.if_nameindex() if name != "lo"]
        except OSError:
            names = []
        self.interface.addItems(names)
        if "eth0" in names:
            self.interface.setCurrentText("eth0")
        controls.addWidget(self.interface)
        self.start_button = QPushButton("Start Monitoring")
        self.stop_button = QPushButton("Stop Monitoring")
        self.history_button = QPushButton("Xem lịch sử…")
        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.history_button.clicked.connect(self.choose_log)
        for button in (self.start_button, self.stop_button, self.history_button):
            controls.addWidget(button)
        controls.addStretch()
        layout.addLayout(controls)
        self.status = plain_label("Sẵn sàng. Chọn interface lab và nhấn Start.")
        layout.addWidget(self.status)
        self.metrics = plain_label()
        self.metrics.setStyleSheet("background: #eaf3f5; padding: 10px; border-radius: 6px; font-weight: bold;")
        layout.addWidget(self.metrics)
        layout.addWidget(plain_label("Lưu lượng mạng thời gian thực"))
        self.flow_table = make_table([
            "UTC", "IP nguồn", "IP đích", "Giao thức", "Số gói", "CTU-13",
            "Gaussian", "AI (x/12)", "CIC/UNSW", "Trễ (ms)",
        ])
        self.flow_table.itemSelectionChanged.connect(self.show_selected)
        layout.addWidget(self.flow_table, 2)
        self.detail = plain_label("Chọn flow để xem kết quả.")
        layout.addWidget(self.detail)
        layout.addWidget(plain_label("So sánh 6 thuật toán trên cả hai dataset"))
        self.model_table = make_table(["Model", "CIC-IDS2017", "UNSW-NB15"])
        self.model_table.setRowCount(len(MODELS))
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.model_table, 2)
        layout.addWidget(plain_label(
            "Đồng thuận và confidence không phải ground truth. Không xác định model đúng/sai khi chưa có nhãn thực tế."))
        self.update_metrics()
        self.show_models({})
        self.update_controls()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(500)

    def is_running(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def update_controls(self):
        active = self.is_running()
        self.interface.setEnabled(not active)
        self.start_button.setEnabled(not active and self.interface.count() > 0)
        self.stop_button.setEnabled(active and not self.stopping)
        self.history_button.setEnabled(not active)

    def start_monitoring(self):
        if self.is_running():
            return
        if not sys.platform.startswith("linux"):
            self.status.setText("Giám sát live được thiết kế cho Kali Linux. Windows chỉ dùng xem lịch sử.")
            return
        interface = self.interface.currentText()
        if not interface:
            self.status.setText("Chưa chọn interface.")
            return
        session = self.root / "results" / "stage4" / (
            datetime.now().strftime("live_%Y%m%d_%H%M%S_") + uuid4().hex[:8])
        try:
            session.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            self.status.setText(f"Không tạo được phiên giám sát: {error}")
            return
        self.path = session / "events.jsonl"
        self.stop_file = session / "stop.request"
        self.reset()
        self.output_tail = ""
        self.stopping = False
        self.phase = "starting"
        self.status.setText(f"Đang khởi động {interface}: {session.name}")
        args = ["-u", "-m", "src.stage4_monitor", "--interface", interface,
                "--seconds", "0", "--capture-backend", "dumpcap", "--dual",
                "--output", str(session), "--stop-file", str(self.stop_file)]
        # Only configure this after independently validating the real extractor.
        provider = os.environ.get("ATTT_FEATURE_PROVIDER", "").strip()
        if provider:
            args.extend(["--feature-provider", provider])
        self.process.start(sys.executable, args)
        self.update_controls()

    def stop_monitoring(self):
        if not self.is_running() or self.stopping:
            return
        self.stopping = True
        self.phase = "stopping"
        self.status.setText("Đang dừng bắt gói, hoàn tất flow và lưu summary…")
        try:
            self.stop_file.touch(exist_ok=True)
        except (OSError, AttributeError):
            # SIGTERM is handled by the worker to flush pending flows.
            self.process.terminate()
        self.update_controls()

    def read_process_output(self):
        text = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.output_tail = (self.output_tail + text)[-12000:]
        if self.phase == "starting" and "Listening on " in self.output_tail:
            self.phase = "running"
            self.status.setText(f"Đang giám sát {self.interface.currentText()} | {self.path.parent.name}")
        self.status.setToolTip(self.output_tail)

    def process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.phase = "error"
            self.status.setText(f"Không khởi động được monitor: {self.process.errorString()}")
            self.stopping = False
            self.update_controls()
            self.session_finished.emit()
            if self.close_requested:
                QTimer.singleShot(0, self.close)

    def process_finished(self, exit_code, exit_status):
        self.read_process_output()
        self.poll()
        ok = exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit
        self.phase = "finished" if ok else "error"
        self.stopping = False
        if ok:
            self.status.setText(f"Đã dừng. Log và summary: {self.path.parent.name}")
        else:
            message = self.output_tail.strip().splitlines()
            self.status.setText(f"Monitor lỗi ({exit_code}): {message[-1] if message else 'Xem quyền dumpcap và log.'}")
        self.update_controls()
        self.session_finished.emit()
        if self.close_requested:
            QTimer.singleShot(0, self.close)

    def choose_log(self):
        if self.is_running():
            return
        name, _ = QFileDialog.getOpenFileName(self, "Xem log IDS", str(self.root / "results" / "stage4"), "JSONL (*.jsonl)")
        if name:
            self.path = Path(name)
            self.phase = "history"
            self.reset()
            self.status.setText(f"Đang xem lịch sử: {self.path.parent.name}")
            self.poll()

    def reset(self):
        self.offset = 0
        self.events.clear()
        self.flows = self.alerts = self.graph_alerts = self.packets = self.invalid = 0
        self.flow_table.setRowCount(0)
        self.detail.setText("Chọn flow để xem kết quả.")
        self.show_models({})
        self.update_metrics()

    def update_metrics(self):
        self.metrics.setText(f"Flow: {self.flows}  |  Gói trong flow: {self.packets}  |  "
                             f"Cảnh báo flow: {self.alerts}  |  Cảnh báo đồ thị: {self.graph_alerts}  |  Dòng lỗi: {self.invalid}")

    def poll(self):
        if self.path is None or not self.path.is_file():
            return
        try:
            if self.path.stat().st_size < self.offset:
                self.reset()
            events, self.offset, invalid = read_new_events(self.path, self.offset)
            self.invalid += invalid
            self.flow_table.blockSignals(True)
            try:
                added = False
                for event in events:
                    kind = event.get("event")
                    if kind == "graph_alert":
                        self.graph_alerts += 1
                    elif kind in {"flow_decision", "flow_alert"}:
                        self.add_flow(event)
                        added = True
            finally:
                self.flow_table.blockSignals(False)
            if added:
                self.flow_table.selectRow(self.flow_table.rowCount() - 1)
                self.flow_table.scrollToBottom()
                self.show_selected()
            self.update_metrics()
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.status.setText(f"Lỗi đọc log: {error}")

    def add_flow(self, event):
        self.flows += 1
        count = event.get("packets", 0)
        if type(count) is int and count >= 0:
            self.packets += count
        self.alerts += int(event["event"] == "flow_alert")
        if len(self.events) >= 500:
            self.events.pop(0)
            self.flow_table.removeRow(0)
        self.events.append(event)
        row = self.flow_table.rowCount()
        self.flow_table.insertRow(row)
        dual = event.get("dual_inference") or {}
        decision = dual.get("decision", {})
        values = (event.get("logged_utc"), event.get("src"), event.get("dst"),
                  event.get("protocol"), event.get("packets"), event.get("label"),
                  event.get("gaussian_flag"), decision.get("completed_models", "—"),
                  translate(decision.get("status")), event.get("post_last_packet_ms"))
        for column, value in enumerate(values):
            set_cell(self.flow_table, row, column, value)

    def show_selected(self):
        row = self.flow_table.currentRow()
        if not 0 <= row < len(self.events):
            return
        event = self.events[row]
        dual = event.get("dual_inference") or {}
        audits = dual.get("feature_audit", {})
        cic = audits.get("cic") or extract_cic_features(event)
        unsw = audits.get("unsw") or extract_unsw_features(event)
        decision = dual.get("decision", {})
        candidates = ", ".join(decision.get("candidate_labels", [])) or "—"
        self.detail.setText(
            f"CTU-13: {event.get('label')} | Gaussian: {event.get('gaussian_score')} | "
            f"AI: {decision.get('completed_models', '—')}/12\n"
            f"CIC: {cic['available_count']}/77 ứng viên · {translate(cic['status'])} | "
            f"UNSW: {unsw['available_count']}/34 ứng viên · {translate(unsw['status'])}\n"
            f"Kết quả: {translate(decision.get('status'))} | Nhãn ứng viên: {candidates}")
        self.detail.setToolTip(
            "CIC còn thiếu: " + ", ".join(cic.get("missing_features", [])) + "\nUNSW còn thiếu: "
            + ", ".join(unsw.get("missing_features", [])) + "\n" + str(dual.get("error", "")))
        self.show_models(dual)

    def show_models(self, dual):
        for row, model in enumerate(MODELS):
            set_cell(self.model_table, row, 0, model)
            for column, dataset in enumerate(("cic", "unsw"), 1):
                branch = dual.get(dataset, {})
                result = next((item for item in branch.get("models", []) if item.get("model") == model), None)
                text = translate(branch.get("status", "FEATURES_UNAVAILABLE"))
                tooltip = text
                if result:
                    predictions = result.get("predictions", [])
                    if result.get("status") == "OK" and len(predictions) == 1:
                        pred = predictions[0]
                        latency = result.get("inference_batch_ms")
                        text = f"{pred['label']} ({pred['confidence']:.1%})"
                        if latency is not None:
                            text += f" · {latency:.2f} ms"
                        tooltip = text
                    else:
                        text = translate(result.get("status", "ERROR"))
                        tooltip = result.get("error", text) + "\n" + result.get("version_warning", "")
                set_cell(self.model_table, row, column, text, tooltip)

    def closeEvent(self, event):
        if self.is_running():
            self.close_requested = True
            self.stop_monitoring()
            event.ignore()
        else:
            super().closeEvent(event)


if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = RealtimeDashboard()
    window.resize(1200, 760)
    window.show()
    raise SystemExit(app.exec())
