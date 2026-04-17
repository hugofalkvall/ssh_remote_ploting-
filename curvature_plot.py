import io
import sys

import numpy as np
import paramiko
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

pi_ip = "raspberrypi.local"
username = "raspberrypi"
password = "paj"
remote_path = (
    "/home/raspberrypi/Examensarbete/"
    "Posture-estimation-for-motorcycle-riders-using-IMU-based-systems/"
    "curvature_estimates.txt"
)

POLL_INTERVAL_MS = 100
MAX_ROLL_DEG = 180.0
MAX_BEND = 0.6


def parse_latest_sample(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None

    data = np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 3:
        raise ValueError(
            "Expected at least 3 columns in curvature_estimates.txt"
        )

    latest = data[-1].astype(float)
    return {
        "runtime": float(latest[0]),
        "epoch": float(latest[1]),
        "roll": float(latest[2]),
    }


def make_bent_curve(roll_deg, num_points=200):
    start = np.array([-2.0, 0.0], dtype=float)
    end = np.array([2.0, 0.0], dtype=float)
    t = np.linspace(0.0, 1.0, num_points)

    bend = np.clip(roll_deg / MAX_ROLL_DEG, -1.0, 1.0) * MAX_BEND
    control = np.array([0.0, bend], dtype=float)

    one_minus_t = 1.0 - t
    points = (
        (one_minus_t[:, None] ** 2) * start
        + (2.0 * one_minus_t[:, None] * t[:, None]) * control
        + (t[:, None] ** 2) * end
    )
    return start, end, control, points


class CurvaturePlotWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Curvature View")
        self.resize(1100, 760)

        self.ssh = None
        self.sftp = None
        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self.poll_remote_data)

        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel("Preparing viewer...")
        layout.addWidget(self.status_label)

        self.summary_label = QtWidgets.QLabel(
            "Showing the current bent line between two fixed sensor nodes."
        )
        layout.addWidget(self.summary_label)

        self.plot_widget = pg.PlotWidget(title="Current Curve Between Sensors")
        self.plot_widget.setBackground("w")
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setXRange(-0.6, 0.6, padding=0)
        self.plot_widget.setYRange(-0.6, 0.6, padding=0)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.hideButtons()
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        layout.addWidget(self.plot_widget, stretch=1)

        self.curve_item = self.plot_widget.plot(
            [],
            [],
            pen=pg.mkPen("#d9480f", width=4),
        )
        self.node_item = pg.ScatterPlotItem(
            size=18,
            brush=pg.mkBrush("#1f1f1f"),
            pen=pg.mkPen("#1f1f1f"),
        )
        self.control_item = pg.ScatterPlotItem(
            size=10,
            brush=pg.mkBrush("#1971c2"),
            pen=pg.mkPen("#1971c2"),
        )
        self.plot_widget.addItem(self.node_item)
        self.plot_widget.addItem(self.control_item)

        QtCore.QTimer.singleShot(0, self.start)

    def start(self):
        self._connect_remote()
        if self.sftp is not None:
            self.poll_timer.start(POLL_INTERVAL_MS)

    def _connect_remote(self):
        self.status_label.setText(f"Connecting to {pi_ip} as {username}...")

        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                pi_ip,
                username=username,
                password=password,
                timeout=5,
                banner_timeout=5,
                auth_timeout=5,
            )
            sftp = ssh.open_sftp()
        except Exception as exc:
            self.status_label.setText(
                f"Connection to {pi_ip} failed: {exc}. Retrying in 2s."
            )
            self._close_remote()
            QtCore.QTimer.singleShot(2000, self.start)
            return

        self.ssh = ssh
        self.sftp = sftp
        self.status_label.setText("Connected. Waiting for curvature data...")

    def _close_remote(self):
        if self.sftp is not None:
            try:
                self.sftp.close()
            except Exception:
                pass
            self.sftp = None

        if self.ssh is not None:
            try:
                self.ssh.close()
            except Exception:
                pass
            self.ssh = None

    def poll_remote_data(self):
        if self.sftp is None:
            return

        try:
            with self.sftp.open(remote_path, "r") as remote_file:
                text = remote_file.read().decode()
        except Exception as exc:
            self.status_label.setText(f"Read failed: {exc}")
            self.poll_timer.stop()
            self._close_remote()
            QtCore.QTimer.singleShot(2000, self.start)
            return

        try:
            sample = parse_latest_sample(text)
        except Exception as exc:
            self.status_label.setText(f"Parse failed: {exc}")
            return

        if sample is None:
            self.curve_item.setData([], [])
            self.node_item.setData([], [])
            self.control_item.setData([], [])
            self.status_label.setText("Connected. Waiting for curvature data...")
            self.summary_label.setText("Remote file is empty.")
            return

        start, end, control, points = make_bent_curve(sample["roll"])
        self.curve_item.setData(points[:, 0], points[:, 1])
        self.node_item.setData(
            [start[0], end[0]],
            [start[1], end[1]],
        )
        self.control_item.setData([control[0]], [control[1]])

        self.status_label.setText(
            f"Streaming current curve at runtime {sample['runtime']:.2f}s"
        )
        self.summary_label.setText(
            f"Roll: {sample['roll']:+.3f} deg | Epoch: {sample['epoch']:.3f}"
        )

    def closeEvent(self, event):
        self.poll_timer.stop()
        self._close_remote()
        super().closeEvent(event)


def main():
    app = pg.mkQApp("Curvature View")
    window = CurvaturePlotWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
