import io
import sys

import numpy as np
import paramiko
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6 import QtCore, QtWidgets
from scipy.spatial.transform import Rotation as R

ANGLE_LIMIT_DEG = 180
POLL_INTERVAL_MS = 50
AXIS_LENGTH = 0.9
pi_ip = "raspberrypi.local"
username = "raspberrypi"
password = "paj"
remote_path = "/home/raspberrypi/Examensarbete/Posture-estimation-for-motorcycle-riders-using-IMU-based-systems/euler_angles.txt"
CHANNELS = [0, 2, 3]
TIME_RESET_THRESHOLD_S = 5.0
AXIS_COLORS = {
    "roll": "#d94f4f",
    "pitch": "#2f9e44",
    "yaw": "#0e79d7",
}
AXIS_INDEX = {"roll": 0, "pitch": 1, "yaw": 2}
ANGLE_DEFINITIONS = [
    ("roll", "Roll", "x"),
    ("pitch", "Pitch", "y"),
    ("yaw", "Yaw", "z"),
]


def axes_from_euler(angles_deg):
    rotation = R.from_euler("xyz", angles_deg, degrees=True)
    return rotation.apply(np.eye(3))


class AngleDialPlot(pg.PlotWidget):
    def __init__(self, angle_key, angle_label, color):
        super().__init__(title=f"{angle_label} ({angle_key.upper()})")
        self.angle_key = angle_key
        self.angle_label = angle_label
        self.color = color

        self.setBackground("w")
        self.setAspectLocked(True)
        self.hideButtons()
        self.setMenuEnabled(False)
        self.setMouseEnabled(x=False, y=False)
        self.showGrid(x=True, y=True, alpha=0.2)
        self.setXRange(-1.15, 1.15, padding=0)
        self.setYRange(-1.15, 1.15, padding=0)
        self.setMinimumSize(260, 220)

        circle_angles = np.linspace(0.0, 2.0 * np.pi, 361)
        self.plot(
            np.cos(circle_angles),
            np.sin(circle_angles),
            pen=pg.mkPen((160, 160, 160), width=1),
        )
        self.plot([-1.05, 1.05], [0.0, 0.0], pen=pg.mkPen((210, 210, 210), width=1))
        self.plot([0.0, 0.0], [-1.05, 1.05], pen=pg.mkPen((210, 210, 210), width=1))

        self.vector = pg.PlotCurveItem(
            [0.0, 1.0],
            [0.0, 0.0],
            pen=pg.mkPen(color, width=4),
        )
        self.addItem(self.vector)

        self.tip = pg.ScatterPlotItem(
            [1.0],
            [0.0],
            size=12,
            brush=pg.mkBrush(color),
            pen=pg.mkPen(color),
        )
        self.addItem(self.tip)

        self.value_text = pg.TextItem(color=color, anchor=(0.5, 0.0))
        self.value_text.setPos(0.0, -1.12)
        self.addItem(self.value_text)

        self.update_angle(0)

    def update_angle(self, angle_deg):
        radians = np.deg2rad(angle_deg)
        x = float(np.cos(radians))
        y = float(np.sin(radians))
        self.vector.setData([0.0, x], [0.0, y])
        self.tip.setData([x], [y])
        self.value_text.setText(f"{self.angle_label}: {angle_deg:+d} deg")


class EulerAngleWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Euler Angle Visualizer")
        self.resize(1250, 900)
        self.setMinimumSize(1050, 780)

        self.current_angles = {"roll": 0, "pitch": 0, "yaw": 0}
        self.latest_samples = {ch: None for ch in CHANNELS}
        self.current_channel = CHANNELS[0]
        self.axis_items = []
        self.slider_widgets = {}
        self.value_labels = {}
        self.angle_plots = {}
        self.ssh = None
        self.sftp = None

        root_layout = QtWidgets.QHBoxLayout(self)

        controls_panel = QtWidgets.QWidget()
        controls_panel.setMinimumWidth(340)
        controls_panel.setMaximumWidth(380)
        controls_layout = QtWidgets.QVBoxLayout(controls_panel)
        controls_layout.setSpacing(14)
        controls_layout.addWidget(QtWidgets.QLabel("Euler Angles"))

        self.status_label = QtWidgets.QLabel("Preparing viewer...")
        self.status_label.setFixedHeight(44)
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)

        channel_row = QtWidgets.QHBoxLayout()
        channel_row.addWidget(QtWidgets.QLabel("Channel"))
        self.channel_selector = QtWidgets.QComboBox()
        self.channel_selector.setFixedWidth(120)
        for ch in CHANNELS:
            self.channel_selector.addItem(f"CH{ch}", ch)
        self.channel_selector.currentIndexChanged.connect(self._on_channel_changed)
        channel_row.addWidget(self.channel_selector)
        controls_layout.addLayout(channel_row)

        for angle_key, angle_label, _ in ANGLE_DEFINITIONS:
            row = self._build_slider_row(angle_key, angle_label)
            controls_layout.addWidget(row)

        self.sample_label = QtWidgets.QLabel("Waiting for streamed data...")
        self.sample_label.setFixedHeight(24)
        controls_layout.addWidget(self.sample_label)
        controls_layout.addStretch(1)

        root_layout.addWidget(controls_panel, stretch=0)

        plot_panel = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_panel)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("white")
        self.view.opts["distance"] = 5.5
        self.view.opts["elevation"] = 20
        self.view.opts["azimuth"] = 40
        self.view.setMinimumHeight(520)
        plot_layout.addWidget(self.view, stretch=3)

        angle_plot_layout = QtWidgets.QHBoxLayout()
        for angle_key, angle_label, _ in ANGLE_DEFINITIONS:
            dial = AngleDialPlot(angle_key, angle_label, AXIS_COLORS[angle_key])
            self.angle_plots[angle_key] = dial
            angle_plot_layout.addWidget(dial, stretch=1)
        plot_layout.addLayout(angle_plot_layout, stretch=2)

        root_layout.addWidget(plot_panel, stretch=1)

        self._init_3d_scene()
        self._sync_visuals()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll_remote_data)
        self.timer.start(POLL_INTERVAL_MS)
        QtCore.QTimer.singleShot(0, self._connect_remote)

    def _build_slider_row(self, angle_key, angle_label):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        header_layout = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel(angle_label)
        value_label = QtWidgets.QLabel("0 deg")
        value_label.setFixedWidth(90)
        value_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        header_layout.addWidget(title)
        header_layout.addWidget(value_label)
        layout.addLayout(header_layout)

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(-ANGLE_LIMIT_DEG, ANGLE_LIMIT_DEG)
        slider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(30)
        slider.setEnabled(False)

        layout.addWidget(slider)

        limit_layout = QtWidgets.QHBoxLayout()
        limit_layout.addWidget(QtWidgets.QLabel("-180 deg"))
        limit_layout.addStretch(1)
        limit_layout.addWidget(QtWidgets.QLabel("+180 deg"))
        layout.addLayout(limit_layout)

        self.slider_widgets[angle_key] = slider
        self.value_labels[angle_key] = value_label
        widget.setFixedHeight(115)
        return widget

    def _init_3d_scene(self):
        grid = gl.GLGridItem()
        grid.setSize(4, 4)
        grid.setSpacing(0.5, 0.5)
        grid.setColor((170, 170, 170, 140))
        self.view.addItem(grid)

        # World axes stay fixed so the combined orientation is easy to compare.
        world_colors = [
            (0.5, 0.5, 0.5, 0.7),
            (0.5, 0.5, 0.5, 0.7),
            (0.5, 0.5, 0.5, 0.7),
        ]
        for axis_index, color in enumerate(world_colors):
            direction = np.zeros(3)
            direction[axis_index] = AXIS_LENGTH
            line = gl.GLLinePlotItem(
                pos=np.vstack([np.zeros(3), direction]),
                color=color,
                width=2,
                antialias=True,
                mode="lines",
            )
            self.view.addItem(line)

        for angle_key, _, _ in ANGLE_DEFINITIONS:
            color = pg.mkColor(AXIS_COLORS[angle_key])
            rgba = (
                color.redF(),
                color.greenF(),
                color.blueF(),
                1.0,
            )
            line = gl.GLLinePlotItem(
                pos=np.zeros((2, 3)),
                color=rgba,
                width=5,
                antialias=True,
                mode="lines",
            )
            self.view.addItem(line)
            self.axis_items.append((angle_key, line))

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
            QtCore.QTimer.singleShot(2000, self._connect_remote)
            return

        self.ssh = ssh
        self.sftp = sftp
        self.status_label.setText(
            f"Connected. Waiting for Euler angle data for CH{self.current_channel}..."
        )

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

    def _on_channel_changed(self):
        self.current_channel = self.channel_selector.currentData()
        latest = self.latest_samples.get(self.current_channel)
        if latest is None:
            self.sample_label.setText(f"CH{self.current_channel}: waiting for samples...")
            self.status_label.setText(
                f"Connected. Waiting for Euler angle data for CH{self.current_channel}..."
            )
            return

        self._apply_sample(self.current_channel, latest["t_rel"], latest["angles"])

    def poll_remote_data(self):
        if self.sftp is None:
            return

        try:
            with self.sftp.open(remote_path, "r") as remote_file:
                text = remote_file.read().decode()
        except Exception as exc:
            self.status_label.setText(f"Read failed: {exc}")
            self._close_remote()
            QtCore.QTimer.singleShot(2000, self._connect_remote)
            return

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            self.status_label.setText("Waiting for Euler angle data...")
            self.sample_label.setText("Remote file is empty.")
            return

        try:
            data = np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",")
            if data.ndim == 1:
                data = data.reshape(1, -1)
        except Exception as exc:
            self.status_label.setText(f"Parse failed: {exc}")
            return

        current_channel_updated = False

        for ch in CHANNELS:
            ch_rows = data[data[:, 0] == ch]
            if ch_rows.size == 0:
                continue

            latest_row = ch_rows[-1]
            t_rel = float(latest_row[1])
            angles = latest_row[3:6].astype(float)
            current = self.latest_samples[ch]

            should_accept = (
                current is None
                or t_rel > current["t_rel"]
                or (current["t_rel"] - t_rel) > TIME_RESET_THRESHOLD_S
            )
            if not should_accept:
                continue

            self.latest_samples[ch] = {"t_rel": t_rel, "angles": angles}
            if ch == self.current_channel:
                self._apply_sample(ch, t_rel, angles)
                current_channel_updated = True

        if not current_channel_updated and self.latest_samples[self.current_channel] is None:
            self.sample_label.setText(f"CH{self.current_channel}: waiting for samples...")

    def _apply_sample(self, channel, t_rel, angles):
        for (angle_key, _, _), value in zip(ANGLE_DEFINITIONS, angles):
            self.current_angles[angle_key] = int(round(float(value)))
        self._sync_visuals()
        self.status_label.setText(
            f"Streaming CH{channel} at {t_rel:.2f}s from {pi_ip}"
        )
        self.sample_label.setText(
            f"CH{channel} sample: t_rel={t_rel:.2f}s"
        )

    def _sync_visuals(self):
        ordered_angles = np.array(
            [self.current_angles[key] for key, _, _ in ANGLE_DEFINITIONS],
            dtype=float,
        )
        rotated_axes = axes_from_euler(ordered_angles)

        for angle_key, _, _ in ANGLE_DEFINITIONS:
            value = self.current_angles[angle_key]
            self.value_labels[angle_key].setText(f"{value:+d} deg")
            self.angle_plots[angle_key].update_angle(value)

        for angle_key, line in self.axis_items:
            axis_index = AXIS_INDEX[angle_key]
            direction = rotated_axes[axis_index] * AXIS_LENGTH
            line.setData(pos=np.vstack([np.zeros(3), direction]))

        roll, pitch, yaw = (self.current_angles[key] for key, _, _ in ANGLE_DEFINITIONS)
        for angle_key, _, _ in ANGLE_DEFINITIONS:
            self.slider_widgets[angle_key].blockSignals(True)
            self.slider_widgets[angle_key].setValue(self.current_angles[angle_key])
            self.slider_widgets[angle_key].blockSignals(False)

        self.sample_label.setToolTip(
            f"Roll {roll:+d} deg, Pitch {pitch:+d} deg, Yaw {yaw:+d} deg"
        )

    def closeEvent(self, event):
        self.timer.stop()
        self._close_remote()
        super().closeEvent(event)


def main():
    app = pg.mkQApp("Euler Angle Visualizer")
    window = EulerAngleWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
