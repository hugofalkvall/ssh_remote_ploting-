import io
import sys

import numpy as np
import paramiko
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6 import QtCore, QtWidgets
from scipy.spatial.transform import Rotation as R

pi_ip = "raspberrypi.local"
username = "raspberrypi"
password = "paj"
remote_path = "/home/raspberrypi/Examensarbete/Posture-estimation-for-motorcycle-riders-using-IMU-based-systems/euler_angles.txt"

CHANNELS = [0, 2, 3]

# IMU local axes
AXIS_COLORS = [
    (1.0, 0.2, 0.2, 1.0),  # X red
    (0.2, 0.8, 0.2, 1.0),  # Y green
    (0.2, 0.4, 1.0, 1.0),  # Z blue
]
AXIS_LENGTH = 0.6

CHANNEL_POSITIONS = {
    0: np.array([0.0, 0.0, 1.0], dtype=float),
    2: np.array([2.0, 0.0, 0.0], dtype=float),
    3: np.array([2.0, 0.0, 1.0], dtype=float),
}

YAW_LOG_INTERVAL_S = 60.0
TIME_RESET_THRESHOLD_S = 5.0


def axes_from_euler(angles):
    rot = R.from_euler("xyz", angles, degrees=True)
    return rot.apply(np.eye(3))


def wrap_angle_delta(delta):
    return (delta + 180.0) % 360.0 - 180.0


class OrientationWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IMU Orientation Viewer")
        self.resize(1200, 900)

        self.latest_samples = {ch: None for ch in CHANNELS}
        self.axis_items = {ch: [] for ch in CHANNELS}
        self.yaw_baseline = {}
        self.next_yaw_log_t = {}
        self.current_log_cluster = None

        self.ssh = None
        self.sftp = None

        layout = QtWidgets.QVBoxLayout(self)

        self.yaw_log = QtWidgets.QPlainTextEdit()
        self.yaw_log.setReadOnly(True)
        self.yaw_log.setMaximumBlockCount(200)
        self.yaw_log.setFixedHeight(180)
        layout.addWidget(self.yaw_log)

        self.status_label = QtWidgets.QLabel("Preparing viewer...")
        layout.addWidget(self.status_label)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("white")
        self.view.opts["distance"] = 8
        self.view.opts["elevation"] = 18
        self.view.opts["azimuth"] = 35
        layout.addWidget(self.view, stretch=1)

        self._init_scene()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll_remote_data)
        self.timer.start(50)

        # Defer network setup until the event loop starts so the window can open
        # even if the Raspberry Pi is slow to respond or temporarily unavailable.
        QtCore.QTimer.singleShot(0, self._connect_remote)

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
            if self.ssh is not None:
                try:
                    self.ssh.close()
                except Exception:
                    pass
            self.ssh = None
            self.sftp = None
            QtCore.QTimer.singleShot(2000, self._connect_remote)
            return

        self.ssh = ssh
        self.sftp = sftp
        self.status_label.setText("Connected. Waiting for Euler angle data...")

    def _init_scene(self):
        self._add_3d_grids()

        ch23_line = gl.GLLinePlotItem(
            pos=np.vstack([CHANNEL_POSITIONS[2], CHANNEL_POSITIONS[3]]),
            color=(0.15, 0.15, 0.15, 1.0),
            width=3,
            antialias=True,
            mode="lines",
        )
        self.view.addItem(ch23_line)

        for ch, position in CHANNEL_POSITIONS.items():
            marker = gl.GLScatterPlotItem(
                pos=np.array([position]),
                color=(0.1, 0.1, 0.1, 1.0),
                size=10,
                pxMode=True,
            )
            self.view.addItem(marker)

            for color in AXIS_COLORS:
                line = gl.GLLinePlotItem(
                    pos=np.vstack([position, position]),
                    color=color,
                    width=5,
                    antialias=True,
                    mode="lines",
                )
                self.view.addItem(line)
                self.axis_items[ch].append(line)

    def _add_3d_grids(self):
        grid_color = (160, 160, 160, 110)

        # XY plane (floor)
        self.grid_xy = gl.GLGridItem()
        self.grid_xy.setSize(20, 20)
        self.grid_xy.setSpacing(0.5, 0.5)
        self.grid_xy.setColor(grid_color)
        self.grid_xy.translate(1.5, 0.0, 0.0)
        self.view.addItem(self.grid_xy)

        # YZ plane
        self.grid_yz = gl.GLGridItem()
        self.grid_yz.setSize(20, 20)
        self.grid_yz.setSpacing(0.5, 0.5)
        self.grid_yz.setColor(grid_color)
        self.grid_yz.rotate(90, 0, 1, 0)
        self.grid_yz.translate(0.0, 0.0, 1.0)
        self.view.addItem(self.grid_yz)

        # XZ plane
        self.grid_xz = gl.GLGridItem()
        self.grid_xz.setSize(20, 20)
        self.grid_xz.setSpacing(0.5, 0.5)
        self.grid_xz.setColor(grid_color)
        self.grid_xz.rotate(90, 1, 0, 0)
        self.grid_xz.translate(1.5, 0.0, 1.0)
        self.view.addItem(self.grid_xz)

    def poll_remote_data(self):
        if self.sftp is None:
            return

        try:
            with self.sftp.open(remote_path, "r") as remote_file:
                text = remote_file.read().decode()
        except Exception as exc:
            self.status_label.setText(f"Read failed: {exc}")
            try:
                self.sftp.close()
            except Exception:
                pass
            try:
                self.ssh.close()
            except Exception:
                pass
            self.sftp = None
            self.ssh = None
            QtCore.QTimer.singleShot(2000, self._connect_remote)
            return

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            self.status_label.setText("Waiting for Euler angle data...")
            return

        try:
            # Format: ch,t_rel,epoch,roll,pitch,yaw
            data = np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",")
            if data.ndim == 1:
                data = data.reshape(1, -1)
        except Exception as exc:
            self.status_label.setText(f"Parse failed: {exc}")
            return

        has_new_sample = False

        for ch in CHANNELS:
            ch_rows = data[data[:, 0] == ch]
            if ch_rows.size == 0:
                continue

            # Use the last row for this channel, not the maximum t_rel
            latest_row = ch_rows[-1]
            t_rel = float(latest_row[1])
            angles = latest_row[3:6].astype(float)
            current = self.latest_samples[ch]

            # Accept normal forward time, but also allow reset after restart
            should_accept = (
                current is None
                or t_rel > current["t_rel"]
                or (current["t_rel"] - t_rel) > TIME_RESET_THRESHOLD_S
            )

            if should_accept:
                # Reset yaw logging state if time appears to restart
                if current is not None and (current["t_rel"] - t_rel) > TIME_RESET_THRESHOLD_S:
                    self.yaw_baseline.pop(ch, None)
                    self.next_yaw_log_t.pop(ch, None)

                self.latest_samples[ch] = {"t_rel": t_rel, "angles": angles}
                self._update_channel_graphics(ch, angles)
                self._update_yaw_log(ch, t_rel, angles[2])
                has_new_sample = True

        if has_new_sample:
            newest_t = max(
                sample["t_rel"]
                for sample in self.latest_samples.values()
                if sample is not None
            )
            self.status_label.setText(f"Streaming latest samples at {newest_t:.2f}s")

    def _update_channel_graphics(self, ch, angles):
        position = CHANNEL_POSITIONS[ch]
        axes = axes_from_euler(angles)

        for i, line in enumerate(self.axis_items[ch]):
            direction = axes[i] * AXIS_LENGTH
            points = np.vstack([position, position + direction])
            line.setData(
                pos=points,
                color=AXIS_COLORS[i],
                width=5,
                antialias=True,
            )

    def _update_yaw_log(self, ch, t_rel, yaw):
        if ch not in self.yaw_baseline:
            self.yaw_baseline[ch] = yaw
            self.next_yaw_log_t[ch] = t_rel + YAW_LOG_INTERVAL_S
            self._append_yaw_log_line(ch, "start", t_rel, yaw, 0.0, ("start", ch))
            return

        if t_rel >= self.next_yaw_log_t[ch]:
            drift = wrap_angle_delta(yaw - self.yaw_baseline[ch])
            minute_index = int(self.next_yaw_log_t[ch] // YAW_LOG_INTERVAL_S)
            cluster_key = ("minute", minute_index)
            self._append_yaw_log_line(
                ch,
                f"{minute_index}min",
                t_rel,
                yaw,
                drift,
                cluster_key,
            )
            while t_rel >= self.next_yaw_log_t[ch]:
                self.next_yaw_log_t[ch] += YAW_LOG_INTERVAL_S

    def _append_yaw_log_line(self, ch, label, t_rel, yaw, drift, cluster_key):
        if self.current_log_cluster is not None and cluster_key != self.current_log_cluster:
            self.yaw_log.appendPlainText("-----------------------------------------------------------")
        self.current_log_cluster = cluster_key

        line = (
            f"CH{ch} | {label:>8} | yaw={yaw:8.2f} deg | drift={drift:+7.2f} deg"
        )
        self.yaw_log.appendPlainText(line)

    def closeEvent(self, event):
        self.timer.stop()
        try:
            self.sftp.close()
        except Exception:
            pass
        try:
            self.ssh.close()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = pg.mkQApp("IMU Orientation Viewer")
    window = OrientationWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
