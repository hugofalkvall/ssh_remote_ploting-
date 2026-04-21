import sys
import time

import numpy as np
import paramiko
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets
from scipy.spatial.transform import Rotation as R

pi_ip = "raspberrypi.local"
username = "raspberrypi"
password = "paj"
remote_path = "/home/raspberrypi/Examensarbete/Posture-estimation-for-motorcycle-riders-using-IMU-based-systems/quaternions.txt"

OUTPUT_HZ = 50
COLLECTION_DURATION_S = 900
POLL_INTERVAL_MS = int(1000 / OUTPUT_HZ)
CHANNEL_COLORS = ["r", "g", "b", "c", "m", "y", "w"]
PLOT_CHANNELS = {0}


def rotation_from_quaternion(quaternion):
    # Input rows are qw,qx,qy,qz, while scipy expects qx,qy,qz,qw.
    quat = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Quaternion has zero length")

    qw, qx, qy, qz = quat / norm
    return R.from_quat([qx, qy, qz, qw])


def yaw_from_quaternion(quaternion):
    rotation = rotation_from_quaternion(quaternion)
    return float(rotation.as_euler("xyz", degrees=True)[2])


class RemoteYawBuffer:
    def __init__(self):
        self.ssh = None
        self.sftp = None
        self.samples = {}
        self.skipped_rows = 0

    def connect(self):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh.connect(
            pi_ip,
            username=username,
            password=password,
            timeout=5,
            banner_timeout=5,
            auth_timeout=5,
        )
        self.sftp = self.ssh.open_sftp()

    def fetch_latest_samples(self):
        if self.sftp is None:
            raise RuntimeError("SSH connection is not open")

        with self.sftp.open(remote_path, "r") as remote_file:
            text = remote_file.read().decode()

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            self.samples = {}
            self.skipped_rows = 0
            return self.samples

        # Expected format per row: ch,t_rel,epoch,qw,qx,qy,qz
        rows_by_channel = {}
        skipped_rows = 0
        for line_number, line in enumerate(lines, start=1):
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 7:
                skipped_rows += 1
                continue

            try:
                channel = int(float(parts[0]))
                if channel not in PLOT_CHANNELS:
                    continue

                t_rel = float(parts[1])
                quaternion = [float(part) for part in parts[3:7]]
                yaw = yaw_from_quaternion(quaternion)
            except ValueError:
                skipped_rows += 1
                continue

            if channel not in rows_by_channel:
                rows_by_channel[channel] = {"time": [], "yaw": []}

            rows_by_channel[channel]["time"].append(t_rel)
            rows_by_channel[channel]["yaw"].append(yaw)

        self.samples = {}
        for channel, values in rows_by_channel.items():
            time_values = np.array(values["time"], dtype=float)
            yaw_values = np.array(values["yaw"], dtype=float)
            sort_order = np.argsort(time_values)
            self.samples[channel] = {
                "time": time_values[sort_order],
                "yaw": yaw_values[sort_order],
            }

        self.skipped_rows = skipped_rows
        return self.samples

    def close(self):
        if self.sftp is not None:
            self.sftp.close()
            self.sftp = None

        if self.ssh is not None:
            self.ssh.close()
            self.ssh = None


class YawPlotWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.buffer = RemoteYawBuffer()
        self.collected_samples = {}
        self.last_seen_time = {}
        self.failed_fetches = 0
        self.collection_start_time = None
        self.finish_requested = False
        self.poll_timer = QtCore.QTimer(self)
        self.poll_timer.timeout.connect(self.collect_samples)

        self.setWindowTitle("Yaw Plot")
        self.resize(1100, 700)

        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel(
            "Connecting..."
        )
        layout.addWidget(self.status_label)

        self.plot_widget = pg.PlotWidget(title="Yaw Over Time")
        self.plot_widget.setLabel("left", "Yaw", units="deg")
        self.plot_widget.setLabel("bottom", "Time", units="s")
        self.plot_widget.setYRange(-5, 5, padding=0)
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        layout.addWidget(self.plot_widget)

        QtCore.QTimer.singleShot(0, self.start)

    def start(self):
        try:
            self.buffer.connect()
        except Exception as exc:
            self.status_label.setText(f"SSH connection failed: {exc}")
            return

        self.status_label.setText(
            f"Connected. Collecting at {OUTPUT_HZ} Hz for {COLLECTION_DURATION_S:.0f} seconds..."
        )
        self.collection_start_time = time.monotonic()
        self.poll_timer.start(POLL_INTERVAL_MS)
        QtCore.QTimer.singleShot(int(COLLECTION_DURATION_S * 1000), self.finish_collection)

    def collect_samples(self):
        if self.finish_requested:
            return
        if self.collection_start_time is None:
            return

        elapsed_time = time.monotonic() - self.collection_start_time
        if elapsed_time > COLLECTION_DURATION_S:
            self.finish_collection()
            return

        try:
            samples = self.buffer.fetch_latest_samples()
        except Exception as exc:
            self.failed_fetches += 1
            self.status_label.setText(
                f"Skipped failed fetch #{self.failed_fetches}: {exc}"
            )
            return

        for index, ch in enumerate(sorted(samples)):
            time_values = samples[ch]["time"]
            yaw_values = samples[ch]["yaw"]
            if len(time_values) == 0:
                continue

            latest_time = float(time_values[-1])
            latest_yaw = float(yaw_values[-1])
            if self.last_seen_time.get(ch) == latest_time:
                continue

            self.last_seen_time[ch] = latest_time
            if ch not in self.collected_samples:
                self.collected_samples[ch] = {"time": [], "yaw": []}

            self.collected_samples[ch]["time"].append(elapsed_time)
            self.collected_samples[ch]["yaw"].append(latest_yaw)

        channel_summaries = [
            f"CH{ch}: {len(values['time'])}"
            for ch, values in sorted(self.collected_samples.items())
        ]
        if channel_summaries:
            status = (
                f"Collecting {elapsed_time:.1f}/{COLLECTION_DURATION_S:.0f}s: "
                + ", ".join(channel_summaries)
            )
            skipped = self.buffer.skipped_rows
            if skipped:
                status += f" | skipped invalid rows: {skipped}"
            if self.failed_fetches:
                status += f" | skipped failed fetches: {self.failed_fetches}"
            self.status_label.setText(status)

    def finish_collection(self):
        if self.finish_requested:
            return
        self.finish_requested = True

        self.poll_timer.stop()
        self.buffer.close()

        self.plot_widget.clear()
        self.plot_widget.addLegend()
        self.plot_widget.setXRange(0, COLLECTION_DURATION_S, padding=0)

        plotted_channels = 0
        channel_summaries = []
        for index, ch in enumerate(sorted(self.collected_samples)):
            time_values = np.array(self.collected_samples[ch]["time"], dtype=float)
            yaw_values = np.array(self.collected_samples[ch]["yaw"], dtype=float)
            if len(time_values) == 0:
                continue

            sort_order = np.argsort(time_values)
            time_values = time_values[sort_order]
            yaw_values = yaw_values[sort_order]
            in_window = time_values <= COLLECTION_DURATION_S
            time_values = time_values[in_window]
            yaw_values = yaw_values[in_window]
            if len(time_values) == 0:
                continue

            self.plot_widget.plot(
                time_values,
                yaw_values,
                pen=pg.mkPen(CHANNEL_COLORS[index % len(CHANNEL_COLORS)], width=2),
                symbol="o",
                symbolSize=4,
                symbolBrush=CHANNEL_COLORS[index % len(CHANNEL_COLORS)],
                symbolPen=CHANNEL_COLORS[index % len(CHANNEL_COLORS)],
                name=f"CH{ch}",
            )
            plotted_channels += 1
            channel_summaries.append(f"CH{ch}: {len(time_values)}")

        if plotted_channels == 0:
            self.status_label.setText("No samples were collected during the capture window.")
        else:
            joined = ", ".join(channel_summaries)
            self.status_label.setText(f"Plotted {plotted_channels} channel(s): {joined}")

    def closeEvent(self, event):
        self.poll_timer.stop()
        self.buffer.close()
        super().closeEvent(event)


def main():
    app = pg.mkQApp("Yaw Plot")
    window = YawPlotWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
