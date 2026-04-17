import sys

import numpy as np
import paramiko
import pyqtgraph as pg
from PyQt6 import QtCore, QtWidgets

pi_ip = "raspberrypi.local"
username = "raspberrypi"
password = "paj"
remote_path = "/home/raspberrypi/Examensarbete/Posture-estimation-for-motorcycle-riders-using-IMU-based-systems/euler_angles.txt"

OUTPUT_HZ = 50
COLLECTION_DURATION_S = 1000
POLL_INTERVAL_MS = int(1000 / OUTPUT_HZ)
CHANNEL_COLORS = ["r", "g", "b", "c", "m", "y", "w"]


class RemoteYawBuffer:
    def __init__(self):
        self.ssh = None
        self.sftp = None
        self.samples = {}

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
            return self.samples

        # Expected format per row: ch,t_rel,epoch,roll,pitch,yaw
        rows_by_channel = {}
        for line_number, line in enumerate(lines, start=1):
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 6:
                raise ValueError(f"Line {line_number} has {len(parts)} columns, expected 6")

            try:
                channel = int(float(parts[0]))
                t_rel = float(parts[1])
                yaw = float(parts[5])
            except ValueError as exc:
                raise ValueError(f"Failed to parse line {line_number}: {line}") from exc

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
        self.plot_widget.setYRange(-180, 180, padding=0)
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
        self.poll_timer.start(POLL_INTERVAL_MS)
        QtCore.QTimer.singleShot(int(COLLECTION_DURATION_S * 1000), self.finish_collection)

    def collect_samples(self):
        if self.finish_requested:
            return

        try:
            samples = self.buffer.fetch_latest_samples()
        except Exception as exc:
            self.status_label.setText(f"Failed to fetch remote data: {exc}")
            self.poll_timer.stop()
            self.buffer.close()
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

            self.collected_samples[ch]["time"].append(latest_time)
            self.collected_samples[ch]["yaw"].append(latest_yaw)

        channel_summaries = [
            f"CH{ch}: {len(values['time'])}"
            for ch, values in sorted(self.collected_samples.items())
        ]
        if channel_summaries:
            self.status_label.setText(
                "Collecting samples: " + ", ".join(channel_summaries)
            )

    def finish_collection(self):
        if self.finish_requested:
            return
        self.finish_requested = True

        self.poll_timer.stop()
        self.buffer.close()

        self.plot_widget.clear()
        self.plot_widget.addLegend()

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
