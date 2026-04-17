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
remote_path = "/home/raspberrypi/Examensarbete/Posture-estimation-for-motorcycle-riders-using-IMU-based-systems/quaternions.txt"

BIKE_CHANNEL = 0
LOWER_SPINE_CHANNEL = 2
UPPER_SPINE_CHANNEL = 3
CHANNELS = [BIKE_CHANNEL, LOWER_SPINE_CHANNEL, UPPER_SPINE_CHANNEL]

AXIS_COLORS = [
    (1.0, 0.2, 0.2, 1.0),  # X red
    (0.2, 0.8, 0.2, 1.0),  # Y green
    (0.2, 0.4, 1.0, 1.0),  # Z blue
]
AXIS_LENGTH = 0.35
YAW_LOG_INTERVAL_S = 60.0
TIME_RESET_THRESHOLD_S = 5.0

# Viewer coordinates are bike-relative: X forward, Y rider left, Z up.
BIKE_ORIGIN = np.array([-1.4, 0.0, 0.05], dtype=float)
LOWER_SPINE_POINT = np.array([0.0, 0.0, 0.15], dtype=float)
UPPER_SPINE_POINT = np.array([0.0, 0.0, 1.45], dtype=float)
CHANNEL_POSITIONS = {
    BIKE_CHANNEL: BIKE_ORIGIN,
    LOWER_SPINE_CHANNEL: LOWER_SPINE_POINT,
    UPPER_SPINE_CHANNEL: UPPER_SPINE_POINT,
}

SPINE_SEGMENTS = 13
SPINE_CURVE_POINTS = 120
VERTEBRA_SIZE = np.array([0.16, 0.28, 0.055], dtype=float)
DISC_COLOR = (0.78, 0.78, 0.72, 1.0)
CURVE_COLOR = (0.05, 0.05, 0.05, 1.0)
BODY_PANEL_COLOR = (0.25, 0.55, 0.95, 0.18)


def rotation_from_quaternion(quaternion):
    # Input rows are qw,qx,qy,qz, while scipy expects qx,qy,qz,qw.
    quat = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Quaternion has zero length")

    qw, qx, qy, qz = quat / norm
    return R.from_quat([qx, qy, qz, qw])


def wrap_angle_delta(delta):
    return (delta + 180.0) % 360.0 - 180.0


def normalize(vector, fallback):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return np.asarray(fallback, dtype=float)
    return vector / norm


def cubic_bezier(p0, p1, p2, p3, t_values):
    t = np.asarray(t_values, dtype=float)[:, None]
    one_minus = 1.0 - t
    return (
        (one_minus**3) * p0
        + (3.0 * one_minus**2 * t) * p1
        + (3.0 * one_minus * t**2) * p2
        + (t**3) * p3
    )


def cubic_bezier_tangent(p0, p1, p2, p3, t):
    one_minus = 1.0 - t
    return (
        3.0 * one_minus**2 * (p1 - p0)
        + 6.0 * one_minus * t * (p2 - p1)
        + 3.0 * t**2 * (p3 - p2)
    )


def make_oriented_ellipsoid_mesh(base_vertices, base_faces, center, tangent, size):
    z_axis = normalize(tangent, [0.0, 0.0, 1.0])
    x_axis = normalize(np.cross([0.0, 1.0, 0.0], z_axis), [1.0, 0.0, 0.0])
    y_axis = normalize(np.cross(z_axis, x_axis), [0.0, 1.0, 0.0])
    basis = np.column_stack([x_axis, y_axis, z_axis])
    vertices = center + (base_vertices * size) @ basis.T
    return gl.MeshData(vertexes=vertices, faces=base_faces)


def make_panel_mesh(left_points, right_points):
    vertices = np.vstack([left_points, right_points])
    n = len(left_points)
    faces = []
    for i in range(n - 1):
        faces.append([i, i + 1, n + i + 1])
        faces.append([i, n + i + 1, n + i])
    return gl.MeshData(vertexes=vertices, faces=np.array(faces, dtype=np.uint32))


class SpineSlicerWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bike-Relative Spine Viewer")
        self.resize(1200, 900)

        self.latest_samples = {ch: None for ch in CHANNELS}
        self.axis_items = {ch: [] for ch in CHANNELS}
        self.yaw_baseline = {}
        self.next_yaw_log_t = {}
        self.current_log_cluster = None

        self.ssh = None
        self.sftp = None

        sphere = gl.MeshData.sphere(rows=10, cols=18)
        self.base_vertices = sphere.vertexes()
        self.base_faces = sphere.faces()

        layout = QtWidgets.QVBoxLayout(self)

        self.yaw_log = QtWidgets.QPlainTextEdit()
        self.yaw_log.setReadOnly(True)
        self.yaw_log.setMaximumBlockCount(200)
        self.yaw_log.setFixedHeight(160)
        layout.addWidget(self.yaw_log)

        self.status_label = QtWidgets.QLabel("Preparing viewer...")
        layout.addWidget(self.status_label)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor("white")
        self.view.opts["distance"] = 5.5
        self.view.opts["elevation"] = 18
        self.view.opts["azimuth"] = 38
        layout.addWidget(self.view, stretch=1)

        self._init_scene()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll_remote_data)
        self.timer.start(50)

        QtCore.QTimer.singleShot(0, self._connect_remote)

    def _init_scene(self):
        self._add_slicer_style_reference_planes()
        self._add_static_labels()

        self.spine_curve = gl.GLLinePlotItem(
            pos=np.zeros((SPINE_CURVE_POINTS, 3), dtype=float),
            color=CURVE_COLOR,
            width=5,
            antialias=True,
            mode="line_strip",
        )
        self.view.addItem(self.spine_curve)

        self.body_panel = gl.GLMeshItem(
            meshdata=make_panel_mesh(
                np.zeros((2, 3), dtype=float),
                np.zeros((2, 3), dtype=float),
            ),
            color=BODY_PANEL_COLOR,
            drawEdges=False,
            drawFaces=True,
            smooth=False,
            computeNormals=False,
        )
        self.body_panel.setGLOptions("translucent")
        self.view.addItem(self.body_panel)

        self.vertebra_items = []
        for _ in range(SPINE_SEGMENTS):
            item = gl.GLMeshItem(
                meshdata=make_oriented_ellipsoid_mesh(
                    self.base_vertices,
                    self.base_faces,
                    LOWER_SPINE_POINT,
                    [0.0, 0.0, 1.0],
                    VERTEBRA_SIZE,
                ),
                color=DISC_COLOR,
                drawEdges=True,
                edgeColor=(0.35, 0.35, 0.35, 0.45),
                drawFaces=True,
                smooth=True,
            )
            item.setGLOptions("opaque")
            self.view.addItem(item)
            self.vertebra_items.append(item)

        for ch, position in CHANNEL_POSITIONS.items():
            marker = gl.GLScatterPlotItem(
                pos=np.array([position]),
                color=(0.06, 0.06, 0.06, 1.0),
                size=12,
                pxMode=True,
            )
            self.view.addItem(marker)

            for color in AXIS_COLORS:
                line = gl.GLLinePlotItem(
                    pos=np.vstack([position, position]),
                    color=color,
                    width=4,
                    antialias=True,
                    mode="lines",
                )
                self.view.addItem(line)
                self.axis_items[ch].append(line)

        self._refresh_visualization()

    def _add_slicer_style_reference_planes(self):
        grid_color = (150, 150, 150, 95)

        floor = gl.GLGridItem()
        floor.setSize(4, 4)
        floor.setSpacing(0.25, 0.25)
        floor.setColor(grid_color)
        floor.translate(0.0, 0.0, 0.0)
        self.view.addItem(floor)

        sagittal = gl.GLGridItem()
        sagittal.setSize(3.0, 2.0)
        sagittal.setSpacing(0.25, 0.25)
        sagittal.setColor((45, 100, 200, 105))
        sagittal.rotate(90, 1, 0, 0)
        sagittal.translate(0.0, 0.0, 0.85)
        self.view.addItem(sagittal)

        coronal = gl.GLGridItem()
        coronal.setSize(3.0, 2.0)
        coronal.setSpacing(0.25, 0.25)
        coronal.setColor((200, 70, 70, 95))
        coronal.rotate(90, 0, 1, 0)
        coronal.translate(0.0, 0.0, 0.85)
        self.view.addItem(coronal)

    def _add_static_labels(self):
        labels = [
            ("CH0 bike", BIKE_ORIGIN + np.array([0.06, 0.04, 0.08])),
            ("CH2 lower spine", LOWER_SPINE_POINT + np.array([0.06, 0.04, 0.03])),
            ("CH3 upper spine", UPPER_SPINE_POINT + np.array([0.06, 0.04, 0.03])),
        ]
        for text, position in labels:
            self.view.addItem(
                gl.GLTextItem(
                    pos=position,
                    text=text,
                    color=(30, 30, 30, 255),
                )
            )

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
        self.status_label.setText("Connected. Waiting for quaternion data...")

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
            self._close_remote()
            QtCore.QTimer.singleShot(2000, self._connect_remote)
            return

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            self.status_label.setText("Waiting for quaternion data...")
            return

        try:
            data = np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",")
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] < 7:
                raise ValueError(
                    f"Expected ch,t_rel,epoch,qw,qx,qy,qz; got {data.shape[1]} columns"
                )
        except Exception as exc:
            self.status_label.setText(f"Parse failed: {exc}")
            return

        accepted_channels = []
        for ch in CHANNELS:
            ch_rows = data[data[:, 0] == ch]
            if ch_rows.size == 0:
                continue

            latest_row = ch_rows[-1]
            t_rel = float(latest_row[1])
            quaternion = latest_row[3:7].astype(float)
            current = self.latest_samples[ch]

            should_accept = (
                current is None
                or t_rel > current["t_rel"]
                or (current["t_rel"] - t_rel) > TIME_RESET_THRESHOLD_S
            )
            if not should_accept:
                continue

            if (
                current is not None
                and (current["t_rel"] - t_rel) > TIME_RESET_THRESHOLD_S
            ):
                self.yaw_baseline.pop(ch, None)
                self.next_yaw_log_t.pop(ch, None)

            try:
                rotation_from_quaternion(quaternion)
            except ValueError as exc:
                self.status_label.setText(f"Quaternion failed for CH{ch}: {exc}")
                continue

            self.latest_samples[ch] = {
                "t_rel": t_rel,
                "quaternion": quaternion,
            }
            accepted_channels.append(ch)

        if not accepted_channels:
            return

        self._refresh_visualization()
        for ch in accepted_channels:
            sample = self.latest_samples[ch]
            self._update_yaw_log(ch, sample["t_rel"], self._relative_yaw(ch))

        newest_t = max(
            sample["t_rel"]
            for sample in self.latest_samples.values()
            if sample is not None
        )
        self.status_label.setText(
            f"Streaming Slicer-style bike-relative spine at {newest_t:.2f}s"
        )

    def _absolute_rotation(self, ch):
        sample = self.latest_samples[ch]
        if sample is None:
            return None
        return rotation_from_quaternion(sample["quaternion"])

    def _relative_rotation(self, ch):
        sensor_rotation = self._absolute_rotation(ch)
        if sensor_rotation is None:
            return R.identity()

        bike_rotation = self._absolute_rotation(BIKE_CHANNEL)
        if bike_rotation is None or ch == BIKE_CHANNEL:
            return sensor_rotation

        return bike_rotation.inv() * sensor_rotation

    def _relative_yaw(self, ch):
        return float(self._relative_rotation(ch).as_euler("xyz", degrees=True)[2])

    def _spine_control_points(self):
        lower_rotation = self._relative_rotation(LOWER_SPINE_CHANNEL)
        upper_rotation = self._relative_rotation(UPPER_SPINE_CHANNEL)

        lower_up = normalize(lower_rotation.apply([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])
        upper_up = normalize(upper_rotation.apply([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])
        spine_length = np.linalg.norm(UPPER_SPINE_POINT - LOWER_SPINE_POINT)
        tangent_scale = spine_length * 0.38

        p0 = LOWER_SPINE_POINT
        p3 = UPPER_SPINE_POINT
        p1 = p0 + lower_up * tangent_scale
        p2 = p3 - upper_up * tangent_scale
        return p0, p1, p2, p3

    def _refresh_visualization(self):
        self._update_sensor_axes()
        self._update_spine_model()

    def _update_sensor_axes(self):
        for ch in CHANNELS:
            rotation = self._relative_rotation(ch)
            position = CHANNEL_POSITIONS[ch]
            axes = rotation.apply(np.eye(3))

            for i, line in enumerate(self.axis_items[ch]):
                points = np.vstack([position, position + axes[i] * AXIS_LENGTH])
                line.setData(
                    pos=points,
                    color=AXIS_COLORS[i],
                    width=4,
                    antialias=True,
                )

    def _update_spine_model(self):
        p0, p1, p2, p3 = self._spine_control_points()
        curve_t = np.linspace(0.0, 1.0, SPINE_CURVE_POINTS)
        curve = cubic_bezier(p0, p1, p2, p3, curve_t)
        self.spine_curve.setData(
            pos=curve,
            color=CURVE_COLOR,
            width=5,
            antialias=True,
        )

        vertebra_t = np.linspace(0.05, 0.95, SPINE_SEGMENTS)
        for item, t in zip(self.vertebra_items, vertebra_t):
            center = cubic_bezier(p0, p1, p2, p3, [t])[0]
            tangent = cubic_bezier_tangent(p0, p1, p2, p3, float(t))
            item.setMeshData(
                meshdata=make_oriented_ellipsoid_mesh(
                    self.base_vertices,
                    self.base_faces,
                    center,
                    tangent,
                    VERTEBRA_SIZE,
                )
            )

        side = self._body_side_direction()
        left_points = curve + side * 0.18
        right_points = curve - side * 0.18
        self.body_panel.setMeshData(meshdata=make_panel_mesh(left_points, right_points))

    def _body_side_direction(self):
        lower_side = self._relative_rotation(LOWER_SPINE_CHANNEL).apply([0.0, 1.0, 0.0])
        upper_side = self._relative_rotation(UPPER_SPINE_CHANNEL).apply([0.0, 1.0, 0.0])
        return normalize(lower_side + upper_side, [0.0, 1.0, 0.0])

    def _update_yaw_log(self, ch, t_rel, yaw):
        if ch not in self.yaw_baseline:
            self.yaw_baseline[ch] = yaw
            self.next_yaw_log_t[ch] = t_rel + YAW_LOG_INTERVAL_S
            self._append_yaw_log_line(ch, "start", yaw, 0.0, ("start", ch))
            return

        if t_rel >= self.next_yaw_log_t[ch]:
            drift = wrap_angle_delta(yaw - self.yaw_baseline[ch])
            minute_index = int(self.next_yaw_log_t[ch] // YAW_LOG_INTERVAL_S)
            cluster_key = ("minute", minute_index)
            self._append_yaw_log_line(
                ch,
                f"{minute_index}min",
                yaw,
                drift,
                cluster_key,
            )
            while t_rel >= self.next_yaw_log_t[ch]:
                self.next_yaw_log_t[ch] += YAW_LOG_INTERVAL_S

    def _append_yaw_log_line(self, ch, label, yaw, drift, cluster_key):
        if (
            self.current_log_cluster is not None
            and cluster_key != self.current_log_cluster
        ):
            self.yaw_log.appendPlainText("-----------------------------------------------------------")
        self.current_log_cluster = cluster_key

        if ch == BIKE_CHANNEL:
            name = "bike"
        elif ch == LOWER_SPINE_CHANNEL:
            name = "lower"
        else:
            name = "upper"

        self.yaw_log.appendPlainText(
            f"CH{ch} {name:>5} | {label:>8} | rel_yaw={yaw:8.2f} deg | "
            f"drift={drift:+7.2f} deg"
        )

    def closeEvent(self, event):
        self.timer.stop()
        self._close_remote()
        super().closeEvent(event)


def main():
    app = pg.mkQApp("Bike-Relative Spine Viewer")
    window = SpineSlicerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
