"""
Run this file inside 3D Slicer's Python environment.

It uses Michael-M-Judd/spinal-segmentation-analysis by cloning/importing the
SpineSeg Slicer module, then adds a live bike-relative IMU spine visualization
to the active Slicer scene.
"""

import io
import os
import subprocess
import sys

import numpy as np

try:
    import vtk
    import qt
    import slicer
except ImportError as exc:
    raise RuntimeError(
        "This script must be run inside 3D Slicer, not normal Python."
    ) from exc

try:
    import paramiko
except ImportError as exc:
    raise RuntimeError(
        "paramiko is required in Slicer's Python environment for SSH polling."
    ) from exc


SCRIPT_NAME = "slicer_spineseg_quaternion_stream.py"
DEFAULT_SCRIPT_DIR = "/Users/hugofalkvall/local_test"

if (
    "__file__" in globals()
    and os.path.basename(os.path.abspath(__file__)) == SCRIPT_NAME
):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
else:
    # exec(open(...).read()) inside Slicer may inherit Slicer's own __file__.
    SCRIPT_DIR = DEFAULT_SCRIPT_DIR

REPO_URL = "https://github.com/Michael-M-Judd/spinal-segmentation-analysis.git"
REPO_DIR = os.path.join(SCRIPT_DIR, "external", "spinal-segmentation-analysis")
SPINESEG_MODULE_DIR = os.path.join(REPO_DIR, "SpineSeg")

pi_ip = os.environ.get("PI_IP", "raspberrypi.local")
username = "raspberrypi"
password = "paj"
remote_path = "/home/raspberrypi/Examensarbete/Posture-estimation-for-motorcycle-riders-using-IMU-based-systems/quaternions.txt"

BIKE_CHANNEL = 0
LOWER_SPINE_CHANNEL = 2
UPPER_SPINE_CHANNEL = 3
CHANNELS = [BIKE_CHANNEL, LOWER_SPINE_CHANNEL, UPPER_SPINE_CHANNEL]

POLL_INTERVAL_MS = 50
TIME_RESET_THRESHOLD_S = 5.0
CLEAR_SCENE_ON_START = False

BIKE_ORIGIN = np.array([-180.0, 0.0, 20.0], dtype=float)
LOWER_SPINE_POINT = np.array([0.0, 0.0, 30.0], dtype=float)
UPPER_SPINE_POINT = np.array([0.0, 0.0, 180.0], dtype=float)
CHANNEL_POSITIONS = {
    BIKE_CHANNEL: BIKE_ORIGIN,
    LOWER_SPINE_CHANNEL: LOWER_SPINE_POINT,
    UPPER_SPINE_CHANNEL: UPPER_SPINE_POINT,
}

AXIS_LENGTH = 45.0
SPINE_CURVE_POINTS = 120
SPINE_SEGMENTS = 13
VERTEBRA_RADIUS = 13.0
VERTEBRA_SCALE = (1.2, 1.8, 0.38)

AXIS_COLORS = [
    (1.0, 0.05, 0.05),  # X
    (0.05, 0.75, 0.1),  # Y
    (0.05, 0.25, 1.0),  # Z
]


def ensure_spineseg_repository():
    if not os.path.exists(SPINESEG_MODULE_DIR):
        os.makedirs(os.path.dirname(REPO_DIR), exist_ok=True)
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR])

    if SPINESEG_MODULE_DIR not in sys.path:
        sys.path.insert(0, SPINESEG_MODULE_DIR)

    from SpineSeg import SpineSegLogic  # pylint: disable=import-error

    return SpineSegLogic()


def quaternion_to_matrix(quaternion):
    quat = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Quaternion has zero length")

    qw, qx, qy, qz = quat / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=float,
    )


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


def make_polyline(points):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetNumberOfPoints(len(points))
    for i, point in enumerate(points):
        vtk_points.SetPoint(i, float(point[0]), float(point[1]), float(point[2]))

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(len(points))
    for i in range(len(points)):
        polyline.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(cells)
    return polydata


def make_model_node(name, polydata=None, color=(1.0, 1.0, 1.0), opacity=1.0, line_width=1):
    model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
    display = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelDisplayNode", f"{name}Display")
    display.SetColor(color)
    display.SetOpacity(opacity)
    display.SetLineWidth(line_width)
    model.SetAndObserveDisplayNodeID(display.GetID())
    if polydata is not None:
        model.SetAndObservePolyData(polydata)
    return model


def make_text_node(name, text, position):
    node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTextNode", name)
    node.SetText(text)

    fiducial = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"{name}Label")
    fiducial.AddControlPoint(vtk.vtkVector3d(*[float(x) for x in position]))
    fiducial.SetNthControlPointLabel(0, text)
    display = fiducial.GetDisplayNode()
    if display:
        display.SetTextScale(0.7)
        display.SetGlyphScale(0.5)
    return node


class SlicerSpineSegQuaternionStream:
    def __init__(self):
        self.spineseg_logic = ensure_spineseg_repository()
        self.latest_samples = {ch: None for ch in CHANNELS}
        self.ssh = None
        self.sftp = None

        self.spine_curve_node = None
        self.axis_nodes = {}
        self.vertebra_nodes = []
        self.vertebra_transforms = []
        self.status_node = None
        self.timer = qt.QTimer()
        self.timer.connect("timeout()", self.poll_remote_data)

        self._build_scene()

    def start(self):
        self._connect_remote()
        self.timer.start(POLL_INTERVAL_MS)
        slicer.util.infoDisplay(
            "SpineSeg quaternion stream started. Close Slicer or run stream.stop() to stop."
        )

    def stop(self):
        self.timer.stop()
        self._close_remote()

    def _build_scene(self):
        if CLEAR_SCENE_ON_START:
            slicer.mrmlScene.Clear(0)

        self.status_node = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLTextNode", "SpineSeg IMU Stream Status"
        )
        self.status_node.SetText("Preparing SpineSeg repository-based IMU viewer...")

        self._add_reference_geometry()

        self.spine_curve_node = make_model_node(
            "SpineSeg live spine curve",
            make_polyline(np.linspace(LOWER_SPINE_POINT, UPPER_SPINE_POINT, SPINE_CURVE_POINTS)),
            color=(0.05, 0.05, 0.05),
            opacity=1.0,
            line_width=6,
        )

        for i in range(SPINE_SEGMENTS):
            source = vtk.vtkSuperquadricSource()
            source.SetToroidal(False)
            source.SetPhiRoundness(0.7)
            source.SetThetaRoundness(0.7)
            source.SetSize(VERTEBRA_RADIUS)
            source.Update()

            node = make_model_node(
                f"SpineSeg vertebra {i + 1:02d}",
                source.GetOutput(),
                color=(0.82, 0.80, 0.72),
                opacity=1.0,
            )
            transform = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLLinearTransformNode",
                f"SpineSeg vertebra {i + 1:02d} transform",
            )
            node.SetAndObserveTransformNodeID(transform.GetID())
            self.vertebra_nodes.append(node)
            self.vertebra_transforms.append(transform)

        for ch, position in CHANNEL_POSITIONS.items():
            for axis_index, color in enumerate(AXIS_COLORS):
                node = make_model_node(
                    f"CH{ch} axis {axis_index}",
                    make_polyline(np.vstack([position, position])),
                    color=color,
                    opacity=1.0,
                    line_width=4,
                )
                self.axis_nodes[(ch, axis_index)] = node

        make_text_node("CH0BikeText", "CH0 bike reference", BIKE_ORIGIN + [5.0, 0.0, 12.0])
        make_text_node("CH2LowerText", "CH2 lower spine", LOWER_SPINE_POINT + [5.0, 0.0, 8.0])
        make_text_node("CH3UpperText", "CH3 upper spine", UPPER_SPINE_POINT + [5.0, 0.0, 8.0])

        self._refresh_visualization()
        slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
        three_d = slicer.app.layoutManager().threeDWidget(0).threeDView()
        three_d.resetFocalPoint()

    def _add_reference_geometry(self):
        # Add Slicer-like anatomical reference planes as wire rectangles.
        planes = [
            ("SagittalPlane", [0, 0, 0], [0, 0, 210], [0, 90, 0], (0.1, 0.25, 0.95)),
            ("CoronalPlane", [0, -90, 0], [0, -90, 210], [90, -90, 0], (0.95, 0.1, 0.1)),
            ("FloorPlane", [-120, -90, 0], [120, -90, 0], [120, 90, 0], (0.1, 0.55, 0.1)),
        ]
        for name, p0, p1, p2, color in planes:
            p0 = np.array(p0, dtype=float)
            p1 = np.array(p1, dtype=float)
            p2 = np.array(p2, dtype=float)
            p3 = p0 + (p2 - p1)
            points = np.vstack([p0, p1, p2, p3, p0])
            make_model_node(name, make_polyline(points), color=color, opacity=0.55, line_width=2)

    def _connect_remote(self):
        self._set_status(f"Connecting to {pi_ip} as {username}...")
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
            self.ssh = ssh
            self.sftp = ssh.open_sftp()
        except Exception as exc:
            self._set_status(f"Connection failed: {exc}. Retrying in 2s.")
            self._close_remote()
            qt.QTimer.singleShot(2000, self._connect_remote)
            return
        self._set_status("Connected. Waiting for quaternion data...")

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
            self._set_status(f"Read failed: {exc}. Reconnecting...")
            self._close_remote()
            qt.QTimer.singleShot(2000, self._connect_remote)
            return

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            self._set_status("Connected. Waiting for quaternion data...")
            return

        try:
            data = np.loadtxt(io.StringIO("\n".join(lines)), delimiter=",")
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] < 7:
                raise ValueError(f"Expected 7 columns, got {data.shape[1]}")
        except Exception as exc:
            self._set_status(f"Parse failed: {exc}")
            return

        accepted = False
        for ch in CHANNELS:
            rows = data[data[:, 0] == ch]
            if rows.size == 0:
                continue

            latest = rows[-1]
            t_rel = float(latest[1])
            quaternion = latest[3:7].astype(float)
            current = self.latest_samples[ch]
            should_accept = (
                current is None
                or t_rel > current["t_rel"]
                or (current["t_rel"] - t_rel) > TIME_RESET_THRESHOLD_S
            )
            if not should_accept:
                continue

            try:
                quaternion_to_matrix(quaternion)
            except ValueError as exc:
                self._set_status(f"Quaternion failed for CH{ch}: {exc}")
                continue

            self.latest_samples[ch] = {"t_rel": t_rel, "quaternion": quaternion}
            accepted = True

        if accepted:
            self._refresh_visualization()
            newest_t = max(
                sample["t_rel"]
                for sample in self.latest_samples.values()
                if sample is not None
            )
            self._set_status(f"Streaming SpineSeg/Slicer scene at {newest_t:.2f}s")

    def _absolute_matrix(self, ch):
        sample = self.latest_samples[ch]
        if sample is None:
            return np.eye(3)
        return quaternion_to_matrix(sample["quaternion"])

    def _relative_matrix(self, ch):
        sensor = self._absolute_matrix(ch)
        bike = self._absolute_matrix(BIKE_CHANNEL)
        if ch == BIKE_CHANNEL:
            return np.eye(3)
        return bike.T @ sensor

    def _spine_control_points(self):
        lower = self._relative_matrix(LOWER_SPINE_CHANNEL)
        upper = self._relative_matrix(UPPER_SPINE_CHANNEL)
        lower_up = normalize(lower @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])
        upper_up = normalize(upper @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0])
        length = np.linalg.norm(UPPER_SPINE_POINT - LOWER_SPINE_POINT)
        scale = length * 0.38
        p0 = LOWER_SPINE_POINT
        p3 = UPPER_SPINE_POINT
        return p0, p0 + lower_up * scale, p3 - upper_up * scale, p3

    def _refresh_visualization(self):
        self._update_axes()
        self._update_spine()

    def _update_axes(self):
        for ch, position in CHANNEL_POSITIONS.items():
            matrix = self._relative_matrix(ch)
            for axis_index in range(3):
                end = position + matrix[:, axis_index] * AXIS_LENGTH
                self.axis_nodes[(ch, axis_index)].SetAndObservePolyData(
                    make_polyline(np.vstack([position, end]))
                )

    def _update_spine(self):
        p0, p1, p2, p3 = self._spine_control_points()
        curve_t = np.linspace(0.0, 1.0, SPINE_CURVE_POINTS)
        curve = cubic_bezier(p0, p1, p2, p3, curve_t)
        self.spine_curve_node.SetAndObservePolyData(make_polyline(curve))

        vertebra_t = np.linspace(0.05, 0.95, SPINE_SEGMENTS)
        for node, transform, t in zip(self.vertebra_nodes, self.vertebra_transforms, vertebra_t):
            center = cubic_bezier(p0, p1, p2, p3, [float(t)])[0]
            tangent = normalize(
                cubic_bezier_tangent(p0, p1, p2, p3, float(t)),
                [0.0, 0.0, 1.0],
            )
            transform.SetMatrixTransformToParent(
                self._vertebra_transform_matrix(center, tangent)
            )

    def _vertebra_transform_matrix(self, center, tangent):
        z_axis = normalize(tangent, [0.0, 0.0, 1.0])
        x_axis = normalize(np.cross([0.0, 1.0, 0.0], z_axis), [1.0, 0.0, 0.0])
        y_axis = normalize(np.cross(z_axis, x_axis), [0.0, 1.0, 0.0])

        matrix = vtk.vtkMatrix4x4()
        basis = np.column_stack([x_axis, y_axis, z_axis])
        for row in range(3):
            matrix.SetElement(row, 0, basis[row, 0] * VERTEBRA_SCALE[0])
            matrix.SetElement(row, 1, basis[row, 1] * VERTEBRA_SCALE[1])
            matrix.SetElement(row, 2, basis[row, 2] * VERTEBRA_SCALE[2])
            matrix.SetElement(row, 3, float(center[row]))
        matrix.SetElement(3, 0, 0.0)
        matrix.SetElement(3, 1, 0.0)
        matrix.SetElement(3, 2, 0.0)
        matrix.SetElement(3, 3, 1.0)

        return matrix

    def _set_status(self, text):
        print(text)
        if self.status_node is not None:
            self.status_node.SetText(text)


try:
    stream.stop()
except Exception:
    pass

stream = SlicerSpineSegQuaternionStream()
stream.start()
