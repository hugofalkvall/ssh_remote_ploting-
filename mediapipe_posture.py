import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
OUT_PATH = "mediapipe_angles.txt"
TARGET_HZ = 10
SPLIT_RATIO = 0.5

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return v
    return v / n


def angle_signed(v1, v2):
    v1 = normalize(v1)
    v2 = normalize(v2)

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))

    # 2D signed angle
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    if cross < 0:
        angle *= -1

    return angle


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError(
            "Could not open camera 0. On macOS, allow camera access for the terminal "
            "or Python app in System Settings > Privacy & Security > Camera."
        )

    with open(OUT_PATH, "w", encoding="utf-8"):
        pass

    frame_time = 1.0 / TARGET_HZ

    try:
        with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose, open(OUT_PATH, "a", encoding="utf-8") as output_file:
            program_start_time = time.time()
            loop_start = time.time()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(
                        "Camera opened but no frames were returned. Check that no other "
                        "app is using the webcam."
                    )

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(img_rgb)
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark

                    hip_l = np.array([lm[23].x, lm[23].y], dtype=np.float32)
                    hip_r = np.array([lm[24].x, lm[24].y], dtype=np.float32)
                    pelvis = (hip_l + hip_r) / 2.0

                    sh_l = np.array([lm[11].x, lm[11].y], dtype=np.float32)
                    sh_r = np.array([lm[12].x, lm[12].y], dtype=np.float32)
                    thorax = (sh_l + sh_r) / 2.0

                    shoulder_axis = normalize(sh_r - sh_l)
                    chest_dir = normalize(
                        np.array([-shoulder_axis[1], shoulder_axis[0]], dtype=np.float32)
                    )

                    if chest_dir[1] > 0:
                        chest_dir = -chest_dir

                    spine_axis = thorax - pelvis
                    split_pt = pelvis + SPLIT_RATIO * spine_axis

                    lower_vec = split_pt - pelvis
                    upper_vec = chest_dir
                    vertical = np.array([0.0, -1.0], dtype=np.float32)

                    lower_angle = angle_signed(lower_vec, vertical)
                    upper_angle = angle_signed(upper_vec, vertical)
                    intersegment_angle = angle_signed(upper_vec, lower_vec)

                    h, w, _ = img.shape
                    p = tuple((pelvis * [w, h]).astype(int))
                    s = tuple((split_pt * [w, h]).astype(int))
                    t = tuple((thorax * [w, h]).astype(int))
                    u = tuple(((split_pt + upper_vec * 0.20) * [w, h]).astype(int))

                    color_upper = (0, 255, 0) if abs(intersegment_angle) <= 5 else (0, 0, 255)

                    cv2.line(img, p, s, (255, 255, 255), 3)
                    cv2.line(img, s, u, color_upper, 3)
                    cv2.line(img, s, t, (100, 100, 100), 1)
                    cv2.circle(img, p, 6, (255, 255, 255), -1)
                    cv2.circle(img, s, 6, (0, 255, 255), -1)
                    cv2.circle(img, t, 6, (255, 255, 255), -1)

                    cv2.putText(
                        img,
                        f"Lower angle: {int(lower_angle)} deg",
                        (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        img,
                        f"Upper angle: {int(upper_angle)} deg",
                        (30, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color_upper,
                        2,
                    )
                    cv2.putText(
                        img,
                        f"Between seg: {int(intersegment_angle)} deg",
                        (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color_upper,
                        2,
                    )
                    output_file.write(
                        f"{time.time()},{time.time() - program_start_time:.4f},"
                        f"{lower_angle:.2f},{upper_angle:.2f},{intersegment_angle:.2f}\n"
                    )
                    output_file.flush()

                cv2.imshow("Spine segmented in two parts", img)

                elapsed = time.time() - loop_start
                sleep_time = frame_time - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                loop_start = time.time()

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
