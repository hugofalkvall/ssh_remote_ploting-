import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose


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


cap = cv2.VideoCapture(0)

# Hur långt upp delningspunkten ligger mellan pelvis (0.0) och thorax (1.0)
# Testa t.ex. 0.45–0.6 beroende på vad som ser bäst ut.
split_ratio = 0.5

with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(img_rgb)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        if res.pose_landmarks:
            try:
                lm = res.pose_landmarks.landmark

                # Höfter -> pelvis center
                hip_l = np.array([lm[23].x, lm[23].y], dtype=np.float32)
                hip_r = np.array([lm[24].x, lm[24].y], dtype=np.float32)
                pelvis = (hip_l + hip_r) / 2.0

                # Axlar -> thorax center
                sh_l = np.array([lm[11].x, lm[11].y], dtype=np.float32)
                sh_r = np.array([lm[12].x, lm[12].y], dtype=np.float32)
                thorax = (sh_l + sh_r) / 2.0

                # Axellinje
                shoulder_axis = sh_r - sh_l
                shoulder_axis = normalize(shoulder_axis)

                # Normal mot axellinjen = approx "thorax-riktning"
                chest_dir = np.array([-shoulder_axis[1], shoulder_axis[0]], dtype=np.float32)
                chest_dir = normalize(chest_dir)

                # Säkerställ att chest_dir pekar ungefär uppåt i bild
                # (i bildkoordinater är y mindre uppåt)
                if chest_dir[1] > 0:
                    chest_dir = -chest_dir

                # Delningspunkt på ryggradsaxeln
                spine_axis = thorax - pelvis
                split_pt = pelvis + split_ratio * spine_axis

                # Nedre segment: pelvis -> split
                lower_vec = split_pt - pelvis

                # Övre segment: använd thorax-orientering från axlarna
                upper_vec = chest_dir

                # Alternativ om du hellre vill ha rent geometriskt segment:
                # upper_vec = thorax - split_pt

                vertical = np.array([0.0, -1.0], dtype=np.float32)

                lower_angle = angle_signed(lower_vec, vertical)
                upper_angle = angle_signed(upper_vec, vertical)
                intersegment_angle = angle_signed(upper_vec, lower_vec)

                # Rita
                h, w, _ = img.shape
                p = tuple((pelvis * [w, h]).astype(int))
                s = tuple((split_pt * [w, h]).astype(int))
                t = tuple((thorax * [w, h]).astype(int))

                # För att visa riktningen på övre segmentet
                upper_end = split_pt + upper_vec * 0.20
                u = tuple((upper_end * [w, h]).astype(int))

                color_upper = (0, 255, 0)
                if abs(intersegment_angle) > 5:
                    color_upper = (0, 0, 255)

                # Nedre ryggsegment
                cv2.line(img, p, s, (255, 255, 255), 3)

                # Övre ryggsegment
                cv2.line(img, s, u, color_upper, 3)

                # Hjälplinje till thorax
                cv2.line(img, s, t, (100, 100, 100), 1)

                # Punkter
                cv2.circle(img, p, 6, (255, 255, 255), -1)
                cv2.circle(img, s, 6, (0, 255, 255), -1)
                cv2.circle(img, t, 6, (255, 255, 255), -1)

                # Text
                cv2.putText(
                    img,
                    f"Lower angle: {int(lower_angle)} deg",
                    (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    img,
                    f"Upper angle: {int(upper_angle)} deg",
                    (30, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color_upper,
                    2
                )

                cv2.putText(
                    img,
                    f"Between seg: {int(intersegment_angle)} deg",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color_upper,
                    2
                )

            except Exception:
                pass

        cv2.imshow("Spine segmented in two parts", img)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
