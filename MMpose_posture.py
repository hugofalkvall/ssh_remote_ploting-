import cv2
import numpy as np
from mmpose.apis import MMPoseInferencer

# ✅ Stabil och kompatibel modell
inferencer = MMPoseInferencer('human')

def normalize(v):
    n = np.linalg.norm(v)
    return v if n < 1e-6 else v / n

def angle_signed(v1, v2):
    v1 = normalize(v1)
    v2 = normalize(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    cross = v1[0]*v2[1] - v1[1]*v2[0]
    if cross < 0:
        angle *= -1
    return angle

cap = cv2.VideoCapture(0)
split_ratio = 0.5

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = next(inferencer(frame))
    preds = result.get('predictions', [])

    # ✅ 1. Finns någon person?
    if len(preds) == 0 or len(preds[0]) == 0:
        cv2.imshow("MMPose Spine", frame)
        continue

    # ✅ 2. Välj bästa person
    best_person = None
    best_score = -1

    for person in preds[0]:
        scores = person.get('keypoint_scores', [])
        if len(scores) == 0:
            continue
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_person = person

    if best_person is None:
        cv2.imshow("MMPose Spine", frame)
        continue

    keypoints = best_person['keypoints']
    scores = best_person.get('keypoint_scores', None)

    # ✅ 3. Validera viktiga punkter
    required = [5, 6, 11, 12]  # shoulders + hips
    valid = True

    if scores is not None:
        for i in required:
            if scores[i] < 0.4:
                valid = False

    for i in required:
        if keypoints[i][0] == 0 and keypoints[i][1] == 0:
            valid = False

    if not valid:
        cv2.imshow("MMPose Spine", frame)
        continue

    try:
        # --- BODY POINTS ---
        hip_l = np.array(keypoints[11][:2])
        hip_r = np.array(keypoints[12][:2])
        pelvis = (hip_l + hip_r) / 2

        sh_l = np.array(keypoints[5][:2])
        sh_r = np.array(keypoints[6][:2])
        thorax = (sh_l + sh_r) / 2

        spine_axis = thorax - pelvis
        split_pt = pelvis + split_ratio * spine_axis

        lower_vec = split_pt - pelvis
        upper_vec = thorax - split_pt

        vertical = np.array([0, -1])

        lower_angle = angle_signed(lower_vec, vertical)
        upper_angle = angle_signed(upper_vec, vertical)
        between = angle_signed(upper_vec, lower_vec)

        # --- DRAW ---
        p = tuple(pelvis.astype(int))
        s = tuple(split_pt.astype(int))
        t = tuple(thorax.astype(int))

        color = (0, 255, 0)
        if abs(between) > 5:
            color = (0, 0, 255)

        cv2.line(frame, p, s, (255, 255, 255), 3)
        cv2.line(frame, s, t, color, 3)

        cv2.circle(frame, p, 6, (255, 255, 255), -1)
        cv2.circle(frame, s, 6, (0, 255, 255), -1)
        cv2.circle(frame, t, 6, (255, 255, 255), -1)

        cv2.putText(frame, f"Lower: {int(lower_angle)}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.putText(frame, f"Upper: {int(upper_angle)}", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"Between: {int(between)}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    except Exception as e:
        print("Error:", e)

    cv2.imshow("MMPose Spine", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()