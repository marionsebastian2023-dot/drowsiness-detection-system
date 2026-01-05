import cv2
import mediapipe as mp
import numpy as np
import math
import threading
from playsound3 import playsound
from scipy.spatial import distance

# ---------------------------
# CONFIG
# ---------------------------
ALARM_PATH = "alarm.mp3"
EYE_AR_THRESH = 0.25
MAR_THRESH = 4.5  # tuned for accurate yawning detection
EYE_AR_CONSEC_FRAMES = 18
MOUTH_AR_CONSEC_FRAMES = 18

# ---------------------------
# STATE
# ---------------------------
alarm_playing = False
closed_frames = 0
yawn_frames = 0

def play_alarm_loop():
    global alarm_playing
    while alarm_playing:
        playsound(ALARM_PATH)

# ---------------------------
# EAR / MAR FUNCTIONS
# ---------------------------
def euclidean_dist(p1, p2):
    return math.dist(p1, p2)

def eye_aspect_ratio(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    A = euclidean_dist(p2, p6)
    B = euclidean_dist(p3, p5)
    C = euclidean_dist(p1, p4)
    return (A + B) / (2.0 * C)

# --- accurate MAR using only key mouth landmarks ---
def mouth_aspect_ratio(landmarks):
    top_lip = np.array(landmarks[13])
    bottom_lip = np.array(landmarks[14])
    left = np.array(landmarks[78])
    right = np.array(landmarks[308])
    mar = distance.euclidean(top_lip, bottom_lip) / distance.euclidean(left, right)
    return mar * 10  # scaled for stability

# ---------------------------
# SETUP MEDIAPIPE
# ---------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    max_num_faces=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    # Normalize lighting for stability
    frame = cv2.flip(frame, 1)
    frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=30)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

            points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]

            LEFT_EYE = [33, 160, 158, 133, 153, 144]
            RIGHT_EYE = [362, 385, 387, 263, 373, 380]

            leftEAR = eye_aspect_ratio(points, LEFT_EYE)
            rightEAR = eye_aspect_ratio(points, RIGHT_EYE)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(points)

            # Drowsiness logic
            if ear < EYE_AR_THRESH:
                closed_frames += 1
            else:
                closed_frames = 0

            # Yawning logic 
            if mar > MAR_THRESH:
                yawn_frames += 1
            else:
                yawn_frames = 0

            if (closed_frames >= EYE_AR_CONSEC_FRAMES or
                yawn_frames >= MOUTH_AR_CONSEC_FRAMES) and not alarm_playing:
                alarm_playing = True
                threading.Thread(target=play_alarm_loop, daemon=True).start()

            if ear >= EYE_AR_THRESH and mar <= MAR_THRESH and alarm_playing:
                alarm_playing = False

            # Status display
            status_text = ""
            color = (0, 255, 0)
            if closed_frames >= EYE_AR_CONSEC_FRAMES:
                status_text = "DROWSY ALERT!"
                color = (0, 0, 255)
            elif yawn_frames >= MOUTH_AR_CONSEC_FRAMES:
                status_text = "YAWNING ALERT!"
                color = (0, 128, 255)
            else:
                status_text = "Normal"

            cv2.putText(frame, f"{status_text}", (40, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"EAR: {ear:.2f}  MAR: {mar:.2f}",
                        (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow("Drowsiness + Yawning Detection (Final)", frame)
    if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q'), 27]:
        break

alarm_playing = False
cap.release()
cv2.destroyAllWindows()
