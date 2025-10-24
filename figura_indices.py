import cv2
import mediapipe as mp
import math
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    left_index = None
    right_index = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handedness.classification[0].label
            h, w, _ = frame.shape
            index_tip = hand_landmarks.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            if label == 'Left':
                left_index = (x, y)
            elif label == 'Right':
                right_index = (x, y)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if left_index and right_index:
            cv2.circle(frame, left_index, 8, (255, 0, 0), -1)
            cv2.circle(frame, right_index, 8, (0, 0, 255), -1)

            dx = right_index[0] - left_index[0]
            dy = right_index[1] - left_index[1]
            distancia = math.hypot(dx, dy)

            cx = int((left_index[0] + right_index[0]) / 2)
            cy = int((left_index[1] + right_index[1]) / 2)

            lado = int(distancia)
            s = lado / 2.0

            # Cuadrado centrado en (0,0)
            base = np.array([[-s, -s],
                             [ s, -s],
                             [ s,  s],
                             [-s,  s]], dtype=np.float32)

            # Ángulo de la línea entre índices (rotación cuando subes/bajas un dedo)
            ang = math.degrees(math.atan2(dy, dx))
            rad = math.radians(ang)
            c, si = math.cos(rad), math.sin(rad)
            R = np.array([[c, -si],
                          [si,  c]], dtype=np.float32)

            rotado = (base @ R.T)
            rotado[:, 0] += cx
            rotado[:, 1] += cy
            pts = rotado.astype(np.int32).reshape((-1, 1, 2))

            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=3)

    cv2.imshow("Cuadrado rotado por indices", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
