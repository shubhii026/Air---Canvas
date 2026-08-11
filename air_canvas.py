import mediapipe as mp
import cv2
import numpy as np
import time
from collections import deque

# ---------- Setup ----------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

canvas = None

colors = {
    "Blue":  (255, 0, 0),
    "Green": (0, 255, 0),
    "Red":   (0, 0, 255),
    "Yellow":(0, 255, 255),
    "Eraser":(0, 0, 0)
}
color_names = list(colors.keys())
current_color = colors["Blue"]

# ---------- Brush size control ----------
brush_thickness = 8
eraser_thickness = 50
MIN_BRUSH = 2
MAX_BRUSH = 40

xp, yp = 0, 0

# ---------- Smoothing setup (moving average buffer + exponential) ----------
smooth_factor = 0.75   # higher = smoother but more delay
prev_x, prev_y = 0, 0
point_buffer_x = deque(maxlen=5)
point_buffer_y = deque(maxlen=5)


def fingers_up(landmarks):
    """Return list of booleans: [thumb, index, middle, ring, pinky] up or not"""
    tips = [4, 8, 12, 16, 20]
    fingers = []
    fingers.append(landmarks[tips[0]][0] > landmarks[tips[0]-1][0])
    for i in range(1, 5):
        fingers.append(landmarks[tips[i]][1] < landmarks[tips[i]-2][1])
    return fingers


def draw_menu(img):
    bar_h = 80
    seg_w = img.shape[1] // len(color_names)
    for i, name in enumerate(color_names):
        x1, x2 = i * seg_w, (i + 1) * seg_w
        col = colors[name] if name != "Eraser" else (50, 50, 50)
        cv2.rectangle(img, (x1, 0), (x2, bar_h), col, -1)
        cv2.putText(img, name, (x1 + 10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)
    return bar_h


print("Controls:")
print(" - INDEX finger only        -> Draw")
print(" - INDEX + MIDDLE           -> Select color from top bar")
print(" - '+' / '-'                -> Increase / Decrease brush size")
print(" - 'c'                      -> Clear canvas")
print(" - 'q'                      -> Quit")

while True:
    success, frame = cap.read()
    if not success:
        break
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w, 3), np.uint8)

    bar_h = draw_menu(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_lms in result.multi_hand_landmarks:
            lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms.landmark]
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

            x1, y1 = lm_list[8]   # index tip
            x2, y2 = lm_list[12]  # middle tip

            # ---------- Apply smoothing to index tip ----------
            point_buffer_x.append(x1)
            point_buffer_y.append(y1)
            avg_x = int(sum(point_buffer_x) / len(point_buffer_x))
            avg_y = int(sum(point_buffer_y) / len(point_buffer_y))

            if prev_x == 0 and prev_y == 0:
                prev_x, prev_y = avg_x, avg_y
            smooth_x = int(prev_x + (avg_x - prev_x) * (1 - smooth_factor))
            smooth_y = int(prev_y + (avg_y - prev_y) * (1 - smooth_factor))
            prev_x, prev_y = smooth_x, smooth_y
            x1, y1 = smooth_x, smooth_y

            fingers = fingers_up(lm_list)

            # Selection mode: index + middle up, others down
            if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                xp, yp = 0, 0
                if y1 < bar_h:
                    seg_w = w // len(color_names)
                    idx = x1 // seg_w
                    if 0 <= idx < len(color_names):
                        current_color = colors[color_names[idx]]
                cv2.rectangle(frame, (x1, y1 - 15), (x2, y2 + 15), current_color, cv2.FILLED)

            # Drawing mode: only index up
            elif fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
                cv2.circle(frame, (x1, y1), brush_thickness, current_color, cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                thickness = eraser_thickness if current_color == (0, 0, 0) else brush_thickness
                cv2.line(canvas, (xp, yp), (x1, y1), current_color, thickness)
                xp, yp = x1, y1
            else:
                xp, yp = 0, 0
    else:
        prev_x, prev_y = 0, 0

    # Show current brush size on screen
    cv2.putText(frame, f"Brush: {brush_thickness}", (w - 150, bar_h + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Merge canvas with live frame
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv_mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
    inv_mask = cv2.cvtColor(inv_mask, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, inv_mask)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.imshow("AI Air Canvas", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros((h, w, 3), np.uint8)
    elif key == ord('+') or key == ord('='):
        brush_thickness = min(brush_thickness + 2, MAX_BRUSH)
    elif key == ord('-') or key == ord('_'):
        brush_thickness = max(brush_thickness - 2, MIN_BRUSH)

cap.release()
cv2.destroyAllWindows()