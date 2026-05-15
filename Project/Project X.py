
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ================= CAMERA SELECTION =================
def find_available_cameras(max_cameras=10):
    """Find all available camera devices"""
    available = []
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available

print("Scanning for available cameras...")
cameras = find_available_cameras()

if cameras:
    print(f"Found cameras at indices: {cameras}")
    if len(cameras) > 1:
        print("\nAvailable cameras:")
        for cam_idx in cameras:
            print(f"  Camera {cam_idx}")
        camera_index = int(input(f"Enter camera index (default 0): ") or "0")
        if camera_index not in cameras:
            print(f"Camera {camera_index} not found. Using camera 0.")
            camera_index = 0
    else:
        camera_index = cameras[0]
        print(f"Using camera {camera_index}")
else:
    print("No cameras found! Exiting.")
    exit()

# ================= SETUP =================
cam = cv2.VideoCapture(camera_index)

# 🔥 Reduce resolution (BIG performance boost)
cam.set(3, 640)
cam.set(4, 480)

screen_w, screen_h = pyautogui.size()
pyautogui.FAILSAFE = False

# Hand tracking (optimized)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=0
)

draw = mp.solutions.drawing_utils

last_click_time = 0

# Smooth movement (reduced lag)
prev_x, prev_y = 0, 0
smooth = 0.8  # Lower value = more sensitive (hand moves double)

# Frame skipping (extra performance boost)
frame_count = 0

# Create window for display (before main loop)
window_name = "Hand Mouse Control (Optimized)"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, screen_w, screen_h)  # Full screen (100%)
cv2.moveWindow(window_name, 0, 0)  # Move to top-left corner

# ================= MAIN LOOP =================
while True:

    ret, frame = cam.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue  # skip every alternate frame

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ================= HAND TRACKING =================
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:

            # ❌ Comment this if you want even more speed
            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            lm = hand.landmark

            # Finger tips
            index_tip = lm[8]
            thumb_tip = lm[4]
            middle_tip = lm[12]

            # Convert to screen coords
            index_x = screen_w * index_tip.x
            index_y = screen_h * index_tip.y

            thumb_x = screen_w * thumb_tip.x
            thumb_y = screen_h * thumb_tip.y

            middle_x = screen_w * middle_tip.x
            middle_y = screen_h * middle_tip.y

            # ================= SMOOTH MOUSE MOVE =================
            curr_x = prev_x + (index_x - prev_x) / smooth
            curr_y = prev_y + (index_y - prev_y) / smooth

            pyautogui.moveTo(curr_x, curr_y, duration=0)

            prev_x, prev_y = curr_x, curr_y

            # ================= DISTANCE =================
            thumb_index_dist = np.hypot(index_x - thumb_x, index_y - thumb_y)
            index_middle_dist = np.hypot(index_x - middle_x, index_y - middle_y)

            # ================= GESTURES =================

            # Double Click
            if thumb_index_dist < 40:
                cv2.putText(frame, "DOUBLE CLICK", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if time.time() - last_click_time > 1:
                    pyautogui.doubleClick()
                    last_click_time = time.time()

            # Left Click
            elif thumb_index_dist < 80:
                cv2.putText(frame, "LEFT CLICK", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if time.time() - last_click_time > 1:
                    pyautogui.click()
                    last_click_time = time.time()

            # Right Click
            elif index_middle_dist < 50:
                cv2.putText(frame, "RIGHT CLICK", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                if time.time() - last_click_time > 1:
                    pyautogui.rightClick()
                    last_click_time = time.time()

    # ================= DISPLAY =================
    cv2.imshow("Hand Mouse Control (Optimized)", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):  # Press 'f' to toggle fullscreen
        pass  # Fullscreen toggling handled below

# ================= CLEANUP =================
cam.release()
cv2.destroyAllWindows()
