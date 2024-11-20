import cv2
import mediapipe as mp
import pyautogui
import math

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)


# def for distance beetween two place
def calculate_distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


# for clicking
clicking = False
start_drag_position = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # detection for if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # status of your point finger and thumb
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = frame.shape
            thumb_pos = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))

            pyautogui.moveTo(index_pos[0], index_pos[1])

            distance = calculate_distance(thumb_pos, index_pos)
            if distance < 20:
                if not clicking:
                    pyautogui.mouseDown()
                    clicking = True
                    start_drag_position = index_pos
            else:
                if clicking:
                    pyautogui.mouseUp()
                    clicking = False

            # if clicking is anabel mouse will be in drag status
            if clicking and start_drag_position:
                offset_x = index_pos[0] - start_drag_position[0]
                offset_y = index_pos[1] - start_drag_position[1]
                pyautogui.move(offset_x, offset_y)

    cv2.imshow('Hand Gesture Mouse Control', frame)

    # press q to exit code
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
