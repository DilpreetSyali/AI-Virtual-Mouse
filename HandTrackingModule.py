import cv2
import mediapipe as mp
import pyautogui

# Initialize mediapipe hands solution and drawing options
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands()

# Open camera capture
camera = cv2.VideoCapture(0)

# Check if the camera is opened
if not camera.isOpened():
    print("Error: Camera not opened!")
    exit()

# Get screen width and height for mouse control
screen_width, screen_height = pyautogui.size()

# Variables to track previous position for scrolling
prev_y = None
dragging = False

while True:
    _, image = camera.read()
    
    # Check if the frame was successfully captured
    if image is None:
        print("Error: Failed to capture image from camera.")
        break
    
    # Flip the image horizontally for a mirror effect
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Convert image back to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # If hands are detected, process each hand
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the position of the index finger tip (landmark 8)
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * screen_width)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * screen_height)

            # Move the mouse to the position of the index finger tip
            pyautogui.moveTo(x, y)

            # Gesture for left click (when thumb and index finger are touching)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            distance = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5

            if distance < 0.05:  # If the thumb and index finger are close
                pyautogui.click()

            # Detect scrolling by tracking vertical hand movement
            if prev_y is not None:
                if prev_y - y > 50:  # Moving hand up
                    pyautogui.scroll(5)  # Scroll up
                elif y - prev_y > 50:  # Moving hand down
                    pyautogui.scroll(-5)  # Scroll down

            prev_y = y

            # Gesture for dragging (when index and middle fingers are close)
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            if distance < 0.05 and not dragging:
                pyautogui.mouseDown()
                dragging = True
            elif distance > 0.1 and dragging:
                pyautogui.mouseUp()
                dragging = False

    # Show the video feed with landmarks
    cv2.imshow("Hand Movement Video Capture", image)

    # Exit the loop when the 'Esc' key is pressed
    key = cv2.waitKey(1)  # Change this to 1 for non-blocking
    if key == 27:
        break

# Release camera and close OpenCV windows
camera.release()
cv2.destroyAllWindows()
