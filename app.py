import streamlit as st
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from PIL import Image
import base64

# Set page config
st.set_page_config(page_title="HandyPointer", page_icon=":hand:", layout="wide")

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load the logo and convert to base64 for background
logo_path = "logo.png"  # Replace with your logo path
logo_base64 = image_to_base64(logo_path)

# Streamlit app title and background style
st.markdown(
    f"""
    <style>
    body {{
        background-color: #1e1e1e;
        color: white;
        text-align: center;
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        flex-direction: column;
        margin: 0;
    }}
    .logo {{
        max-width: 200px;  /* Allow the logo to take up significant space */
        margin-top: 0px;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }}
    h1 {{
        font-family: 'Arial', sans-serif;
        color: white;
        font-size: 55px;
        margin-top: 20px;
        text-align: center;
        font-weight: bold;
    }}
    .stButton > button {{
        display: block;
        margin: 20px auto;
        padding: 10px 20px;
        font-size: 20px;
        background-color: #0078D4;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Add the logo image centered with markdown
st.markdown(
    f'<div style="text-align: center;"><img src="data:image/png;base64,{logo_base64}" class="logo" width="700"/></div>',
    unsafe_allow_html=True
)

# Add a line break
st.markdown("<br>", unsafe_allow_html=True)

# App title
st.markdown("<h1>HandyPointer: AI Virtual Mouse Using Hand Gestures</h1>", unsafe_allow_html=True)

# Buttons for Start and Stop Hand Tracking
start_button = st.button("Start Hand Tracking")
stop_button = st.button("Stop Hand Tracking")

# Initialize session state for tracking
if "tracking_active" not in st.session_state:
    st.session_state.tracking_active = False

# Start or Stop Tracking Logic
if start_button:
    st.session_state.tracking_active = True

if stop_button:
    st.session_state.tracking_active = False

# Placeholder for video feed
frame_placeholder = st.empty()

# Hand tracking logic
if st.session_state.tracking_active:
    st.info("Hand tracking is active! Perform gestures.")
    
    # Initialize MediaPipe hands solution
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands()

    # Get screen dimensions for mouse movement control
    screen_width, screen_height = pyautogui.size()

    # Initialize camera
    camera = cv2.VideoCapture(0)

    # Variables to track previous position for scrolling
    prev_y = None
    dragging = False

    while st.session_state.tracking_active:
        ret, frame = camera.read()
        
        if not ret:
            st.error("Failed to capture image from camera!")
            break

        # Flip the image horizontally for a mirror effect
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(frame)

        # Process hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get position of the index finger tip
                x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * screen_width)
                y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * screen_height)

                # Move mouse to index finger tip position
                pyautogui.moveTo(x, y)

                # Gesture for left-click (thumb and index finger close)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                distance = np.linalg.norm(np.array([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]))

                if distance < 0.05:
                    pyautogui.click()

                # Detect scrolling by tracking vertical hand movement
                if prev_y is not None:
                    if prev_y - y > 50:  # Hand moving up
                        pyautogui.scroll(5)
                    elif y - prev_y > 50:  # Hand moving down
                        pyautogui.scroll(-5)

                prev_y = y

                # Gesture for dragging (index and middle fingers close)
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                if distance < 0.05 and not dragging:
                    pyautogui.mouseDown()
                    dragging = True
                elif distance > 0.1 and dragging:
                    pyautogui.mouseUp()
                    dragging = False

        # Convert OpenCV frame to an image for Streamlit
        frame_placeholder.image(frame, channels="RGB", use_column_width=True)

    # Cleanup after stopping
    camera.release()
    cv2.destroyAllWindows()

# Show black screen when tracking is stopped
else:
    frame_placeholder.image(
        np.zeros((480, 640, 3), dtype=np.uint8),  # Black frame
        channels="RGB",
        use_column_width=True,
    )
    st.warning("Hand tracking is not active. Press 'Start' to begin.")
