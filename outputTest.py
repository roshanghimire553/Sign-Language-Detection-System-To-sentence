import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from tensorflow.keras.models import load_model
import os

# Disable OneDNN optimizations to prevent errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# Initialize video capture and hand detector
capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)  # Main hand detector

# Paths you provided
white_image_path = "C:\\Users\\RG_AK\\Desktop\\FinalProject\\white1.png"

model_path="ASLMODEL.h5"
# model_path="hand_gesture_cnn_model.h5"
# model_path="cnn8grps_rad1_model.h5"


# Load the trained model
model = load_model(model_path)

# Labels corresponding to the classes
# labels = ["A", "B", "C", "D"]
labels = ['A', 'B', 'C', 'D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']  # Update with your actual labels if different

while True:
    # Capture the frame from the camera
    _, frame = capture.read()
    frame = cv2.flip(frame, 1)  # Flip frame for a mirror view

    # Detect hands in the frame
    hands, frame = hd.findHands(frame, draw=False, flipType=True)

    # Load the white image (400x400)
    white = cv2.imread(white_image_path)

    if hands:
        hand = hands[0]  # Extract the first hand
        lmList = hand['lmList']  # Landmark list
        bbox = hand['bbox']  # Bounding box of the hand

        # Extract bounding box coordinates
        x, y, w, h = bbox

        # Center the hand in the white image (400x400)
        os_x = (white.shape[1] - w) // 2
        os_y = (white.shape[0] - h) // 2

        # Draw skeleton (lines between landmarks) on the white image
        for i in range(0, 4):
            cv2.line(white, (lmList[i][0] - x + os_x, lmList[i][1] - y + os_y),
                     (lmList[i + 1][0] - x + os_x, lmList[i + 1][1] - y + os_y), (0, 255, 0), 3)
        for i in range(5, 8):
            cv2.line(white, (lmList[i][0] - x + os_x, lmList[i][1] - y + os_y),
                     (lmList[i + 1][0] - x + os_x, lmList[i + 1][1] - y + os_y), (0, 255, 0), 3)
        for i in range(9, 12):
            cv2.line(white, (lmList[i][0] - x + os_x, lmList[i][1] - y + os_y),
                     (lmList[i + 1][0] - x + os_x, lmList[i + 1][1] - y + os_y), (0, 255, 0), 3)
        for i in range(13, 16):
            cv2.line(white, (lmList[i][0] - x + os_x, lmList[i][1] - y + os_y),
                     (lmList[i + 1][0] - x + os_x, lmList[i + 1][1] - y + os_y), (0, 255, 0), 3)
        for i in range(17, 20):
            cv2.line(white, (lmList[i][0] - x + os_x, lmList[i][1] - y + os_y),
                     (lmList[i + 1][0] - x + os_x, lmList[i + 1][1] - y + os_y), (0, 255, 0), 3)

        # Additional connections for a proper hand skeleton
        cv2.line(white, (lmList[5][0] - x + os_x, lmList[5][1] - y + os_y),
                 (lmList[9][0] - x + os_x, lmList[9][1] - y + os_y), (0, 255, 0), 3)
        cv2.line(white, (lmList[9][0] - x + os_x, lmList[9][1] - y + os_y),
                 (lmList[13][0] - x + os_x, lmList[13][1] - y + os_y), (0, 255, 0), 3)
        cv2.line(white, (lmList[13][0] - x + os_x, lmList[13][1] - y + os_y),
                 (lmList[17][0] - x + os_x, lmList[17][1] - y + os_y), (0, 255, 0), 3)
        cv2.line(white, (lmList[0][0] - x + os_x, lmList[0][1] - y + os_y),
                 (lmList[5][0] - x + os_x, lmList[5][1] - y + os_y), (0, 255, 0), 3)
        cv2.line(white, (lmList[0][0] - x + os_x, lmList[0][1] - y + os_y),
                 (lmList[17][0] - x + os_x, lmList[17][1] - y + os_y), (0, 255, 0), 3)

        # Draw landmarks (circles) on the white image
        for i in range(21):
            cv2.circle(white, (lmList[i][0] - x + os_x, lmList[i][1] - y + os_y), 5, (0, 0, 255), cv2.FILLED)

        # Preprocess the image for model prediction
        input_image = cv2.resize(white, (200, 200))  # Resize to the input size
        input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension
        input_image = input_image / 255.0  # Normalize the image

        # Make prediction
        prediction = model.predict(input_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        label = labels[predicted_class]

        # Display predicted label on frame
        frame = cv2.putText(frame, f"Predicted: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Display the white image with the hand skeleton
    cv2.imshow("Hand Skeleton", white)

    # Handle key presses
    interrupt = cv2.waitKey(1)
    if interrupt & 0xFF == 27:  # Exit on 'ESC' key
        break

# Release resources
capture.release()
cv2.destroyAllWindows()
