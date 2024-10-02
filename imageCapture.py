import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback

# Initialize video capture and hand detector
capture = cv2.VideoCapture(0)
hd = HandDetector(maxHands=1)  # Main hand detector

# Base dataset directory
base_dataset_path = "Dataset\\A"

# Initial directory for label 'A'
c_dir = 'A'
dataset_path = os.path.join(base_dataset_path, c_dir)
os.makedirs(dataset_path, exist_ok=True)  # Automatically create label directory if it doesn't exist

white_image_path = "white.jpg"  # Path to the white image with 400x400 resolution

# Get the count of images already in the dataset folder
count = len(os.listdir(dataset_path))

# Configuration variables
offset = 15  # Offset for ROI extraction
step = 1     # Step counter
flag = False  # Flag for image saving
suv = 0  # Incremental counter for image collection

while True:
    try:
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
            for i in range(0, 4,1):
                cv2.line(white, (lmList[i][0] - x + os_x, lmList[i][1] - y + os_y),
                         (lmList[i + 1][0] - x + os_x, lmList[i + 1][1] - y + os_y), (0, 255, 0), 3)
            for i in range(5, 8,1):
                cv2.line(white, (lmList[i][0] - x + os_x, lmList[i][1] - y + os_y),
                         (lmList[i + 1][0] - x + os_x, lmList[i + 1][1] - y + os_y), (0, 255, 0), 3)
            for i in range(9, 12,1):
                cv2.line(white, (lmList[i][0] - x + os_x, lmList[i][1] - y + os_y),
                         (lmList[i + 1][0] - x + os_x, lmList[i + 1][1] - y + os_y), (0, 255, 0), 3)
            for i in range(13,  16,1):
                cv2.line(white, (lmList[i][0] - x + os_x, lmList[i][1] - y + os_y),
                         (lmList[i + 1][0] - x + os_x, lmList[i + 1][1] - y + os_y), (0, 255, 0), 3)
            for i in range(17, 20,1):
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
                cv2.circle(white, (lmList[i][0] - x + os_x, lmList[i][1] - y + os_y), 2, (0, 0, 255), 1)

            # Display the white image with the hand skeleton
            cv2.imshow("Hand Skeleton", white)

        # Display frame with current status of dataset collection
        frame = cv2.putText(frame, f"dir={c_dir}  count={count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Frame", frame)

        # Handle key presses
        interrupt = cv2.waitKey(1)

        if interrupt & 0xFF == 27:  # Exit on 'ESC' key
            break

        if interrupt & 0xFF == ord('n'):  # Move to the next directory (label) on 'n' key press
            c_dir = chr(ord(c_dir) + 1)
            if ord(c_dir) > ord('Z'):  # Wrap around from Z to A
                c_dir = 'A'

            # Update dataset path and create directory if it doesn't exist
            dataset_path = os.path.join(base_dataset_path, c_dir)
            os.makedirs(dataset_path, exist_ok=True)

            # Update count based on the new directory
            count = len(os.listdir(dataset_path))

            flag = False  # Reset flag to ensure new label capture starts fresh
            suv = 0  # Reset image collection counter for new label

        if interrupt & 0xFF == ord('a'):  # Start/Stop image saving on 'a' key press
            flag = not flag  # Toggle the flag

        # Save skeleton images when the flag is True
        if flag and suv < 180:
            if step % 3 == 0:
                save_path = f"{dataset_path}\\{count}.jpg"
                cv2.imwrite(save_path, white)  # Save the white image with landmarks
                count += 1
                suv += 1
            step += 1

    except Exception as e:
        print("Error occurred:", traceback.format_exc())

# Release resources
capture.release()
cv2.destroyAllWindows()
