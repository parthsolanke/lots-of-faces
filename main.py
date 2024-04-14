import cv2
import numpy as np
import mediapipe as mp
import utils.detector as dtr

# Create an instance of the FaceDetector class
face_detector = dtr.FaceDetector(model_asset_path='./utils/weights/detector.tflite')

# Open the video file
video_path = 0 # Set to 0 to use the webcam
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Create an instance of the mediapipe Image class
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Detect faces in the frame
    detection_result = face_detector.detector.detect(image)

    # Process the detection result and visualize it
    image_copy = np.copy(frame)
    annotated_image = face_detector.visualize(
        image_copy,
        detection_result,
        show_keypoints=False,
        show_label_score=False
    )

    # Display the annotated frame
    cv2.imshow("Face Detection", annotated_image)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
