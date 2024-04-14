import cv2
import numpy as np
import mediapipe as mp
import utils.detector_utils as dtu
import utils.image_utils as imu

VIDEO_PATH = 0  # Webcam
SAVE_DIR = "./dynamic/data"
WEIGHTS_PATH = "./utils/weights/detector.tflite"

face_detector = dtu.FaceDetector(model_asset_path=WEIGHTS_PATH)
image_manager = imu.ImageManager(save_dir=SAVE_DIR)

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

image_manager.clear_directory()
image_manager.create_directory()

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = face_detector.detector.detect(image)
        image_copy = np.copy(frame)
        annotated_image = face_detector.visualize(
            image_copy,
            detection_result,
            show_keypoints=False,
            show_label_score=False
        )

        cv2.imshow("Face Detection", annotated_image)

        # Check for key press events
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            image_manager.capture_image(frame)
finally:
    cap.release()
    cv2.destroyAllWindows()
