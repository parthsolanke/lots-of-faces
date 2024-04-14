import cv2
import asyncio
import numpy as np
import mediapipe as mp

import utils.detector_utils as dtu
import utils.image_utils as imu

VIDEO_PATH = 0  # Webcam
SAVE_DIR = "./dynamic/data"
WEIGHTS_PATH = "./utils/weights/detector.tflite"

async def run_inference():
    """
    Runs the face detection inference on a video stream.

    This function initializes a face detector, captures frames from a video stream,
    performs face detection on each frame, and displays the annotated image with
    detected faces. It also provides the option to capture multiple images by pressing
    the 'c' key.

    Returns:
        None
    """
    face_detector = dtu.FaceDetector(model_asset_path=WEIGHTS_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    image_manager = imu.ImageManager(save_dir=SAVE_DIR)
    image_manager.create_directory()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detection_result = face_detector.detector.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame
            )
        )
        annotated_image = face_detector.visualize(
            np.copy(frame),
            detection_result,
            show_keypoints=False,
            show_label_score=False
        )

        cv2.imshow("Face Detection", annotated_image)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            cv2.destroyWindow("Face Detection")
            await asyncio.create_task(image_manager.capture_multiple_images(frame))

        await asyncio.sleep(0)

    image_manager.clear_directory()
    cap.release()

async def main():
    await asyncio.create_task(run_inference())

try:
    asyncio.run(main())
finally:
    cv2.destroyAllWindows()
