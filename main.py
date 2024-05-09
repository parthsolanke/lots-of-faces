import cv2
import asyncio
import numpy as np
import mediapipe as mp

from utils.train_utils import Trainer
from utils.image_utils import ImageManager
from utils.detector_utils import FaceDetector
from utils.recognizer_utils import FaceRecognizer

VIDEO_PATH = 0  # Webcam
SAVE_DIR = "./dynamic/data"
SAVE_DIR_WEIGHTS = "./dynamic/weights"
WEIGHTS_PATH = "./utils/weights/detector.tflite"

async def detector_inference():
    """
    """
    face_detector = FaceDetector(model_asset_path=WEIGHTS_PATH)
    trainer = Trainer()

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    image_manager = ImageManager(save_dir=SAVE_DIR)
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
            # await asyncio.create_task(trainer.train(SAVE_DIR_WEIGHTS, image_manager.get_images()))
            

        await asyncio.sleep(0)

    image_manager.clear_directory()
    cap.release()
    
async def recognizer_inference():
    face_recognizer = FaceRecognizer(model_path="./dynamic/weights/trained_model.xml")

    cap = cv2.VideoCapture(VIDEO_PATH)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        recognized_faces = face_recognizer.recognize_faces(frame)
        for (label, confidence) in recognized_faces:
            cv2.putText(
                frame,
                f"Label: {label}, Confidence: {confidence}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        await asyncio.sleep(0)

    cap.release()

async def main():
    await asyncio.create_task(detector_inference())
    await asyncio.create_task(recognizer_inference())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        cv2.destroyAllWindows()
