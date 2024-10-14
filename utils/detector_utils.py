# utils/detector_utils.py

import math
import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Union
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 255, 0)  # green
ANNOTATION_COLOR = (255, 255, 255)  # white

class FaceDetector:
    def __init__(self, model_asset_path: str):
        try:
            base_options = python.BaseOptions(model_asset_path=model_asset_path)
            options = vision.FaceDetectorOptions(base_options=base_options)
            self.detector = vision.FaceDetector.create_from_options(options)
            self.keypoints = []
            self.bounding_boxes = []
        except Exception as e:
            raise ValueError(f"Error loading face detection model: {str(e)}")

    def _normalized_to_pixel_coordinates(
        self, normalized_x: float, normalized_y: float, image_width: int, image_height: int
    ) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates.

        Args:
            normalized_x (float): The normalized x-coordinate value.
            normalized_y (float): The normalized y-coordinate value.
            image_width (int): The width of the image.
            image_height (int): The height of the image.

        Returns:
            Union[None, Tuple[int, int]]: The pixel coordinates corresponding to the normalized values.
                Returns None if the normalized values are invalid.

        """
        def is_valid_normalized_value(value: float) -> bool:
            return (0 <= value <= 1) or math.isclose(value, 0) or math.isclose(value, 1)

        if not (is_valid_normalized_value(normalized_x) and is_valid_normalized_value(normalized_y)):
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    def visualize(
        self,
        image,
        detection_result,
        show_label_score=True,
        show_keypoints=True,
    ) -> np.ndarray:
        """Draws bounding boxes and keypoints on the input image and return it.

        Args:
            image (np.ndarray): The input RGB image.
            detection_result: The list of all "Detection" entities to be visualized.
            show_label_score (bool): Whether to show the label and score. Default is True.
            show_keypoints (bool): Whether to show the keypoints. Default is True.

        Returns:
            np.ndarray: Image with bounding boxes.

        """
        annotated_image = image.copy()
        height, width, _ = image.shape

        for detection in detection_result.detections:
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            
            cv2.line(annotated_image, start_point, (start_point[0] + 20, start_point[1]), ANNOTATION_COLOR, 3)
            cv2.line(annotated_image, start_point, (start_point[0], start_point[1] + 20), ANNOTATION_COLOR, 3)
            cv2.line(annotated_image, (end_point[0], start_point[1]), (end_point[0] - 20, start_point[1]), ANNOTATION_COLOR, 3)
            cv2.line(annotated_image, (end_point[0], start_point[1]), (end_point[0], start_point[1] + 20), ANNOTATION_COLOR, 3)
            cv2.line(annotated_image, (start_point[0], end_point[1]), (start_point[0] + 20, end_point[1]), ANNOTATION_COLOR, 3)
            cv2.line(annotated_image, (start_point[0], end_point[1]), (start_point[0], end_point[1] - 20), ANNOTATION_COLOR, 3)
            cv2.line(annotated_image, end_point, (end_point[0] - 20, end_point[1]), ANNOTATION_COLOR, 3)
            cv2.line(annotated_image, end_point, (end_point[0], end_point[1] - 20), ANNOTATION_COLOR, 3)
    
            self.bounding_boxes.append((start_point, end_point))

            if show_keypoints:
                keypoints = []
                for keypoint in detection.keypoints:
                    keypoint_px = self._normalized_to_pixel_coordinates(
                        keypoint.x, keypoint.y, width, height
                    )
                    keypoints.append(keypoint_px)
                    color, thickness, radius = (0, 255, 0), 2, 2
                    cv2.circle(annotated_image, keypoint_px, thickness, color, radius)
                self.keypoints.append(keypoints)

            if show_label_score:
                category = detection.categories[0]
                category_name = category.category_name or ''
                probability = round(category.score, 2)
                result_text = f"{category_name} ({probability})"
                text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
                cv2.putText(
                    annotated_image,
                    result_text,
                    text_location,
                    cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE,
                    TEXT_COLOR,
                    FONT_THICKNESS,
                )

        return annotated_image


if __name__ == "__main__":
    # Create an instance of the FaceDetector class
    face_detector = FaceDetector(model_asset_path='./utils/weights/detector.tflite')

    # Load the input image
    image = mp.Image.create_from_file("./utils/data/parth (2).jpg")

    # Detect faces in the input image
    detection_result = face_detector.detector.detect(image)

    # Process the detection result and visualize it
    image_copy = np.copy(image.numpy_view())
    annotated_image = face_detector.visualize(image_copy, detection_result)

    # Show the bounding box coordinates and keypoints coordinates
    for i, (start_point, end_point) in enumerate(face_detector.bounding_boxes):
        cv2.putText(
            annotated_image,
            f"Box {i+1}: {start_point}, {end_point}",
            (MARGIN, 30 + i * ROW_SIZE),
            cv2.FONT_HERSHEY_PLAIN,
            FONT_SIZE,
            TEXT_COLOR,
            FONT_THICKNESS,
        )

    for i, keypoints in enumerate(face_detector.keypoints):
        for j, keypoint in enumerate(keypoints):
            cv2.putText(
                annotated_image,
                f"Keypoint {j+1}: {keypoint}",
                (MARGIN, 30 + (i + 1) * ROW_SIZE + j * ROW_SIZE),
                cv2.FONT_HERSHEY_PLAIN,
                FONT_SIZE,
                TEXT_COLOR,
                FONT_THICKNESS,
            )

    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    cv2.imshow("Face Detection", rgb_annotated_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
