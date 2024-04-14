import cv2
import numpy as np
import mediapipe as mp
import utils.detector as dtr

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (0, 255, 0)  # green
ANNOTATION_COLOR = (255, 0, 0)  # red


# Create an instance of the FaceDetector class
face_detector = dtr.FaceDetector(model_asset_path='./utils/weights/detector.tflite')

# Load the input image
image = mp.Image.create_from_file("./utils/test/parth (2).jpg")

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
