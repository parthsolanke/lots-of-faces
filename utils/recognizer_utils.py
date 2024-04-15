import cv2

class FaceRecognizer:
    def __init__(self, model_path):
        self.model = cv2.face.LBPHFaceRecognizer_create()
        self.model.read(model_path)

    def recognize_faces(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detect_faces(gray_image)

        recognized_faces = []
        for (x, y, w, h) in faces:
            roi = gray_image[y:y+h, x:x+w]
            label, confidence = self.model.predict(roi)
            recognized_faces.append((label, confidence))

        return recognized_faces

    def detect_faces(self, image):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces