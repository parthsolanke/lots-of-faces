import os
import cv2 as cv
import numpy as np

TRAIN_DIR = "./data"
MODELS_DIR = "./models"

# labels
labels = []
for i in os.listdir(TRAIN_DIR):
    labels.append(i)

detector = cv.CascadeClassifier("./haar_face.xml")

def preprocess(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        return face
    
def train():
    faces = []
    ids = []
    for label in labels:
        for img in os.listdir(f"{TRAIN_DIR}/{label}"):
            image = cv.imread(f"{TRAIN_DIR}/{label}/{img}")
            face = preprocess(image)
            if face is not None:
                print(f"Training {label}/{img}")
                faces.append(face)
                ids.append(labels.index(label))
    ids = np.array(ids)
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, ids)
    
    if len(os.listdir(MODELS_DIR)) == 0:
        print(f"Saving model and labels")
        # saving model
        recognizer.save(f"{MODELS_DIR}/face_recognizer.yml")
        # saving labels
        with open(f"{MODELS_DIR}/labels.txt", "w") as f:
            f.write("\n".join(labels))
    else:
        for i in os.listdir(MODELS_DIR):
            print(f"Removing {i}")
            os.remove(f"{MODELS_DIR}/{i}")
        # saving model
        recognizer.save(f"{MODELS_DIR}/face_recognizer.yml")
        # saving labels
        with open(f"{MODELS_DIR}/labels.txt", "w") as f:
            f.write("\n".join(labels))

if __name__ == "__main__":
    train()
    print("Training complete.")
    