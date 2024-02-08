import cv2 as cv
import os

detector = cv.CascadeClassifier("./haar_face.xml")
if not os.listdir("./models"):
    print("No model found.")
    exit(0)
else:
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.read("./models/face_recognizer.yml")
    labels = []
    with open("./models/labels.txt", "r") as f:
        labels = f.read().split("\n")
        
frame = cv.imread(r"./data\parth\parth (7).jpg")
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

# detect faces
faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
for (x,y,w,h) in faces:
    face = gray[y:y+h, x:x+w]
    
    id, confidence = recognizer.predict(face)
    cv.putText(frame, f"{labels[id]}",
                (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (144, 238, 144), 2)
    cv.rectangle(frame, (x,y), (x+w, y+h), (144, 238, 144), 2)
    
cv.imshow('Detected Face', frame)

cv.waitKey(0)