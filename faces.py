import cv2 as cv
import os

vid = cv.VideoCapture(0)
vid.set(3, 640)
vid.set(4, 480)


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

while True:
    ret, frame = vid.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # detect faces
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        
        id, confidence = recognizer.predict(face)
        if confidence > 75:
            cv.putText(frame, f"{labels[id]}",
                       (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (144, 238, 144), 2)
            cv.rectangle(frame, (x,y), (x+w, y+h), (144, 238, 144), 2)
        else:
            cv.putText(frame, "Unknown",
                       (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (0,165,255), 2)
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,165,255), 2)
    
    cv.imshow('frame', frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
vid.release()
cv.destroyAllWindows()