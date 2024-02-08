import os
import numpy as np
import cv2 as cv
import streamlit as st

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

def video_feed():
    st.title("Video Feed")
    vid = cv.VideoCapture(0)
    vid.set(3, 640)
    vid.set(4, 480)
    st_frame = st.empty()

    while True:
        ret, frame = vid.read()
        
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
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
                
        st_frame.image(frame)        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    
def image_feed():
    st.title("Image Feed")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    
    if uploaded_file is not None:
        st.write("Image uploaded successfully.")
        image = cv.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            
            id, confidence = recognizer.predict(face)
            if confidence > 75:
                cv.putText(image, f"{labels[id]}",
                           (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (144, 238, 144), 2)
                cv.rectangle(image, (x, y), (x+w, y+h), (144, 238, 144), 2)
            else:
                cv.putText(image, "Unknown",
                           (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                cv.rectangle(image, (x, y), (x+w, y+h), (0, 165, 255), 2)
        
        st.image(image)
        
def main():
    video_feed()
    # choice = st.sidebar.button("Video Feed")
    # if choice:
    #     video_feed()

    # choice = st.sidebar.button("Image Feed")
    # if choice:
    #     image_feed()
        
main()

