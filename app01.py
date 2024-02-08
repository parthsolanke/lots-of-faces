import os 
import numpy as np
import cv2 as cv
import requests
import streamlit as st
from deepface import DeepFace

emotion_model = DeepFace.build_model("Emotion")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]


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

def infer(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        
        id, confidence = recognizer.predict(face)
        if confidence > 65:
            cv.putText(img, f"{labels[id]}",
                        (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (144, 238, 144), 2)
            cv.rectangle(img, (x, y), (x+w, y+h), (144, 238, 144), 2)
        else:
            cv.putText(img, "Unknown",
                        (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 165, 255), 2)
    return img

def fetchQuote(emotion):

    reqUrl = "http://192.168.169.229:3001/api/v1/quote?emotion="+emotion

    payload = ""

    response = requests.request("GET", reqUrl, data=payload)
    json_response = response.json()
    # st.json(json_response)
    quote = json_response["quote"]["quote"]
    author = json_response["quote"]["author"]

    st.markdown(f"> {quote}\n> \n> {author}")
def image_feed():
    st.title("Image Feed")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

    if uploaded_file is not None:
        st.write("Image uploaded successfully.")
        image = cv.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        
        image = cv.resize(image, (640, 480))
        image = infer(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        

def video_feed():
    st.title("Video Feed")
    vid = cv.VideoCapture(0)
    vid.set(3, 640)
    vid.set(4, 480)
    st_frame = st.empty()
    
    while True:
        ret, frame = vid.read()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = infer(frame)
        
        # # emotion prediction
        # reshaped_img = cv.resize(frame, (48, 48), interpolation=cv.INTER_AREA)
        # normalized_img = reshaped_img.astype("float32") / 255.0
        # reshaped_img = np.reshape(normalized_img, (1, 48, 48, 1))
        
        # preds = emotion_model.predict(reshaped_img)
        # label = emotion_labels[np.argmax(preds)]
        
        # st.write(f"Emotion: {label}")
        st_frame.image(frame)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    image_feed()
    # st.title("Face Recognition App")
    
    # choice = st.radio("Select Mode", ("Image", "Video"))
    # if choice == "Image":
    #     image_feed()
    # else:
    #     video_feed()
        
if __name__ == "__main__":
    main()