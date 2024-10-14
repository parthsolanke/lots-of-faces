# utils/train_utils.py

import cv2
import os
import shutil
import asyncio
import numpy as np
import tkinter as tk
import mediapipe as mp
from utils.detector_utils import FaceDetector
from utils.file_utils import create_directory

class Trainer:
    def __init__(self):
        self.WEIGHTS_PATH = "./utils/weights/detector.tflite"
        self.face_detector = FaceDetector(model_asset_path=self.WEIGHTS_PATH)

    def preprocess(self, img):
        res = self.face_detector.detector.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=img
            )
        )
        for detection in res.detections:
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            face = img[start_point[1]:end_point[1], start_point[0]:end_point[0]]
            if face.size == 0:
                print("No face detected in the image.")
                return None
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            return face_gray
        
        print("No detections found.")
        return None

    async def train(self, save_dir, images_list):        
        self.write_name_to_file(save_dir)
        
        faces = []
        labels = []
        index = []
        label_file_path = os.path.join(save_dir, "labels.txt")
        
        if not images_list:
            print("No images provided for training.")
            return
        
        for i, label in enumerate(open(label_file_path).read().splitlines()):
            for img in images_list:
                image = cv2.imread(img)
                if image is None:
                    print(f"Could not read image: {img}")
                    continue
                
                face = self.preprocess(image)
                if face is not None:
                    faces.append(face)
                    labels.append(label)
                    index.append(i)
        
        if len(faces) == 0:
            print("No faces found for training.")
            return
        
        index = np.array(index)
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, index)
        
        self.save_model_and_labels(recognizer, labels)
        await asyncio.sleep(0)

    def save_model_and_labels(self, recognizer, labels):
        model_dir = "./dynamic/weights"
        self.create_directory(model_dir)
        
        model_file_path = os.path.join(model_dir, "recognizer.yml")
        recognizer.save(model_file_path)
        
        labels_file_path = os.path.join(model_dir, "labels.txt")
        with open(labels_file_path, "w") as file:
            file.write("\n".join(labels))
            
    def clear_model_and_labels(self, save_dir):
        create_directory(save_dir)

    def create_directory(self, path):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def write_name_to_file(self, dir, file_name='/labels.txt'):
        self.create_directory(dir)
        root = tk.Tk()
        root.title("Enter your name:")
        
        var = tk.StringVar()
        entry = tk.Entry(root, textvariable=var)
        entry.pack()
        
        button = tk.Button(root, text="Enter", command=root.quit)
        button.pack()
        
        root.mainloop()
        name = var.get()
        root.destroy()

        with open(dir + file_name, 'w') as file:
            file.write(name)
        return name
