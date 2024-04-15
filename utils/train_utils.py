import cv2
import os
import shutil
import asyncio
import numpy as np
import tkinter as tk
import mediapipe as mp
from utils.detector_utils import FaceDetector

class Trainer:
    """
    A class that represents a trainer for face recognition models.
    """

    def __init__(self):
        self.WEIGHTS_PATH = "./utils/weights/detector.tflite"
        self.face_detector = FaceDetector(model_asset_path=self.WEIGHTS_PATH)

    def preprocess(self, img):
        """
        Preprocess the input image by detecting faces.
        
        Args:
            img (numpy.ndarray): The input image as a NumPy array.
        
        Returns:
            numpy.ndarray: The extracted face from the input image.
        """
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
            return face

    async def train(self, save_dir, images_list):
        """
        Train the model on the extracted faces.
        
        Args:
            save_dir (str): The directory path to save the trained model and labels.
            images_list (list): A list of image file paths to train the model on.
        """
        
        self.write_name_to_file(save_dir)
    
        faces = []
        labels = []
        index = []
        label_file_path = os.path.join(save_dir, "labels.txt")
        for i, label in enumerate(open(label_file_path).read().splitlines()):
            for img in images_list:
                image = cv2.imread(img)
                face = self.preprocess(image)
                if face is not None:
                    faces.append(face)
                    labels.append(label)
                    index.append(i)
        index = np.array(index)
        recognizer = cv2.face.LBPHFaceRecognizer()
        recognizer.train(faces, index)
        
        self.save_model_and_labels(recognizer, labels)
        await asyncio.sleep(0)

    def save_model_and_labels(self, recognizer, labels):
        """
        Save the trained model and labels.

        Args:
            recognizer: The trained face recognizer model.
            labels: The list of labels corresponding to the trained faces.
        """
        model_dir = "./dynamic/weights"
        self.create_directory(model_dir)
        
        model_file_path = os.path.join(model_dir, "recognizer.yml")
        recognizer.save(model_file_path)
        
        labels_file_path = os.path.join(model_dir, "labels.txt")
        with open(labels_file_path, "w") as file:
            file.write("\n".join(labels))
            
    def clear_model_and_labels(self, save_dir):
        """
        Clear the save model and labels directory by deleting all files and subdirectories.
        
        Args:
            save_dir (str): The directory path where the model and labels are saved.
        """
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    def create_directory(self, path):
        """
        Create a new directory at the given path by clearing the existing directory.
        
        Args:
            path (str): The path of the directory to be created.
        """
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    def write_name_to_file(self, dir, file_name='/labels.txt'):
        """
        Take input of a name using a GUI and write that name in the labels.txt file.
        
        Args:
            dir (str): The directory path where the labels.txt file will be created.
            file_name (str, optional): The name of the labels.txt file. Defaults to '/labels.txt'.
        
        Returns:
            str: The name entered by the user.
        """
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
