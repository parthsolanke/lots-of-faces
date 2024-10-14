# utils/image_utils.py

import os
import cv2
import numpy as np
import asyncio
from utils.file_utils import create_directory

class ImageManager:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.file_list = []

    def create_directory(self):
        create_directory(self.save_dir)

    def capture_image(self, frame):
        counter = 0
        while True:
            filename = os.path.join(self.save_dir, f"image_{counter}.jpg")
            if not os.path.exists(filename):
                break
            counter += 1
        cv2.imwrite(filename, frame)
        print(f"Image captured and saved as: {filename}")

    def get_images(self):
        self.file_list = [f for f in os.listdir(self.save_dir) if f.endswith('.jpg')]
        image_list = []

        for file_name in self.file_list:
            image_path = os.path.join(self.save_dir, file_name)
            image_list.append(image_path)
            print(f"Adding image to list: {image_path}")

        return image_list

    def clear_directory(self):
        for file_name in self.file_list:
            file_path = os.path.join(self.save_dir, file_name)
            try:
                os.remove(file_path)
                print(f"Removed file: {file_path}")
            except Exception as e:
                print(f"Error removing file: {file_path} - {e}")

        self.file_list.clear()
        print("Image directory cleared.")

    async def capture_multiple_images(self, frame):
        num_images = 50
        interval = 250
        for i in range(num_images):
            self.capture_image(frame)
            
            black_screen = np.zeros_like(frame)
            fill_percentage = (i + 1) / num_images
            fill_height = int(fill_percentage * black_screen.shape[0])
            color = (255, 255, 255)
            cv2.circle(
                black_screen,
                (black_screen.shape[1] // 2, black_screen.shape[0] // 2),
                fill_height,
                color,
                -1
            )
            cv2.putText(
                black_screen,
                f"{int(fill_percentage * 100)}%",
                (((black_screen.shape[1]) // 2) - 30,
                 (black_screen.shape[0]) // 2 ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 0),
                1
            )
            
            cv2.imshow("Face Shape", black_screen)
            key = cv2.waitKey(interval)
            if key == ord('q'):
                break

            await asyncio.sleep(0)
                
        cv2.destroyWindow("Face Shape")
