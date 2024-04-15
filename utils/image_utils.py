import os
import cv2
import numpy as np
import asyncio

class ImageManager:
    """
    A class that manages the capturing and saving of images.

    Args:
        save_dir (str): The directory where the captured images will be saved.

    Attributes:
        save_dir (str): The directory where the captured images will be saved.
        file_list (list): A list of file names in the save directory.

    Methods:
        create_directory: Creates the save directory if it doesn't exist and clears any existing files.
        capture_image: Captures and saves an image.
        capture_num_images: Captures and saves num images at a given interval.
        capture_multiple_images: Captures and saves multiple images with progress info.
        clear_directory: Clears all files in the save directory.
        get_file_list: Returns a list of file names in the save directory.
        get_images: Returns a list of images in the save directory.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        self.file_list = []
            
    def get_file_list(self):
        """
        Returns a list of file names in the save directory.
        """
        self.file_list = os.listdir(self.save_dir)
        return self.file_list
    
    def create_directory(self):
        """
        Creates the save directory if it doesn't exist and clears any existing files.
        """
        os.makedirs(self.save_dir, exist_ok=True)
        self.file_list = self.get_file_list()
        for file_name in self.file_list:
            file_path = os.path.join(self.save_dir, file_name)
            os.remove(file_path)
        
    def capture_image(self, frame):
        """
        Captures and saves an image.

        Args:
            frame: The image frame to be saved.
        """
        counter = 0
        while True:
            filename = os.path.join(self.save_dir, f"image_{counter}.jpg")
            if not os.path.exists(filename):
                break
            counter += 1
        cv2.imwrite(filename, frame)
        print("Image captured and saved as", filename)
        
    def capture_num_images(self, frame, num, interval=1000):
        """
        Captures and saves num images at a given interval.

        Args:
            frame: The image frame to be saved.
            num (int): The number of images to capture.
            interval (int): The interval between captures in milliseconds.
        """
        if interval <= 0:
            interval = 1
        for i in range(num):
            self.capture_image(frame)
            cv2.waitKey(interval)

    def clear_directory(self):
        """
        Clears all files in the save directory.
        """
        self.file_list = os.listdir(self.save_dir)
        for file_name in self.file_list:
            file_path = os.path.join(self.save_dir, file_name)
            os.remove(file_path)
        print("Directory cleared")
        
    def get_images(self):
        """
        Returns a list of images in the save directory.
        """
        image_list = []
        for file_name in self.file_list:
            if file_name.endswith(".jpg"):
                image_path = os.path.join(self.save_dir, file_name)
                image_list.append(image_path)
        return image_list
    
    async def capture_multiple_images(self, frame):
        """
        Captures and saves multiple images with progress info.

        Args:
            frame: The image frame to be saved.
        """
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


if __name__ == "__main__":
    # Test the ImageManager class
    save_directory = "/path/to/save/directory"
    image_manager = ImageManager(save_directory)
    
    # Create the save directory
    image_manager.create_directory()
    
    # Capture and save an image
    frame = cv2.imread("path/to/image.jpg")
    image_manager.capture_image(frame)
    
    # Get the list of images in the save directory
    images = image_manager.get_images()
    print("Images in the save directory:", images)
    
    # Clear the save directory
    image_manager.clear_directory()
    
    # Capture multiple images with a progress bar
    asyncio.run(image_manager.capture_multiple_images(frame))
