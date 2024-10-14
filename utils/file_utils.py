# utils/file_utils.py

import os
import shutil

def create_directory(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

def list_files(directory: str, extension: str = None):
    if not os.path.exists(directory):
        return []
    
    return [
        os.path.join(directory, f) for f in os.listdir(directory)
        if extension is None or f.endswith(extension)
    ]
