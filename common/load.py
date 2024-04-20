from os import path, listdir
import yaml
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


# Define paths
IMAGE_FOLDER = path.join(path.curdir, 'assets', 'images')
VIDEO_FOLDER = path.join(path.curdir, 'assets', 'videos')
APRILTAGS = path.join(path.curdir, 'assets', 'apriltags', 'tag36h11')
PARAMS_FOLDER = path.join(path.curdir, 'assets')
CALIBRATION = path.join(path.curdir, 'calibration')

# List of images in image folder
IMG_PATHS = [path.join(IMAGE_FOLDER, imgPath) for imgPath in listdir(IMAGE_FOLDER)]
IMG_PATHS.sort()

# List of videos from video folder
VIDEO_PATHS = [path.join(VIDEO_FOLDER, vidPath) for vidPath in listdir(VIDEO_FOLDER)]
VIDEO_PATHS.sort()

# Load parameters from yaml file
with open(path.join(PARAMS_FOLDER, 'parameters.yaml'), 'r') as f:
    parameters = yaml.safe_load(f)

# Parameter sections
apriltagParams = parameters['apriltags']
cameraParams = {'K': np.loadtxt(path.join(CALIBRATION, 'K.txt'))}
K = np.array(cameraParams['K']).reshape((3,3)) # Get camera matrix



def timeit(func):
    def wrapper(*args, **kwargs): 
        start = time.time() 
        result = func(*args, **kwargs) 
        end = time.time() 
        print(f'Successfully executed {func.__name__} in {end-start:.2f} seconds')
        return result 
    return wrapper


@timeit
def loadImages(imgPaths):
    frames = []
    for imgPath in imgPaths[:3]:
        img = cv2.imread(imgPath) # Read image from file
        frames.append(img)
    return frames


@timeit
def loadVideo(videoPath):
    cap = cv2.VideoCapture(videoPath)

    frames = []    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # if frame is read correctly ret is True
        if not ret: break
        frames.append(frame)
    cap.release()
    return frames