from os import path, listdir
import yaml
import numpy as np
import cv2


# Define paths
IMAGE_FOLDER = path.join(path.curdir, 'assets', 'images')
APRILTAGS = path.join(path.curdir, 'assets', 'apriltags', 'tag36h11')
PARAMS_FOLDER = path.join(path.curdir, 'assets')
CALIBRATION = path.join(path.curdir, 'calibration')

# List of images in image folder
IMG_PATHS = [path.join(IMAGE_FOLDER, imgPath) for imgPath in listdir(IMAGE_FOLDER)]
IMG_PATHS.sort()

# Load parameters from yaml file
with open(path.join(PARAMS_FOLDER, 'parameters.yaml'), 'r') as f:
    parameters = yaml.safe_load(f)

# Parameter sections
apriltagParams = parameters['apriltags']
cameraParams = {'K': np.loadtxt(path.join(CALIBRATION, 'K.txt'))}