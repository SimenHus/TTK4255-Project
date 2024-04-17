from os import path, listdir
import yaml
import numpy as np


# Define paths
IMAGE_FOLDER = path.join(path.curdir, 'images')
APRILTAGS = path.join(path.curdir, 'apriltags', 'tag36h11')
PARAMS_FOLDER = path.join(path.curdir, 'params')
CALIBRATION = path.join(path.curdir, 'calibration')

# List of images in image folder
IMGS = listdir(IMAGE_FOLDER)

# Load parameters from yaml file
with open('parameters.yaml', 'r') as f:
    parameters = yaml.safe_load(f)

# Parameter sections
apriltagParams = parameters['apriltags']
cameraParams = {'K': np.loadtxt(path.join(CALIBRATION, 'K.txt'))}