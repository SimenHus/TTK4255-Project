from dt_apriltags import Detector
import numpy as np
from os import path, listdir
import cv2
import yaml
import imutils
from common import drawImage
import matplotlib.pyplot as plt

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

# Initialize the apriltags detector
at_detector = Detector(families=apriltagParams['family'],
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)


imgPath = path.join(IMAGE_FOLDER, IMGS[0]) # Choose image to use
colorImg = cv2.imread(imgPath) # Read image from file
colorImg = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)


grayImg = cv2.cvtColor(colorImg, cv2.COLOR_RGB2GRAY) # Convert image to grayscale
intrinsicMatrix = np.array(cameraParams['K']).reshape((3,3)) # Get camera matrix
# Gather useful info from camera matrix
cameraCalibrationParams = (intrinsicMatrix[0,0], intrinsicMatrix[1,1],
                           intrinsicMatrix[0,2], intrinsicMatrix[1,2])

tags = at_detector.detect(grayImg, estimate_tag_pose=True, camera_params=cameraCalibrationParams,
                          tag_size=apriltagParams['tagSize']) # Detect tags



fig = plt.figure()

drawImage(fig, colorImg, intrinsicMatrix, tags)
plt.savefig('result.pdf', format='pdf')
plt.show()