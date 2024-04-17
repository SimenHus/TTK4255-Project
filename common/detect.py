import cv2
import numpy as np


def detectTags(at_detector, colorImg, K, tagSize):

    grayImg = cv2.cvtColor(colorImg, cv2.COLOR_RGB2GRAY) # Convert image to grayscale
    # Gather useful info from camera matrix
    cameraCalibrationParams = (K[0,0], K[1,1],
                               K[0,2], K[1,2])

    tags = at_detector.detect(grayImg, estimate_tag_pose=True, camera_params=cameraCalibrationParams,
                            tag_size=tagSize) # Detect tags
    
    return tags