from dt_apriltags import Detector
import numpy as np
from os import path
import cv2
import yaml


IMAGE_FOLDER = path.join(path.curdir, 'images')
APRILTAGS = path.join(path.curdir, 'apriltags', 'tag36h11')
PARAMS_FOLDER = path.join(path.curdir, 'params')


with open('parameters.yaml', 'r') as f:
    parameters = yaml.safe_load(f)

apriltagParams = parameters['apriltags']
cameraParams = parameters['camera']

at_detector = Detector(families=apriltagParams['family'],
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

imgName = 'test_image_multiple_01.png'
colorImg = cv2.imread(path.join(IMAGE_FOLDER, imgName))
grayImg = cv2.cvtColor(colorImg, cv2.COLOR_RGB2GRAY)
intrinsicMatrix = np.array(cameraParams['K']).reshape((3,3))
cameraCalibrationParams = (intrinsicMatrix[0,0], intrinsicMatrix[1,1],
                           intrinsicMatrix[0,2], intrinsicMatrix[1,2])

tags = at_detector.detect(grayImg, estimate_tag_pose=False, camera_params=cameraCalibrationParams)
print(tags)

for tag in tags:
    for idx in range(len(tag.corners)):
        cv2.line(colorImg, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

    cv2.putText(colorImg, str(tag.tag_id),
                org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 255))

cv2.imshow('Detected tags', colorImg)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()