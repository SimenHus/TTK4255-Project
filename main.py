from os import path
import matplotlib.pyplot as plt
import cv2

from common import *

K = np.array(cameraParams['K']).reshape((3,3)) # Get camera matrix
# Initialize the apriltags detector
detector = DetectorClass(K=K, tagSize=apriltagParams['tagSize'], families=apriltagParams['family'])
map = Map()


fig = plt.figure()

for imgPath in IMG_PATHS[:3]:
    img = cv2.imread(imgPath) # Read image from file
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tags = detector.detect(img)
    print('Found {} tags'.format(len(tags)))
    map.handleDetections(tags)
    pose = map.cameraPose

    drawImage(fig, img, K, tags)
    # plt.savefig('result.pdf', format='pdf')
    plt.xlabel(f'Position: {pose.pos}')
    plt.show()