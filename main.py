from dt_apriltags import Detector
from os import path
import matplotlib.pyplot as plt

from common import *


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


K = np.array(cameraParams['K']).reshape((3,3)) # Get camera matrix
tags = detectTags(at_detector, colorImg, K, apriltagParams['tagSize'])


fig = plt.figure()

drawImage(fig, colorImg, K, tags)
# plt.savefig('result.pdf', format='pdf')
# plt.show()


# True dist = 163
tag = tags[0]
R, t = tag.pose_R, tag.pose_t
t = np.reshape(t, (3,))
X = np.array([0, 0, 0])
Xc = transform(X, R, t)
print(Xc)
