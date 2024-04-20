import numpy as np
import cv2
from common.transformation import Pose

fontFace = cv2.FONT_HERSHEY_PLAIN
fontScale = 0.8
lineThickness = 3

def drawTags(img, K, tags):
    
    for tag in tags:
        # Draw contour of tag
        for idx in range(len(tag.corners)):
            cv2.line(img, tuple(tag.corners[idx-1, :].astype(int)),
                     tuple(tag.corners[idx, :].astype(int)), (0, 255, 0), thickness=lineThickness)

        # Add tag id next to tag
        cv2.putText(img, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                    fontFace=fontFace,
                    fontScale=fontScale,
                    color=(255, 0, 0))

        
        # Draw coordinate frame of tag
        R, t = tag.pose_R, tag.pose_t.reshape((3,))
        T = Pose(R, t)
        drawCoordinateAxes(img, K, T.T, scale=0.1, labels=True)



def drawCoordinateAxes(img, K, T, scale=1, labels=False):
    X = T @ np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]])
    u, v = project(K, X)
    u, v = u.astype(int), v.astype(int)
    cv2.line(img, [u[0], v[0]], [u[1], v[1]], color=(255, 0, 0), thickness=lineThickness)
    cv2.line(img, [u[0], v[0]], [u[2], v[2]], color=(0, 255, 0), thickness=lineThickness)
    cv2.line(img, [u[0], v[0]], [u[3], v[3]], color=(0, 0, 255), thickness=lineThickness)
    if labels:
        cv2.putText(img, 'X', (u[1], v[1]), fontFace, fontScale, (255, 255, 255))
        cv2.putText(img, 'Y', (u[2], v[2]), fontFace, fontScale, (255, 255, 255))
        cv2.putText(img, 'Z', (u[3], v[3]), fontFace, fontScale, (255, 255, 255))


def project(K, X):
    """
    Computes the pinhole projection of a 3xN array of 3D points X
    using the camera intrinsic matrix K. Returns the dehomogenized
    pixel coordinates as an array of size 2xN.
    """
    uvw = K@X[:3,:]
    uvw /= uvw[2,:]
    return uvw[:2,:]