import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

def drawImage(fig, img, K, tags):

    plt.subplot(121)
    plt.imshow(img)
    for tag in tags:

        # Draw box around tag
        for idx in range(len(tag.corners)):
            x = [tag.corners[idx-1, 0], tag.corners[idx, 0]]
            y = [tag.corners[idx-1, 1], tag.corners[idx, 1]]
            plt.plot(x, y, color='green')
        
        # Add tag id as text to image
        textOrigin = (tag.corners[0, 0].astype(int)+10, tag.corners[0, 1].astype(int)+10)
        plt.text(textOrigin[0], textOrigin[1], str(tag.tag_id), color='red')
        
        # Draw coordinate frame of tag
        R, t = tag.pose_R, tag.pose_t.reshape((3,))
        T = np.ones((4, 4))
        T[:3, :3] = R
        T[:3, 3] = t
        drawFrame(K, T, scale=0.1, labels=True)
        
    plt.xlim([0, img.shape[1]])
    plt.ylim([img.shape[0], 0])


def drawFrame(K, T, scale=1, labels=False):
    """
    Visualize the coordinate frame axes of the 4x4 object-to-camera
    matrix T using the 3x3 intrinsic matrix K.

    Control the length of the axes by specifying the scale argument.
    """
    X = T @ np.array([
        [0,scale,0,0],
        [0,0,scale,0],
        [0,0,0,scale],
        [1,1,1,1]])
    u,v = project(K, X)
    plt.plot([u[0], u[1]], [v[0], v[1]], color='#cc4422') # X-axis
    plt.plot([u[0], u[2]], [v[0], v[2]], color='#11ff33') # Y-axis
    plt.plot([u[0], u[3]], [v[0], v[3]], color='#3366ff') # Z-axis
    if labels:
        textargs = {'color': 'w', 'va': 'center', 'ha': 'center', 'fontsize': 'x-small', 'path_effects': [PathEffects.withStroke(linewidth=1.5, foreground='k')]}
        plt.text(u[1], v[1], 'X', **textargs)
        plt.text(u[2], v[2], 'Y', **textargs)
        plt.text(u[3], v[3], 'Z', **textargs)


def project(K, X):
    """
    Computes the pinhole projection of a 3xN array of 3D points X
    using the camera intrinsic matrix K. Returns the dehomogenized
    pixel coordinates as an array of size 2xN.
    """
    uvw = K@X[:3,:]
    uvw /= uvw[2,:]
    return uvw[:2,:]