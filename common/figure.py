import matplotlib.pyplot as plt
import numpy as np
from common.frame import drawFrame

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
    plt.axis('off')