import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from common.transformation import Pose
import cv2
import numpy as np

class Visualize:


    def __init__(self, map, frames, FPS=60, manualPlayback=False,
                 fixedFrameOffset=None, axisScale=0.1, resizeImg=None):
        self.FPS = FPS
        self.axisScale = axisScale
        self.manualPlayback = manualPlayback
        self.fixedFrameOffset = fixedFrameOffset if fixedFrameOffset is not None else Pose()


        self.fig = plt.figure()
        self.graph = self.fig.add_subplot(projection='3d')
        self.map = map


        self.prepareGraph()
        self.frames = []
        for frame in frames:
            if resizeImg is not None: frame = cv2.resize(frame, resizeImg)
            self.frames.append(frame)
        self.size = frame.shape[:2]

    def prepareGraph(self):
        self.graph.set_aspect('equal')
        self.graph.set_xticks([])
        self.graph.set_yticks([])
        self.graph.set_zticks([])
        
        self.cameraGraphCoordSys = self.drawCoordinateAxes('c', self.map.trajectory[0])
        for id, pose in self.map.landmarks.items():
            self.drawCoordinateAxes(id, pose, labels=True)

    def animStep(self, timeStep):
        pose = self.fixedFrameOffset@self.map.trajectory[timeStep]
        origo = pose.t
        for i, axis in enumerate(self.cameraGraphCoordSys[:3]):
            dir = origo + self.axisScale*pose.R[:, i]
            axis.set_data(((origo[0], dir[0]), (origo[1], dir[1])))
            axis.set_3d_properties((origo[2], dir[2]))

        self.cameraGraphCoordSys[3].set_x(origo[0])
        self.cameraGraphCoordSys[3].set_y(origo[1])
        self.cameraGraphCoordSys[3].set_3d_properties(origo[2])

        
    def drawCoordinateAxes(self, id, pose: Pose, labels=False):
        pose = self.fixedFrameOffset@pose
        origo = pose.t
        x = origo + self.axisScale*pose.R[:, 0]
        y = origo + self.axisScale*pose.R[:, 1]
        z = origo + self.axisScale*pose.R[:, 2]
        xAxis, = self.graph.plot((origo[0], x[0]), (origo[1], x[1]), (origo[2], x[2]), color='#cc4422') # X-axis
        yAxis, = self.graph.plot((origo[0], y[0]), (origo[1], y[1]), (origo[2], y[2]), color='#11ff33') # Y-axis
        zAxis, = self.graph.plot((origo[0], z[0]), (origo[1], z[1]), (origo[2], z[2]), color='#3366ff') # Z-axis
        o = self.graph.text(origo[0], origo[1], origo[2], id)
        if labels:
            textargs = {'color': 'w', 'va': 'center', 'ha': 'center', 'fontsize': 'x-small'}
            self.graph.text(x[0], x[1], x[2], 'X')
            self.graph.text(y[0], y[1], y[2], 'Y')
            self.graph.text(z[0], z[1], z[2], 'Z')

        return xAxis, yAxis, zAxis, o
    

    def play(self):
        started = False
        for i, frame in enumerate(self.frames):
            self.animStep(i)
            self.fig.canvas.draw()
            graph = cv2.cvtColor(np.asarray(self.fig.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)

            # vPadding = int((frame.shape[0] - graph.shape[0])/2)
            # hPadding = int((frame.shape[1] - graph.shape[1])/2)

            # graph = cv2.copyMakeBorder(graph, vPadding, vPadding,
            #                            hPadding, hPadding, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            graph = cv2.resize(graph, self.size[::-1])

            image = np.hstack((frame, graph))
            cv2.imshow('Video', image)

            if not started and not self.manualPlayback:
                print('Playback ready, press any key to begin...')
                cv2.waitKey(0)
                started = True


            if self.manualPlayback: cv2.waitKey(0)
            elif cv2.waitKey(int(1000/self.FPS)) & 0xFF == ord('q'): break

        cv2.destroyAllWindows()