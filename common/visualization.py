import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from common.transformation import Pose
import numpy as np

class Visualize:


    def __init__(self, map, FPS=60, fixedFrameOffset=None, axisScale=0.1):
        self.FPS = FPS
        self.axisScale = axisScale
        self.fixedFrameOffset = fixedFrameOffset if fixedFrameOffset is not None else Pose()


        self.fig = plt.figure()
        self.graph = self.fig.add_subplot(projection='3d')
        self.map = map
        self.nFrames = len(self.map.trajectory)

        self.prepareGraph()

    def prepareGraph(self):
        self.graph.set_aspect('equal')
        # self.graph.set_xticks([])
        # self.graph.set_yticks([])
        # self.graph.set_zticks([])
        
        self.cameraGraphCoordSys = self.drawCoordinateAxes('c', self.map.trajectory[0])
        o = self.map.trajectory[0].t
        self.cameraGraphPath, = self.graph.plot(o[0], o[1], o[2])
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

        if timeStep >= self.nFrames: return

        pathData = self.cameraGraphPath.get_data_3d()

        xPath = np.append(pathData[0], origo[0])
        yPath = np.append(pathData[1], origo[1])
        zPath = np.append(pathData[2], origo[2])

        data = np.vstack((xPath.T, yPath.T, zPath.T))
        self.cameraGraphPath.set_data(data[:2, :])
        self.cameraGraphPath.set_3d_properties(data[2, :])

        
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
            textargs = {'color': 'b', 'va': 'center', 'ha': 'center', 'fontsize': 'medium'}
            self.graph.text(x[0], x[1], x[2], 'X', **textargs)
            self.graph.text(y[0], y[1], y[2], 'Y', **textargs)
            self.graph.text(z[0], z[1], z[2], 'Z', **textargs)

        return xAxis, yAxis, zAxis, o
    

    def play(self):
        ani = animation.FuncAnimation(self.fig, self.animStep, frames=self.nFrames,
                                      interval=1000/self.FPS, blit=False)
        plt.show()