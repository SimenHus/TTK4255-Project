import numpy as np

from dataclasses import dataclass

@dataclass
class Pose:
    R: 'np.array[3, 3]' = np.eye(3)
    t: 'np.array[3]' = np.zeros((3,))

    def __post_init__(self):
        if self.R.shape[0] > 3:
            self.T = self.R
            self.R = self.T[:3, :3]
            self.t = self.T[:3, 3]
        else:
            self.T = np.zeros((4, 4))
            self.T[:3, :3] = self.R
            self.T[:3, 3] = self.t
            self.T[3, 3] = 1

    @property
    def inv(self):
        RT = self.R.T
        T = Pose(RT, -RT@self.t) # Faster processing than matrix inversion
        return T

    @property
    def pos(self):
        return self.t
    
    @property
    def rot(self):
        return self.R

    def __repr__(self):
        return repr(self.T)

    def __matmul__(self, other):
        return Pose(self.T@other.T)