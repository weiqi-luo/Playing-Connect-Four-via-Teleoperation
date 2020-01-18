import numpy as np
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D             # 3D plot
from matplotlib.animation import FuncAnimation      # animation
from tfaxis import TfAxis
from quaternion.quaternion_time_series import slerp

import time
import cv2
from matplotlib.animation import FuncAnimation, writers
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

from common.utils import read_video


def euler2quat(euler):
    """ Convert Euler Angles to Quaternions. (XYZ convention)"""
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quaternion.from_float_array(quat)


def random_pose():
    """
    Create a random position and quaternion
    @returns    np.array, np.quaternion
    """
    pos = 2 * (np.random.random(3) - 0.5)
    ang = 6 * (np.random.random(3) - 0.5)
    quat = euler2quat(ang)
    return pos, quat


def init_figure():
    """Init an axis 3D plot"""
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d([-1.0, 1.0])
    ax.set_ylim3d([-1.0, 1.0])
    ax.set_zlim3d([-1.0, 1.0])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('TfAxis Slerp Interpoation')
    return fig, ax


class TfAnimation:
    def __init__(self, ax, frames: int):
        """
        @param ax       mpl_toolkits.mplot3d.Axis3D object
        @param frames   int, number of animation frames
        """

        self.start_tf = TfAxis(ax=ax, color=TfAxis.GRAY_COLOR)
        self.goal_tf = TfAxis(ax=ax)
        self.tf = TfAxis(ax=ax)
        self.frames = frames

    def run(self, fig, interval: int = 40):
        """
        Run the animation.
        @param fig      plt.figure() object 
        @param interval time between two frames in milli seconds
        """
        ani = FuncAnimation(fig, self.func, frames=self.frames,
                            init_func=self.init_func, interval=interval)
        plt.show()

    def init_func(self):
        """Init function that is called at the begin of each animation"""
        goal_pos = random_pose()
        self.goal_tf.set_pose(*goal_pos)
        self.tf.set_pose([0, 0, 0])

        # interpolate between pose and quaternions
        q1 = np.quaternion(1)
        q2 = self.goal_tf.quat
        self.dx = np.linspace([0, 0, 0], goal_pos[0], self.frames)
        self.dq = slerp(q1, q2, 0, 1, np.linspace(0, 1, self.frames))

    def func(self, frame):
        """Update function for the animation"""
        self.tf.set_pose(self.dx[frame], self.dq[frame])


fig, ax = init_figure()
tf_animation = TfAnimation(ax, frames=50)
tf_animation.run(fig, interval=40)
