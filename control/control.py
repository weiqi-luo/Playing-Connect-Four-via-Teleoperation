import numpy as np

from tfaxis import *
from scipy.spatial.transform import Rotation as R

def main():
    pose3d = np.load("../outputs/3dpose.npy")
    # tf viewer
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d([-2.0, 2.0])
    ax.set_ylim3d([-2.0, 2.0])
    ax.set_zlim3d([-2.0, 2.0])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('TfAxis Slerp Interpoation')
    tf1 = TfAxis(origin=matMatch[:3,3], quat=quatMatch, scale=0.4)
    tf2 = TfAxis(origin=matTrue[:3,3], quat=quatTrue, scale=0.4)
    tf1.plot(ax)
    tf2.plot(ax)
    plt.show()
    if key==27:    # Esc key to stop
        return

if __name__ == "__main__":
    main()