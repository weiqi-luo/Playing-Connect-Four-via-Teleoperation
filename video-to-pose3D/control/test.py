import numpy as np
from tfaxis import *
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1, projection='3d')
    # V = np.array([[1,1],[-2,2],[4,-7]])
    # origin = [0], [0] # origin point

    # # plt.quiver(*origin, V[:,0], V[:,1], color=['r','b','g'], scale=21)
    # plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy.linalg as LA
from operator import sub

def draw_vector(ax,p1,p2,absolute=True):
    print(p1,p2)
    p1 = list(p1)
    p2 = list(p2)
    if absolute:
        p_delta = list(map(sub,p2,p1))
        p = p1+p_delta
    else:
        p = p1+p2
        print(p)
    ax.quiver(*p,label=str(p),color=np.random.rand(3,))


def main():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])
    ax.set_zlim([-4,4])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    p0 = np.array([0,0,0])
    p1 = np.array([1,1,1])
    p2 = np.array([1,2,3])
    p3 = np.array([0,1,4])
    for p in [p1,p2,p3]:
        draw_vector(ax,p0,p,True)

    p12 = np.cross(p1,p2)
    p12 = p12/LA.norm(p12)
    print(p12)
    draw_vector(ax,p0,p12,False)

    print(np.inner(p12,p1),np.inner(p12,p2))

    p3_12 = np.inner(p3,p12)
    p3_12 = p3_12*p12
    p3_no12 = p3 - p3_12
    
    for p in [p3_12,p3_no12]:
        draw_vector(ax,p0,p,True)

    print(np.inner(p3_no12,p12),np.inner(p3_no12,p12))
    ax.legend()
    plt.show()

if __name__ == "__main__":
    main()