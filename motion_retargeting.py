import numpy as np
import numpy.linalg as LA
import math
import matplotlib.pyplot as plt
from collections import deque
from common.visualization import Sequencial_animation, RealtimePlot


def compute_torso_coord(neck, lhip, rhip):
    bz = np.cross((rhip - neck), (lhip - neck))
    bz = bz/LA.norm(bz)
    bx = lhip - rhip
    bx = bx/LA.norm(bx)
    by = np.cross(bz, bx)
    by = by/LA.norm(by)
    A = np.stack((bx,by,bz),axis=0)

    return bx, by, bz, A


def compute_shoulder_rotation(upperarm, coord):
    bx, by, bz, A = coord
    upperarm_t = np.dot(A, upperarm)
    #
    plane = np.cross(by, bz)
    inner = np.inner(plane, upperarm_t) / (LA.norm(plane) * LA.norm(upperarm_t))
    roll = -(np.pi/2 - math.acos(inner))
    # pitch = -math.atan2(upperarm_t[1], upperarm_t[2])
    #
    # roll = math.atan2(upperarm_t[0], upperarm_t[2])
    pitch = math.atan2(-upperarm_t[1],math.sqrt(upperarm_t[0]**2 + upperarm_t[2]**2))
    print(upperarm_t)
    return roll, pitch, upperarm_t


def filter_data(Q:'deque', input, filter_func, **kw):
    # print("type: {}".format(type(Q)))
    if len(Q) < Q.maxlen:
        Q.append(input)
        return input

    else:
        temp = Q.copy()
        temp.append(input)
        new_input = filter_func(temp, **kw)
        # print(new_input)
        Q.append(input)
        return new_input


def median_filter(dq:'deque', **kw):
    from numpy import median as median
    # from scipy.signal import medfilt as median
    # from copy import deepcopy as deepcopy
    # new_dq = deepcopy(dq)
    return median(dq)

def amp_lmt_filter(dq:'deque', **kw):

    thresh = kw['thresh']

    from numpy import abs as abs
    if abs(dq[-1] - dq[-2]) > thresh:
        return dq[-2]
    else:
        return dq[-1]

if __name__ == "__main__":
    import time
    import math
    import numpy as np

    fig_angle = plt.figure()
    ax1 = fig_angle.add_subplot(111)
    angle_animation1 = RealtimePlot(fig_angle, ax1, "test","r")
    angle_animation2 = RealtimePlot(fig_angle, ax1, "test","g")

    frame = -1
    Q = deque(maxlen=5)
    while True:
        frame += 1
        data = np.random.randint(-10, 10) + 50*np.math.cos(frame/5)

        # new_data = filter_data(Q, data, amp_lmt_filter, thresh=20)
        new_data = filter_data(Q, data, amp_lmt_filter, thresh=5)
        angle_animation1.call(frame, data)
        angle_animation2.call(frame, new_data)
        
