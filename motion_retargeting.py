import numpy as np
import numpy.linalg as LA
import math
from collections import deque


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
    pitch = -math.atan2(upperarm_t[1], upperarm_t[2])
    #
    # pitch = -math.atan2(-upperarm_t[1], -upperarm_t[0])
    roll = math.atan2(upperarm_t[0], upperarm_t[2])
    pitch = math.atan2(-upperarm_t[1],math.sqrt(upperarm_t[0]**2 + upperarm_t[2]**2))
    print(upperarm_t)
    return roll, pitch, upperarm_t


def filter_data(inputs:'deque', filter_func):
