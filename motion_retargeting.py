import numpy as np
import numpy.linalg as LA
import math
import matplotlib.pyplot as plt
from collections import deque
from common.visualization import Sequencial_animation, RealtimePlot



def get_squatting(left_upperleg, left_lowerleg, right_upperleg, right_lowerleg):

    inner_left = np.inner(left_upperleg, left_lowerleg) / (LA.norm(left_upperleg) * LA.norm(left_lowerleg))
    inner_right = np.inner(right_upperleg, right_lowerleg) / (LA.norm(right_upperleg) * LA.norm(right_lowerleg))

    squattingangle_left = math.degrees(math.acos(inner_left))
    squattingangle_right = math.degrees(math.acos(inner_right))

    squatting = 1 if squattingangle_left<100 and squattingangle_right<100 else 0

    return squatting, squattingangle_left, squattingangle_right


def get_turning(vec):
    """
    0: do nothing or stop  1: turn right  2: turn left
    """
    angle = math.degrees(math.atan2(vec[0],vec[1]))
    if abs(angle)<30: # stop or do nothing
        turning = 0

    else:
        turning = 1 if angle>0 else 2
        
    return turning,angle


def get_moving(top):
    """
    0: do nothing or stop  1: go right 2: go left
    """
    if abs(top)<0.22:
        moving = 0
    elif top>0:
        moving = 1 #right
    else:
        moving = 2 #left        
    return moving, top


class Compute_moving:
    def __init__(self):
        self.old_leftmoving = False # 3
        self.old_rightmoving = False # 1

    def __call__(self,top):
        if abs(top)<0.22:
            if self.old_leftmoving or self.old_rightmoving:
                moving = 2
                self.old_leftmoving = False
                self.old_rightmoving = False
            else: 
                moving = 0
        elif top>0:
            if self.old_rightmoving:
                moving = 0
            else:
                moving = 1 #right
                self.old_rightmoving = True
        else:
            if self.old_leftmoving:
                moving = 0
            else:
                moving = 3 #left
                self.old_leftmoving = True
            
        return moving, top



class Compute_walking:
    def __init__(self):
        self.old_walkingangle = None
    
    def __call__(self, walkingangle):
        if self.old_walkingangle is None or walkingangle<0.04:
            walking = 0
        else:
            walking = 1 if walkingangle*self.old_walkingangle < 0 else 0
        self.old_walkingangle = walkingangle
        return walking, walkingangle


class Compute_squatting:
    def __init__(self):
        self.old_squattingangle_left = False
        self.old_squattingangle_right = False


    def __call__(self, upperleg, lowerleg):
        left_upperleg, right_upperleg = upperleg
        left_lowerleg, right_lowerleg = lowerleg

        inner_left = np.inner(left_upperleg, left_lowerleg) / (LA.norm(left_upperleg) * LA.norm(left_lowerleg))
        inner_right = np.inner(right_upperleg, right_lowerleg) / (LA.norm(right_upperleg) * LA.norm(right_lowerleg))

        squattingangle_left = math.degrees(math.acos(inner_left))
        squattingangle_right = math.degrees(math.acos(inner_right))

        squatting = 1 if squattingangle_left<100 and squattingangle_right<100 else 0

        return squatting, squattingangle_left, squattingangle_right




class Compute_turning:
    def __init__(self):
        self.left_turning = False # 3
        self.right_turning = False # 1


    def __call__(self, vec):
        """
        0: do nothing  1: turn right  2: stop  3: turn left 
        """
        angle = math.degrees(math.atan2(vec[0],vec[1]))
        if abs(angle)<30: # stop or do nothing
            if self.left_turning or self.right_turning:
                turning = 2
                self.left_turning = False
                self.right_turning = False
            else: 
                turning = 0  

        elif angle>0: # turn right or do nothing
            turning = 0 if self.right_turning else 1 
            self.right_turning = True

        else: # turn left
            turning = 0 if self.left_turning else 3 
            self.left_turning = True
            
        return turning,angle



def compute_torso_coord(top, lhip, rhip):
    bz = np.cross((rhip - top), (lhip - top))
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
    # roll = math.atan2(upperarm_t[0], upperarm_t[2])
    pitch = math.atan2(-upperarm_t[1],math.sqrt(upperarm_t[0]**2 + upperarm_t[2]**2))
    return roll, pitch, upperarm_t

def compute_elbow_rotation(upperarm, lowerarm, coord, neg):
    #! upperarm is from shoulder to elbow
    #! lowerarm is from elbow to wrests
    bx, by, bz, A = coord
    upperarm_t = np.dot(A, upperarm)
    lowerarm_t = np.dot(A, lowerarm)

    inner = np.inner(upperarm_t, lowerarm_t) / (LA.norm(upperarm_t) * LA.norm(lowerarm_t))
    # roll = -(np.pi/2 - math.acos(inner))
    roll = neg*math.acos(inner)

    m1 = np.cross(-bx, upperarm_t)
    m2 = np.cross(-upperarm_t, lowerarm_t)
    yaw = math.acos(np.dot(m1, m2) / (LA.norm(m1) * LA.norm(m2))) - math.pi/2

    return yaw, roll, upperarm_t, lowerarm_t


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
        
