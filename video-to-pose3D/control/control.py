#!/usr/bin/env python
import numpy as np
import numpy.linalg as LA

import time
import sys,math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



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


def init_plot3d():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-4,4])
    ax.set_ylim([-4,4])
    ax.set_zlim([-4,4])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return ax


def decompose_shoulder_rotation(v1, v2, v, ax):
    """
    input: 
        body, shoulder, arm
    create_orthogonal_coord:
        bx = forwardbody = v1 x v2
        by = leftshoulder-rightshoulder = corrected v2
        bz = neck-waist = v1
    decompose_vector:
    """
    # create_orthogonal_coord:
    bz = v1
    bx = np.cross(v2,v1)
    bx = bx/LA.norm(bx)
    by = np.cross(bz,bx)
    by = by/LA.norm(by)
    # decompose_vector
    vx = np.inner(v,bx)
    vy = np.inner(v,by)
    vz = np.inner(v,bz)
    # rospy.loginfo("{}, {}, {}".format(vx,vy,vz))
    roll = math.atan2(vy, vx)
    pitch = -math.atan2(vz, math.sqrt(vx**2+vy**2))

    origin = (0,0,0)
    for v in (v1,v2,bx,by,bz):
        draw_vector(origin,v,True)
    plt.show

    return roll, pitch



if __name__ == '__main__':

    ax = init_plot3d()

    #! get 3d pose 
    pose3d = np.load("../output/3dpose.npy")

    #! broadcast 3d pose to tf 
    seq = 0
    for pose in pose3d:
        #! get joint angles
        """
        Top:10,         Face:9,      Neck:8         Waist: 7    MHip: 0 
        LShoulder: 11,  LElbow: 12,  LWrist: 13,    LHip: 4,    LKnee: 5,    LAnkle: 6
        RShoulder: 14,  RElbow: 15,  RWrist: 16,    RHip: 1,    RKnee: 2,    RAnkle: 3
        """
        LShoulder, LElbow, RShoulder, RElbow, Neck, Waist, LHip, RHip = pose[11], pose[12], pose[14], pose[15], pose[8], pose[7], pose[4], pose[1]
        body = Neck - Waist
        shoulder = LShoulder - RShoulder
        larm = LElbow - LShoulder
        rarm = RElbow - RShoulder

        # Compute the angle of each joints
        LShoulderRoll, LShoulderPitch = decompose_shoulder_rotation(body, shoulder, larm, ax)
        RShoulderRoll, RShoulderPitch = decompose_shoulder_rotation(body, shoulder, rarm, ax)

    
        #! send command to nao in order to imitate received pose
        """
        0'HeadYaw',             1'HeadPitch',        
        2'LShoulderPitch',      3'LShoulderRoll',       4'LElbowYaw',       5'LElbowRoll',          6'LWristYaw',           7'LHand', 
        8'LHipYawPitch',        9'LHipRoll',            10'LHipPitch',      11'LKneePitch',         12'LAnklePitch',        13'LAnkleRoll', 
        14'RHipYawPitch',       15'RHipRoll',           16'RHipPitch',      17'RKneePitch',         18'RAnklePitch',        19'RAnkleRoll', 
        20'RShoulderPitch',     21'RShoulderRoll',      22'RElbowYaw',      23'RElbowRoll',         24'RWristYaw',          26'RHand']
        """

        seq += 1 

        time.sleep(5)
