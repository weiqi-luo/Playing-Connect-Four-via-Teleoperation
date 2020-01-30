#!/usr/bin/env python
import numpy as np
import numpy.linalg as LA

import rospy
import time
import almath
import sys,math
import socket
import sys

from naoqi import ALProxy
from nao_control_tutorial_1.srv import MoveJoints, MoveJointsResponse

from sensor_msgs.msg import JointState 
from geometry_msgs.msg import TransformStamped
import tf, tf2_ros
motionProxy =0
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from operator import sub


global qvx
global qvy
global qvz
global lg
################################### Callback functions ##########################################
################################################################################################

def callback_jointStates(data):
    if not list_jointState[0]:
        list_jointState[0] = data
        rospy.logwarn("Robot state is initialized")


################################### Send move control #########################################
################################################################################################
class Move:
    def __init__(self, robotIP, PORT):
        self.robotIP = robotIP
        self.PORT = PORT
        self.motionProxy = ALProxy("ALMotion", self.robotIP, self.PORT)

    def move_joints_callback(self):
        self.motionProxy = ALProxy("ALMotion", self.robotIP, self.PORT)
        self.motionProxy.setAngles('RHand',0.0,1.0)


def broadcast_static_tf():
    static_br = tf2_ros.StaticTransformBroadcaster()
    msg_staticTransform = TransformStamped()
    msg_staticTransform.header.stamp = rospy.Time.now()
    msg_staticTransform.header.frame_id = "base_link"
    msg_staticTransform.child_frame_id = "imitate/base_link"
    msg_staticTransform.transform.translation.x = 0.0
    msg_staticTransform.transform.translation.y = 0.0
    msg_staticTransform.transform.translation.z = 0.0
    quat = tf.transformations.quaternion_from_euler(0.0, 0.0, 0.0)
    msg_staticTransform.transform.rotation.x = quat[0]
    msg_staticTransform.transform.rotation.y = quat[1]
    msg_staticTransform.transform.rotation.z = quat[2]
    msg_staticTransform.transform.rotation.w = quat[3]
    static_br.sendTransform(msg_staticTransform)


################################# Pose mapping #########################################
################################################################################################

def draw_vector(ax,p1,p2,absolute=True,color=np.random.rand(3,)):
    print(p1,p2)
    p1 = list(p1)
    p2 = list(p2)
    if absolute:
        p_delta = list(map(sub,p2,p1))
        p = p1+p_delta
    else:
        p = p1+p2
        print(p)
    qv = ax.quiver(*p,label=str(p),color=color)


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


def compute_torso_coord(neck, lhip, rhip):
    bx = np.cross((rhip - neck), (lhip - neck))
    bx = bx/LA.norm(bx)

    by = lhip - rhip
    by = by/LA.norm(by)

    bz = np.cross(bx, by)
    bz = bz/LA.norm(bz)

    return bx, by, bz


def compute_shoulder_roll(upperarm, coord):
    bx, by, bz = coord
    plane = np.cross(bx, bz)
    inner = np.inner(plane, upperarm) / (LA.norm(plane) * LA.norm(upperarm))
    roll = -(np.pi/2 - math.acos(inner))
    pitch = -math.atan2(upperarm[2], upperarm[0])
    return roll, pitch


def compute_elbow(upperarm, lowerarm, shoulder_roll, shoulder_pitch, coord):
    bx, by, bz = coord
    roll = -math.acos(np.dot(upperarm, lowerarm)/(LA.norm(upperarm)*LA.norm(lowerarm)))

    m1 = np.array([[math.cos(shoulder_roll), -math.sin(shoulder_pitch, 0)],
                   [math.sin(shoulder_roll), math.cos(shoulder_roll), 0],
                   [0, 0, 1]])
    m2 = np.array([[math.cos(shoulder_pitch), 0, math.sin(shoulder_pitch)],
                   [0, 1, 0],
                   [-math.sin(shoulder_pitch), 0, math.cos(shoulder_pitch)]])
    
    b = np.dot(np.dot(m1, m2), bx)

    a_left = np.cross(upperarm, lowerarm)
    a_left = LA.norm(a_left)

    yaw = -(np.pi/2 - math.acos(np.dot(b, a_left)/(LA.norm(b)*LA.norm(a_left))))

    return roll, yaw

def decompose_shoulder_rotation(body, shoulder, upperarm, ax, coord):
    """
    input: 
        body, shoulder, upperarm
    create_orthogonal_coord:
        bx = forwardbody = v1 x v2
        by = leftshoulder-rightshoulder = corrected v2
        bz = neck-waist = v1
    """
    # create_orthogonal_coord:
    # bz = body/LA.norm(body)
    # bx = np.cross(shoulder,body)
    # bx = bx/LA.norm(bx)
    # by = np.cross(bz,bx)
    # by = by/LA.norm(by)
    # # decompose_vector

    bx, by, bz = coord

    vx = np.inner(upperarm,bx)
    vy = np.inner(upperarm,by)
    vz = np.inner(upperarm,bz)
    # rospy.loginfo("{}, {}, {}".format(vx,vy,vz))
    # roll = math.atan2(vy, vx)
    # pitch = -math.atan2(vz, math.sqrt(vx**2+vy**2))
    # roll = math.atan2(vz, math.sqrt(vx**2+vy**2))
    roll = math.atan2(vy,vx)
    pitch = -math.atan2(vz,vx)

    # TODO TESTING
    plt.cla()
    origin = (0,0,0)
    lim = 2
    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])
    ax.set_zlim([-lim,lim])
    draw_vector(ax,origin,shoulder,True,(0.5,0,0))
    draw_vector(ax,origin,body,True,(0,0,0.5))
    draw_vector(ax,origin,bx,True,(1,0,0))
    draw_vector(ax,origin,by,True,(0,1,0))
    draw_vector(ax,origin,bz,True,(0,0,1))
    draw_vector(ax,origin,10*upperarm,True,(0,0,0))
    ax.legend()
    plt.draw()
    plt.pause(0.0000001)
    roll, pitch = 0, 0
    return roll, pitch


def decompose_elbow_rotation(upperarm, shoulder, lowerarm, direc_roll, ax):
    """
    Decompose the lowerarm into two vectors, one aligned with upperarm, another orthogonal to upperarm.
    """
    # _, current_inplane = decompose_vector(upperarm, shoulder)
    # _, lowerarm_inplane = decompose_vector(upperarm, lowerarm)
    # print(shoulder_inplane,lowerarm_inplane)
    # yaw = rotation_between(shoulder_inplane,lowerarm_inplane, upperarm)
    # roll = rotation_between(upperarm, lowerarm, direc_roll)
    roll = 0
    yaw = 0
    return roll, yaw


def decompose_vector(b_otho, v):
    """ 
    decompose the vector v3 into two vectors, one is perpendicular to plane composed 
    by v1 and v2, while the other one is in the plane 
    """
    b_otho = b_otho / LA.norm(b_otho)
    v_otho = np.inner(b_otho, v)
    v_otho = v_otho * b_otho
    v_inplane = v - v_otho
    return v_otho, v_inplane


def rotation_between(v1,v2,vref):
    """
    Compute the rotation from v1 to v2.
    If the rotation axis is aligned with vref, return positive angle,
    otherwise return negative angle
    """
    vaxis = np.cross(v1,v2)
    cosang = np.dot(v1,v2)
    sinang = LA.norm(vaxis)
    angle = np.arctan2(sinang,cosang)
    if isinstance(vref,(int,float,long)):
        sign = vref
    else:
        sign = math.copysign(1,np.inner(vaxis,vref))
    return sign*angle

################################### Main function ##############################################
################################################################################################

if __name__ == '__main__':
    robotIP=str(sys.argv[1])
    PORT=int(sys.argv[2])
    print sys.argv[2]
    rospy.init_node('move_joints_server')
    plt.ion()

    #! set up UDP connection
    UDP_IP = "10.152.246.117"
    UDP_PORT = 9090
    sock = socket.socket(socket.AF_INET, # Internet
                        socket.SOCK_DGRAM) # UDP
    # sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256*8)
    sock.bind((UDP_IP, UDP_PORT))


    #TODO TESTING
    axl = init_plot3d()

    rospy.Subscriber("/joint_states", JointState, callback_jointStates)
    pub_jointStates = rospy.Publisher('/imitate/joint_states', JointState, queue_size=10)
    move = Move(robotIP, PORT)

    list_jointState = [None]
    while not list_jointState[0]:
        rospy.logwarn("Waiting for robot state being initialized")
        time.sleep(3) 

    #TODO TESTING
    # pose3d = np.load("/home/hrs/ros/hrs_ws/src/final_project/data/3dpose.npy")
    # rospy.loginfo(pose3d.shape)

    #! broadcast static tf
    broadcast_static_tf()
    
    #! broadcast 3d pose to tf 
    br = tf.TransformBroadcaster()
    jointname =  ["MHip","RHip","RKnee","RAnkle","LHip","LKnee","LAnkle","Waist","Neck",
    "Face","Top","LShoulder","LElbow","LWrist","RShoulder","RElbow","RWrist"]       
    while True:
        #! receive udp packets
        # pose, __ = sock.recvfrom(256) # buffer size is 1024 bytes
        # pose = np.fromstring(pose, dtype = np.float16)
        # rospy.logwarn("------------ frame {} ------------ ".format(pose[0]))
        pose = np.resize(pose[1:],((17, 3)))
        # rospy.loginfo(pose)
        if rospy.is_shutdown():
            break

        # TODO TESTING
        # count = 0
        # for point in pose:
        #     # print(point)
        #     br.sendTransform(point,
        #                     tf.transformations.quaternion_from_euler(0, 0, 0),
        #                     rospy.Time.now(),
        #                     "hm_"+jointname[count],
        #                     "odom")
        #     count += 1

        #! get joint angles
        """
        Top:10,         Face:9,      Neck:8         Waist: 7    MHip: 0 
        LShoulder: 11,  LElbow: 12,  LWrist: 13,    LHip: 4,    LKnee: 5,    LAnkle: 6
        RShoulder: 14,  RElbow: 15,  RWrist: 16,    RHip: 1,    RKnee: 2,    RAnkle: 3
        """
        LShoulder, LElbow, RShoulder, RElbow, Neck, Waist, LWrist, RWrist = \
            pose[11], pose[12], pose[14], pose[15], pose[8], pose[7], pose[13], pose[16]
        body = Neck - Waist
        shoulder = LShoulder - RShoulder
        lshoulder = Neck - LShoulder
        rshoulder = Neck - RShoulder
        larm_upper = LElbow - LShoulder
        rarm_upper = RElbow - RShoulder
        larm_lower = LWrist - LElbow 
        rarm_lower = RWrist - RElbow 
        LHip, RHip = pose[4], pose[1]
        #TODO Testing
        # larm_upper = np.array((-1,0,0))
        # lshoulder = np.array((0,0,1))
        # larm_lower = np.array((1,1,0)) 
        # Compute the angle of each joints
        # body = np.array((0, 0, 1))
        # shoulder = np.array((0, 1, 0))
        # larm_upper = np.array((1, 0, -1))
        # rarm_upper = np.array((1, 0, -1))
        print("body = {}, shoulder = {}, larm_upper = {}".format(body, shoulder, larm_upper))

        ##############################################
        coord = compute_torso_coord(Neck, LHip, RHip)
        LShoulderRoll, LShoulderPitch = compute_shoulder_roll_left(larm_upper, coord)
        RShoulderRoll, RShoulderPitch = compute_shoulder_roll_left(rarm_upper, coord)
        LElbowRoll, LElbowYaw = compute_elbow(larm_upper, larm_lower, LShoulderRoll, LShoulderPitch, coord)
        RElbowRoll, RElbowYaw = compute_elbow(rarm_upper, rarm_lower, RShoulderRoll, RShoulderPitch, coord)
        ##############################################

        ## TODO OLD
        # LShoulderRoll, LShoulderPitch = decompose_shoulder_rotation(body, shoulder, larm_upper, axl, coord)
        # RShoulderRoll, RShoulderPitch = decompose_shoulder_rotation(body, shoulder, rarm_upper, axr)
        # RShoulderRoll = compute_shoulder_roll_left(RShoulder, RElbow, coord)
        # RShoulderPitch = compute_shoulder_pitch_left(RShoulder, RElbow,coord)
  
        #! send command to nao in order to imitate received pose
        """
        0'HeadYaw',             1'HeadPitch',        
        2'LShoulderPitch',      3'LShoulderRoll',       4'LElbowYaw',       5'LElbowRoll',          6'LWristYaw',           7'LHand', 
        8'LHipYawPitch',        9'LHipRoll',            10'LHipPitch',      11'LKneePitch',         12'LAnklePitch',        13'LAnkleRoll', 
        14'RHipYawPitch',       15'RHipRoll',           16'RHipPitch',      17'RKneePitch',         18'RAnklePitch',        19'RAnkleRoll', 
        20'RShoulderPitch',     21'RShoulderRoll',      22'RElbowYaw',      23'RElbowRoll',         24'RWristYaw',          26'RHand']
        """
        msg_jointState = list_jointState[0]

        # TODO testing
        LShoulderRoll = 0
        LShoulderPitch = 0
        RShoulderRoll = 0
        RShoulderPitch = 0
        LElbowRoll = 0
        LElbowYaw = 0
        RElbowRoll = 0
        RElbowYaw = 0

        position = list(msg_jointState.position)
        position[3] = LShoulderRoll 
        position[2] = LShoulderPitch
        position[21] = RShoulderRoll 
        position[20] = RShoulderPitch
        position[5] = LElbowRoll
        position[4] = LElbowYaw
        position[23] = RElbowRoll
        position[22] = RElbowYaw
        msg_jointState.position = position
        msg_jointState.header.stamp = rospy.Time.now()
        rospy.loginfo("LShoulderRoll={}, LShoulderPitch={}, RShoulderRoll={}, RShoulderPitch={}".format(
            math.degrees(LShoulderRoll),math.degrees(LShoulderPitch),math.degrees(RShoulderRoll),math.degrees(RShoulderPitch)))
        rospy.loginfo("LElbowRoll={}, LElbowYaw={}, RElbowRoll={}, RElbowYaw={}".format(
            math.degrees(LElbowRoll),math.degrees(LElbowYaw),math.degrees(RElbowRoll),math.degrees(RElbowYaw)))

        pub_jointStates.publish(msg_jointState)

        # move.move_joints_callback()
    rospy.spin()
