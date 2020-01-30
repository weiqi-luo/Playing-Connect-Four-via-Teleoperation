import os
import sys
import time
import queue

from common.arguments import parse_args
from common.camera import *
from common.generators import UnchunkedGenerator
from common.loss import *
from common.model import *
from common.utils import Timer, evaluate, add_path
from collections import deque
import cv2
from common.utils import read_video
from motion_retargeting import *
import matplotlib.pyplot as plt
import click
# from joints_detectors.openpose.main import generate_kpts as open_pose
from pylab import get_current_fig_manager

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
metadata = {'layout_name': 'coco', 'num_joints': 17, 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]]}
add_path()


# record time
def ckpt_time(ckpt=None):
    if not ckpt:
        return time.time()
    else:
        return time.time() - float(ckpt), time.time()
time0 = ckpt_time()


class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 9, 10]


def main(args):
    #! UDP client
    import socket

    UDP_IP = "10.152.246.117"
    UDP_PORT = 9090
    sock = socket.socket(socket.AF_INET, # Internet
                        socket.SOCK_DGRAM) # UDP

    print ("UDP target IP:", UDP_IP)
    print ("UDP target port:", UDP_PORT)

    #! read image from camera   
    from joints_detectors.Alphapose.video2d import DetectionLoader
    det_loader = DetectionLoader(size=args.viz_size,device=0)
   

    #! visualization
    from common.visualization import Sequencial_animation, RealtimePlot
    pose3d_animation = Sequencial_animation( skeleton=Skeleton(), i=8,
        size=args.viz_size, azim=np.array(70., dtype=np.float32))
    
    fig_angle = plt.figure()
    ax1 = fig_angle.add_subplot(311)
    ax2 = fig_angle.add_subplot(312)
    ax3 = fig_angle.add_subplot(313)
    angle_animation1_n = RealtimePlot(fig_angle, ax1, "LShoulderRoll","r")
    angle_animation2_n = RealtimePlot(fig_angle, ax2, "LShoulderPitch","r")
    angle_animation1 = RealtimePlot(fig_angle, ax1, "LShoulderRoll","y")
    angle_animation2 = RealtimePlot(fig_angle, ax2, "LShoulderPitch","y")
    angle_animation_turningangle = RealtimePlot(fig_angle, ax3, "turning angle","y")
    angle_animation_turning = RealtimePlot(fig_angle, ax3, "turning","r")
    thismanager = get_current_fig_manager()
    thismanager.window.wm_geometry("+1000+0")
    
    ## TODO TESTING 
    # fig_angle = plt.figure()
    # ax1 = fig_angle.add_subplot(311)
    # ax2 = fig_angle.add_subplot(312)
    # ax3 = fig_angle.add_subplot(313)
    # arm_animation1 = RealtimePlot(fig_angle, ax1, "upperarm x","r", fixylim=False)
    # arm_animation2 = RealtimePlot(fig_angle, ax2, "upperarm y","g", fixylim=False)
    # arm_animation3 = RealtimePlot(fig_angle, ax3, "upperarm z","b", fixylim=False)
    thismanager = get_current_fig_manager()
    thismanager.window.wm_geometry("+1000+1000")

    plt.ion()   # continuously plot


    #! load 3d pose estimation model
    model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=args.causal, 
        dropout=args.dropout, channels=args.channels, dense=args.dense)
    if torch.cuda.is_available():
        model_pos = model_pos.cuda()

    ckpt, time1 = ckpt_time(time0)
    print('-------------- load 3d pose estimaion model spends {:.2f} seconds'.format(ckpt))


    #! load trained weights
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])
    #  Receptive field: 243 frames for args.arc [3, 3, 3, 3, 3]
    receptive_field = model_pos.receptive_field()
    pad = (receptive_field - 1) // 2  # Padding on each side
    causal_shift = 0

    ckpt, time2 = ckpt_time(time1)
    print('-------------- load trained weights for 3D model spends {:.2f} seconds'.format(ckpt))


    #! Initialize some kp
    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])
    rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
    kp_deque = deque(maxlen=9)
    


    #! loop through the frame (now fake frame)
    ckpt, time3 = ckpt_time(time2)

    #! Prepare deque for 8 joints
    q_LShoulderPitch, q_RShoulderPitch, q_LShoulderRoll, q_RShoulderRoll = [deque(maxlen=7) for i in range(4)]
    q_LElbowYaw, q_RElbowYaw, q_LElbowRoll, q_RElbowRoll = [deque(maxlen=7) for i in range(4)]

    try:
        while True:
            frame = -1
            while frame < 65501: # prevent from overspilling
                ckpt, time3 = ckpt_time(time3)
                frame += 1
                print("------------------- frame ",frame, '-------- generate reconstruction 3D data spends {:.2f} seconds'.format(ckpt))
                
                #! get 2d key points
                kp = det_loader.update()
                if kp is None:
                    continue

                kp_deque.append(kp.numpy())    
                if len(kp_deque)<9:
                    continue
                

                #! estimate 3d pose 
                # normlization keypoints  Suppose using the camera parameter
                input_keypoints = normalize_screen_coordinates(np.asarray(kp_deque)[..., :2], w=1000, h=1002)
                prediction = evaluate(input_keypoints, pad, model_pos, return_predictions=True)
                
                # rotate the camera perspective
                prediction = camera_to_world(prediction, R=rot, t=0)

                # We don't have the trajectory, but at least we can rebase the height
                prediction[:, :, 2] -= np.min(prediction[:, :, 2])
                # input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)


                #! Visualize 3d
                prediction = prediction[8]
                pose3d_animation.call(prediction)

                #! Motion retargeting
                LShoulder, LElbow, RShoulder, RElbow, Neck, Waist, LWrist, RWrist = \
                    prediction[11], prediction[12], prediction[14], prediction[15], prediction[8], prediction[7], prediction[13], prediction[16]
                LHip, RHip = prediction[4], prediction[1]
                
                left_upperarm = LElbow - LShoulder
                right_upperarm = RElbow - RShoulder
                left_lowerarm = LElbow - LWrist
                right_lowerarm = RElbow - RWrist
                
                #### Remapping ####
                coord = compute_torso_coord(Neck, LHip, RHip)
                LShoulderRoll, LShoulderPitch, left_upperarm_t = compute_shoulder_rotation(left_upperarm, coord)
                RShoulderRoll, RShoulderPitch, right_upperarm_t = compute_shoulder_rotation(right_upperarm, coord)

                LElbowYaw, LElbowRoll, left_upperarm_t, left_lowerarm_t = compute_elbow_rotation(left_upperarm, left_lowerarm, coord)
                RElbowYaw, RElbowRoll, right_upperarm_t, right_lowerarm_t = compute_elbow_rotation(right_upperarm, right_lowerarm, coord)

                #### Filtering ####
                LShoulderPitch_n = filter_data(q_LShoulderPitch, LShoulderPitch, median_filter)
                LShoulderRoll_n = filter_data(q_LShoulderRoll,  LShoulderRoll, median_filter)
                RShoulderPitch_n = filter_data(q_RShoulderPitch, RShoulderPitch, median_filter)
                RShoulderRoll_n = filter_data(q_RShoulderRoll,  RShoulderRoll, median_filter)
                
                LElbowYaw_n = filter_data(q_LElbowYaw, LElbowYaw, median_filter)
                LElbowRoll_n = filter_data(q_LElbowRoll,  LElbowRoll, median_filter)
                RElbowYaw_n = filter_data(q_RElbowYaw, RElbowYaw, median_filter)
                RElbowRoll_n = filter_data(q_RElbowRoll,  RElbowRoll, median_filter)

                Turning, turningangle = compute_turning(coord[2])
                # msg_LShoulderPitch = if q_LShoulderPitch.full()


                
                #! Plot angle
                print("ShoulderRoll_n {} ShoulderPitch_n {} ".format(math.degrees(LElbowYaw_n), math.degrees(LElbowRoll_n)))

                angle_animation1_n.call(frame,math.degrees(LElbowRoll_n))
                angle_animation2_n.call(frame,math.degrees(LElbowYaw_n))
                angle_animation1.call(frame,math.degrees(LElbowRoll))
                angle_animation2.call(frame,math.degrees(LElbowYaw))

                angle_animation_turningangle.call(frame,turningangle)
                angle_animation_turning.call(frame, 180*Turning)

                # angle_animation1_n.call(frame, LShoulderRoll_n)
                # angle_animation2_n.call(frame, LShoulderPitch_n)
                # angle_animation1.call(frame,   LShoulderRoll)
                # angle_animation2.call(frame,   LShoulderPitch)
                
                # arm_animation1.call(frame,lupperarm_t[0])
                # arm_animation2.call(frame,lupperarm_t[1])
                # arm_animation3.call(frame,lupperarm_t[2])


                #! send udp
                message = np.array((frame,
                                    LShoulderRoll_n, LShoulderPitch_n, RShoulderRoll_n, RShoulderPitch_n, 
                                    LElbowYaw_n, LElbowRoll_n, LElbowYaw_n, LElbowRoll_n, 
                                    Turning))
                MESSAGE = message.astype(np.float16).tostring()
                sock.sendto(MESSAGE, (UDP_IP, UDP_PORT))
                # print(sys.getsizeof(MESSAGE))
                # print(message)

    except KeyboardInterrupt:
        plt.ioff()
        cv2.destroyAllWindows()
        return



if __name__ == '__main__':
    args = parse_args()
    main(args)
