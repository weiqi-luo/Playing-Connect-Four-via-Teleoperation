import os
import time


from common.arguments import parse_args
from common.camera import *
from common.generators import UnchunkedGenerator
from common.loss import *
from common.model import *
from common.utils import Timer, evaluate, add_path
from collections import deque
import cv2
from common.utils import read_video
import matplotlib.pyplot as plt
# from joints_detectors.openpose.main import generate_kpts as open_pose

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

    #! read image from camera   
    from joints_detectors.Alphapose.video2d import DetectionLoader
    det_loader = DetectionLoader(size=args.viz_size)
   

    #! visualization
    from common.visualization import Sequencial_animation
    sequencial_animation = Sequencial_animation( skeleton=Skeleton(), i=8,
        size=args.viz_size, azim=np.array(70., dtype=np.float32), limit=args.viz_limit, fps=25)
    plt.ion()   # continuously plot


    #! load 3d pose estimation model
    model_pos = TemporalModel(17, 2, 17, filter_widths=[3, 3, 3, 3, 3], causal=args.causal, dropout=args.dropout, channels=args.channels,
                            dense=args.dense)
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
    count = 0


    #! loop through the frame (now fake frame)
    try:
        ckpt, time3 = ckpt_time(time2)
        while True:
            ckpt, time3 = ckpt_time(time3)
            print("-------- frame ",count, '-------- generate reconstruction 3D data spends {:.2f} seconds'.format(ckpt))
            count += 1
            
            #! get 2d key points
            kp = det_loader.update()

            if isinstance(kp,int): 
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


            #! Visualization
            sequencial_animation.call(prediction) 


    except KeyboardInterrupt:
        plt.ioff()
        cv2.destroyAllWindows()
        return


def inference_camera():
    args = parse_args()
    main(args)


if __name__ == '__main__':
    inference_camera()
