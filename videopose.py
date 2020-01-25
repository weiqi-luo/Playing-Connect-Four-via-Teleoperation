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


def get_detector_2d(detector_name):
    def get_alpha_pose():
        from joints_detectors.Alphapose.gene_npz import generate_kpts as alpha_pose
        return alpha_pose

    def get_hr_pose():
        from joints_detectors.hrnet.pose_estimation.video import generate_kpts as hr_pose
        return hr_pose

    detector_map = {
        'alpha_pose': get_alpha_pose,
        'hr_pose': get_hr_pose,
        # 'open_pose': open_pose
    }
    # assert detector_name in detector_map, f'2D detector: {detector_name} not implemented yet!'
    return detector_map[detector_name]()


def decode_video(downsample=1,input_video_path=None,input_video_skip=0,limit=0):
    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        all_frames = []
        for f in read_video(input_video_path, fps=None, skip=input_video_skip):
            all_frames.append(f)
        # effective_length = min(keypoints.shape[0], len(all_frames))
        # all_frames = all_frames[:effective_length]
    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))
    return all_frames


class Skeleton:
    def parents(self):
        return np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])

    def joints_right(self):
        return [1, 2, 3, 9, 10]


# ==================================================================================
def main_cam(args):
    from joints_detectors.Alphapose.gene_npz import generate_kpts_online
    generate_kpts_online()





def main(args):

    #! read image from camera
    # cap = cv2.VideoCapture(0)
    # while(True):
    #     ret, frame = cap.read()
    #     cv2.imshow('frame',frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    
    # fake camera
    # all_frames = decode_video(downsample=args.viz_downsample, input_video_path=args.viz_video)

    from joints_detectors.Alphapose.gene_npz import handle_camera
    generator = handle_camera()

    #! 2D kpts loads or generate
    # detector_2d = get_detector_2d(args.detector_2d)
    # assert detector_2d, 'detector_2d should be in ({alpha, hr, open}_pose)'
    # # if not args.input_npz:
    # #     video_name = args.viz_video
    # #     keypoints = detector_2d(video_name)
    # # else:
    # npz = np.load("./outputs/alpha_pose_kunkun_short/kunkun_short.npz")
    # count = -1
    # keypoints = npz['kpts']  # (N, 17, 2)

    

    #! visualization
    from common.visualization import Sequencial_animation
    sequencial_animation = Sequencial_animation( skeleton=Skeleton(), i=8,
        size=args.viz_size, azim=np.array(70., dtype=np.float32), limit=args.viz_limit, fps=25) #todo
    plt.ion()   # continuously plot #TODO


    #! Initialize some kp
    keypoints_symmetry = metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list([4, 5, 6, 11, 12, 13]), list([1, 2, 3, 14, 15, 16])


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


    #! loop through the frame (now fake frame)
    # for kp in keypoints:
    
    count = 0
    kp_deque = deque(maxlen=9)
    try:
        ckpt, time3 = ckpt_time(time2)
        while True:
            ckpt, time3 = ckpt_time(time2)
            print('-------------- generate reconstruction 3D data spends {:.2f} seconds'.format(ckpt))
            
            kp = generator.Q.get()
            # continue
            ## TODO TEST
            cv2.imshow("before 3d",kp["image"])
            cv2.waitKey(10)
            
            

            if isinstance(kp["keypoints"],int): 
                sequencial_animation.call_noperson(kp["image"])

            kp_deque.append(kp["keypoints"].numpy())    
            if len(kp_deque)<9:
                continue
            
            #! estimate 3d pose 
            # normlization keypoints  Suppose using the camera parameter
            input_keypoints = normalize_screen_coordinates(np.asarray(kp_deque)[..., :2], w=1000, h=1002)
            prediction = evaluate(input_keypoints, pad, model_pos, return_predictions=True)

            # save 3D joint points
            # np.save('outputs/test_3d_output.npy', prediction, allow_pickle=True)

            # rotate the camera perspective
            rot = np.array([0.14070565, -0.15007018, -0.7552408, 0.62232804], dtype=np.float32)
            prediction = camera_to_world(prediction, R=rot, t=0)

            # We don't have the trajectory, but at least we can rebase the height
            prediction[:, :, 2] -= np.min(prediction[:, :, 2])
            input_keypoints = image_coordinates(input_keypoints[..., :2], w=1000, h=1002)

            # ckpt, time3 = ckpt_time(time2)
            # print('-------------- generate reconstruction 3D data spends {:.2f} seconds'.format(ckpt))

            #! Visualization
            sequencial_animation.call(input_keypoints, prediction, kp["image"])   # TODO
            # time.sleep(3)
            print("frame ",count)
            count += 1
            # plt.show()  #TODO

    except KeyboardInterrupt:
        return

        pos_list = sequencial_animation.get_pos_list()
        np.save("outputs/3dpose.npy",pos_list)
        plt.ioff()
        plt.show()
        cap.release()
        cv2.destroyAllWindows()
    

def inference_video(video_path, detector_2d):
    """
    Do image -> 2d points -> 3d points to video.
    :param detector_2d: used 2d joints detector. Can be {alpha_pose, hr_pose}
    :param video_path: relative to outputs
    :return: None
    """
    args = parse_args()
    print(args)

    args.detector_2d = detector_2d
    dir_name = os.path.dirname(video_path)
    basename = os.path.basename(video_path)
    video_name = basename[:basename.rfind('.')]
    args.viz_video = video_path
    # args.viz_output = f'{dir_name}/{args.detector_2d}_{video_name}.mp4'
    # args.viz_limit = 20
    # args.input_npz = 'outputs/alpha_pose_dance/dance.npz'
    

    args.evaluate = 'pretrained_h36m_detectron_coco.bin'

    with Timer(video_path):
        main(args)


def inference_camera():
    args = parse_args()
    main(args)
    # main_cam(args)


if __name__ == '__main__':
    inference_camera()
