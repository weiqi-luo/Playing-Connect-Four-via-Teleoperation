import ntpath
import os
import shutil

import numpy as np
import torch.utils.data
from tqdm import tqdm

from SPPE.src.main_fast_inference import *
from common.utils import calculate_area
from fn import getTime
from opt import opt
from pPose_nms import write_json

from yolo.darknet import Darknet
from yolo.preprocess import prep_image, prep_frame
from yolo.util import dynamic_write_results

# from threading import enumerate

args = opt
args.dataset = 'coco'
args.fast_inference = False
args.save_img = True
args.sp = True
if not args.sp:
    torch.multiprocessing.set_start_method('forkserver', force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')


##########################################################################################
##########################################################################################


import os
import sys
import time
from multiprocessing import Queue as pQueue
from threading import Thread

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable

from SPPE.src.utils.eval import getPrediction, getMultiPeakPrediction
from SPPE.src.utils.img import load_image, cropBox, im_to_torch
from SPPE.src.main_fast_inference import *
from common.utils import calculate_area
from matching import candidate_reselect as matching
from opt import opt
from pPose_nms import pose_nms
from yolo.darknet import Darknet
from yolo.preprocess import prep_image, prep_frame
from yolo.util import dynamic_write_results
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use( 'tkagg' )
# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue, LifoQueue
# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue, LifoQueue

if opt.vis_fast:
    from fn import vis_frame_fast as vis_frame
else:
    from fn import vis_frame


##########################################################################################
##########################################################################################

class DetectionLoader:
    def __init__(self, batchSize=1, queueSize=1):
        ## queue
        if opt.sp:
            self.Q = Queue(maxsize=queueSize)
        else:
            self.Q = mp.Queue(maxsize=queueSize)
        ## camera stream
        self.stream = cv2.VideoCapture(0)
        assert self.stream.isOpened(), 'Cannot capture from camera'
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.inp_dim = int(opt.inp_dim)

        ## yolo model
        self.det_model = Darknet("joints_detectors/Alphapose/yolo/cfg/yolov3-spp.cfg")
        self.det_model.load_weights('joints_detectors/Alphapose/models/yolo/yolov3-spp.weights')
        self.det_model.net_info['height'] = opt.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        assert self.det_inp_dim % 32 == 0
        assert self.det_inp_dim > 32
        self.det_model.cuda()
        self.det_model.eval()
        self.batchSize = batchSize
        self.datalen = 1
        leftover = 0
        if (self.datalen) % batchSize:
            leftover = 1
        self.num_batches = self.datalen // batchSize + leftover

        ## alphapose model
        fast_inference = False
        pose_dataset = Mscoco()
        if fast_inference:
            self.pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            self.pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        
        self.pose_model.cuda()
        self.pose_model.eval() 


    def update(self):
        grabbed, frame = self.stream.read()
        img_k, orig_img_k, im_dim_list_k = prep_frame(frame, self.inp_dim)
        
        img = [img_k]
        orig_img = [orig_img_k]
        im_name = ["im_name"]
        im_dim_list = [im_dim_list_k] 

        img = torch.cat(img)
        im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

        ### detector 
        #########################

        with torch.no_grad():
            # Human Detection
            img = img.cuda()
            prediction = self.det_model(img, CUDA=True)
            # NMS process
            dets = dynamic_write_results(prediction, opt.confidence,
                                        opt.num_classes, nms=True, nms_conf=opt.nms_thesh)
            if isinstance(dets, int) or dets.shape[0] == 0:   
                raise NotImplementedError
                
            
            dets = dets.cpu()
            im_dim_list = torch.index_select(im_dim_list, 0, dets[:, 0].long())
            scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

            # coordinate transfer
            dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

            dets[:, 1:5] /= scaling_factor
            for j in range(dets.shape[0]):
                dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]

            #for k in range(len(orig_img)):
            boxes_k = boxes[dets[:, 0] == 0]
            if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                res = {'keypoints': -1,
                        'image': orig_img}
                return res 
            inps = torch.zeros(boxes_k.size(0), 3, opt.inputResH, opt.inputResW)
            pt1 = torch.zeros(boxes_k.size(0), 2)
            pt2 = torch.zeros(boxes_k.size(0), 2)


            ### processor 
            #########################
            orig_img = orig_img[0]
            inp = im_to_torch(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = self.crop_from_dets(inp, boxes, inps, pt1, pt2)

            ### generator
            #########################            
            orig_img = np.array(orig_img, dtype=np.uint8)
            # location prediction (n, kp, 2) | score prediction (n, kp, 1)

            datalen = inps.size(0)
            batchSize = 20 #args.posebatch()
            leftover = 0
            if datalen % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []

            for j in range(num_batches):
                inps_j = inps[j * batchSize:min((j + 1) * batchSize, datalen)].cuda()
                hm_j = self.pose_model(inps_j)
                hm.append(hm_j)
            
            
            hm = torch.cat(hm)
            hm = hm.cpu().data

            preds_hm, preds_img, preds_scores = getPrediction(
                hm, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
            result = pose_nms(
                boxes, scores, preds_img, preds_scores)
                    
            if not result: # No people
                res = {'keypoints': -1,
                        'image': orig_img}
                return res 
            else:
                kpt = max(result,
                        key=lambda x: x['proposal_score'].data[0] * calculate_area(x['keypoints']), )['keypoints']

                res = {'keypoints': kpt,
                        'image': orig_img}
                return res 
            


    def crop_from_dets(self,img, boxes, inps, pt1, pt2):
        '''
        Crop human from origin image according to Dectecion Results
        '''

        imght = img.size(1)
        imgwidth = img.size(2)
        tmp_img = img
        tmp_img[0].add_(-0.406)
        tmp_img[1].add_(-0.457)
        tmp_img[2].add_(-0.480)
        for i, box in enumerate(boxes):
            upLeft = torch.Tensor(
                (float(box[0]), float(box[1])))
            bottomRight = torch.Tensor(
                (float(box[2]), float(box[3])))

            ht = bottomRight[1] - upLeft[1]
            width = bottomRight[0] - upLeft[0]

            scaleRate = 0.3

            upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
            upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
            bottomRight[0] = max(
                min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
            bottomRight[1] = max(
                min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

            try:
                inps[i] = cropBox(tmp_img.clone(), upLeft, bottomRight, opt.inputResH, opt.inputResW)
            except IndexError:
                print(tmp_img.shape)
                print(upLeft)
                print(bottomRight)
                print('===')
            pt1[i] = upLeft
            pt2[i] = bottomRight

        return inps, pt1, pt2



##########################################################################################
##########################################################################################


class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'  # root image folders
        self.is_train = train  # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints_mpii = 16
        self.nJoints = 33

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

