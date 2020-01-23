# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, writers
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import sys
import matplotlib
matplotlib.use( 'tkagg' )

class Sequencial_animation():
    def __init__(self,skeleton, azim,fps, size=6, limit=-1,downsample=1,i=8):
        # plt.ioff()
        self.len_poses=1
        
        ##################################################################################
        # # self.fig = plt.figure(figsize=(size * (1 +  self.len_poses), size))
        # self.fig_in = plt.figure(figsize=(size , size))
        # self.ax_in = self.fig_in.add_subplot(1, 1, 1)
        # # self.ax_in = self.fig.add_subplot(1, 1 +  self.len_poses, 1)
        # self.ax_in.get_xaxis().set_visible(False)
        # self.ax_in.get_yaxis().set_visible(False)
        # self.ax_in.set_axis_off()
        # self.ax_in.set_title('Input')
        ##################################################################################
        
        # self.fig.tight_layout()
        # prevent wired error
        _ = Axes3D.__class__.__name__

        radius = 1.7
        self.fig_3d = plt.figure(figsize=(size , size))
        self.ax_3d = self.fig_3d.add_subplot(1, 1, 1, projection='3d')
        # self.ax_3d = self.fig.add_subplot(1, 1 +  self.len_poses, 2, projection='3d')
        self.ax_3d.view_init(elev=15., azim=azim)
        self.ax_3d.set_xlim3d([-radius / 2, radius / 2])
        self.ax_3d.set_zlim3d([0, radius])
        self.ax_3d.set_ylim3d([-radius / 2, radius / 2])
        self.ax_3d.set_xticklabels([])
        self.ax_3d.set_yticklabels([])
        self.ax_3d.set_zticklabels([])
        self.ax_3d.dist = 12.5
        self.ax_3d.set_title('Reconstruction')  
        self.initialized = False
        self.image = None
        self.lines_3d = []
        self.pos_list = []
        self.point= None

        self.downsample = downsample
        self.parents = skeleton.parents()
        self.joints_right = skeleton.joints_right()
        
        self.i = i

    def ckpt_time(self,ckpt=None, display=0, desc=''):
        if not ckpt:
            return time.time()
        else:
            if display:
                print(desc + ' consume time {:0.4f}'.format(time.time() - float(ckpt)))
            return time.time() - float(ckpt), time.time()


    def get_pos_list(self):
        return self.pos_list


    def set_equal_aspect(self,ax, data):
        """
        Create white cubic bounding box to make sure that 3d axis is in equal aspect.
        :param ax: 3D axis
        :param data: shape of(frames, 3), generated from BVH using convert_bvh2dataset.py
        """
        X, Y, Z = data[..., 0], data[..., 1], data[..., 2]

        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (X.max() + X.min())
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (Y.max() + Y.min())
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (Z.max() + Z.min())

        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')


    def downsample_tensor(self,X, factor):
        length = X.shape[0] // factor * factor
        return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


    def call(self,keypoints, data): #TODO

        # if self.downsample > 1:
        #     all_frames = self.downsample_tensor(np.array(current_frame), self.downsample).astype('uint8')
        #     keypoints = self.downsample_tensor(keypoints, self.downsample)
        #     data = self.downsample_tensor(data, self.downsample)

        # Update 2D poses
        if not self.initialized:
            ###########################################################################################################
            # self.image = self.ax_in.imshow(current_frame, aspect='equal')
            # self.point= self.ax_in.scatter(*keypoints[self.i].T, 5, color='red', edgecolors='white', zorder=10)
            ###########################################################################################################
            for j, j_parent in enumerate(self.parents):
                if j_parent == -1:
                    continue
                col = 'red' if j in self.joints_right else 'black'
                pos = data[self.i]
                self.lines_3d.append(self.ax_3d.plot([pos[j, 0], pos[j_parent, 0]],
                                        [pos[j, 1], pos[j_parent, 1]],
                                        [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
            self.initialized = True
        
        else:
            ######################################################################################
            # self.image = self.ax_in.imshow(current_frame, aspect='equal')
            # self.image.set_data(current_frame)
            # self.point.set_offsets(keypoints[self.i])
            ######################################################################################
            for j, j_parent in enumerate(self.parents):
                if j_parent == -1:
                    continue
                pos = data[self.i]
                self.lines_3d[j - 1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                self.lines_3d[j - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                self.lines_3d[j - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')
        # anim = FuncAnimation(self.fig, update_video, frames=limit, interval=1000.0 / self.fps, repeat=False)
        self.pos_list.append(pos)
        plt.draw()
        plt.pause(0.000000000000000001)
