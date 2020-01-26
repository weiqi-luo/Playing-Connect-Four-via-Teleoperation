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


    def call(self, data):

        # Update 2D poses
        if not self.initialized:
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
            for j, j_parent in enumerate(self.parents):
                if j_parent == -1:
                    continue
                pos = data[self.i]
                self.lines_3d[j - 1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                self.lines_3d[j - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                self.lines_3d[j - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')
        plt.draw()
        plt.pause(0.000000000000000001)
