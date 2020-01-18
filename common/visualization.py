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
from common.utils import read_video
import matplotlib
matplotlib.use( 'tkagg' )

class Sequencial_animation():
    def __init__(self,azim,size=6):
        plt.ion()   # continuously plot
        # plt.ioff()
        self.len_poses=1
        # self.fig = plt.figure(figsize=(size * (1 +  self.len_poses), size))
        self.fig_in = plt.figure(figsize=(size , size))
        self.ax_in = self.fig_in.add_subplot(1, 1, 1)
        # self.ax_in = self.fig.add_subplot(1, 1 +  self.len_poses, 1)
        self.ax_in.get_xaxis().set_visible(False)
        self.ax_in.get_yaxis().set_visible(False)
        self.ax_in.set_axis_off()
        self.ax_in.set_title('Input')
        
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

    def ckpt_time(self,ckpt=None, display=0, desc=''):
        if not ckpt:
            return time.time()
        else:
            if display:
                print(desc + ' consume time {:0.4f}'.format(time.time() - float(ckpt)))
            return time.time() - float(ckpt), time.time()


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


    def call(self,keypoints, poses, skeleton, fps, bitrate, output, viewport,
                        limit=-1, downsample=1, input_video_path=None, input_video_skip=0):
        """
        TODO
        Render an animation. The supported output modes are:
        -- 'interactive': display an interactive self.figure
                        (also works on notebooks if associated with %matplotlib inline)
        -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
        -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
        -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
        """


        lines_3d = [[]]
        trajectories = []
        # for index, (title, data) in enumerate(poses.items()):
        data = poses['Reconstruction']
        trajectories.append(data[:, 0, [0, 1]])

        poses=[data]
        # Decode video
        if input_video_path is None:
            # Black background
            all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
        else:
            # Load video using ffmpeg
            all_frames = []
            for f in read_video(input_video_path, fps=None, skip=input_video_skip):
                all_frames.append(f)

            effective_length = min(keypoints.shape[0], len(all_frames))
            all_frames = all_frames[:effective_length]

        if downsample > 1:
            keypoints = self.downsample_tensor(keypoints, downsample)
            all_frames = self.downsample_tensor(np.array(all_frames), downsample).astype('uint8')
            for idx in range( self.len_poses):
                poses[idx] = self.downsample_tensor(poses[idx], downsample)
                trajectories[idx] = self.downsample_tensor(trajectories[idx], downsample)
            fps /= downsample

        initialized = False
        image = None
        lines = []
        points = None

        if limit < 1:
            limit = len(all_frames)
        else:
            limit = min(limit, len(all_frames))

        parents = skeleton.parents()
        # pbar = tqdm(total=limit)

        def update_video(i):
            self.ax_in.clear()
            self.ax_3d.clear()
            nonlocal initialized, image, lines, points

            # Update 2D poses
            if not initialized:
                image = self.ax_in.imshow(all_frames[i], aspect='equal')

                for j, j_parent in enumerate(parents):
                    if j_parent == -1:
                        continue

                    col = 'red' if j in skeleton.joints_right() else 'black'
                    n=0
                    pos = poses[n][i]
                    lines_3d[n].append(self.ax_3d.plot([pos[j, 0], pos[j_parent, 0]],
                                            [pos[j, 1], pos[j_parent, 1]],
                                            [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
                                

                points = self.ax_in.scatter(*keypoints[i].T, 5, color='red', edgecolors='white', zorder=10)

                initialized = True
            else:
                print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                image.set_data(all_frames[i])

                for j, j_parent in enumerate(parents):
                    if j_parent == -1:
                        continue

                    # if len(parents) == keypoints.shape[1] and 1 == 2:
                    #     lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                    #                              [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                    for n, ax in enumerate(self.ax_3d):
                        pos = poses[n][i]
                        lines_3d[n][j - 1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                        lines_3d[n][j - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                        lines_3d[n][j - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')

                points.set_offsets(keypoints[i])



        # anim = FuncAnimation(self.fig, update_video, frames=limit, interval=1000.0 / fps, repeat=False)
        update_video(0)
        plt.draw()
        plt.pause(0.0000001)

