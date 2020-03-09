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
import datetime as dt
import matplotlib.animation as animation
import math
from collections import deque
from pylab import get_current_fig_manager
import matplotlib
matplotlib.use( 'tkagg' )

class Sequencial_animation():
    def __init__(self,skeleton, azim,size=6,i=8):        
        # self.fig.tight_layout()
        # prevent wired error
        _ = Axes3D.__class__.__name__
        radius = 1.7
        self.fig_3d = plt.figure(figsize=(size , size))
        self.ax_3d = self.fig_3d.add_subplot(1, 1, 1, projection='3d')
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
        self.parents = skeleton.parents()
        self.joints_right = skeleton.joints_right()    
        self.i = i
        thismanager = get_current_fig_manager()
        thismanager.window.wm_geometry("+0+1000")


    def call(self, data):
        # Update 2D poses
        if not self.initialized:
            for j, j_parent in enumerate(self.parents):
                if j_parent == -1:
                    continue
                col = 'red' if j in self.joints_right else 'black'
                self.lines_3d.append(self.ax_3d.plot([data[j, 0], data[j_parent, 0]],
                                        [data[j, 1], data[j_parent, 1]],
                                        [data[j, 2], data[j_parent, 2]], zdir='z', c=col))
            self.initialized = True   
        else:
            for j, j_parent in enumerate(self.parents):
                if j_parent == -1:
                    continue
                self.lines_3d[j - 1][0].set_xdata([data[j, 0], data[j_parent, 0]])
                self.lines_3d[j - 1][0].set_ydata([data[j, 1], data[j_parent, 1]])
                self.lines_3d[j - 1][0].set_3d_properties([data[j, 2], data[j_parent, 2]], zdir='z')
        plt.draw()
        plt.pause(0.000000000000000001)




class RealtimePlot:
    def __init__(self, fig, axes, label, color, fixylim = True, max_entries = 100):
        self.fig = fig
        self.axes = axes
        self.axis_x = deque(maxlen=max_entries)
        self.axis_y = deque(maxlen=max_entries)
        self.max_entries = max_entries
        
        self.lineplot, = self.axes.plot([], [], color+"o-",label=label)
        self.axes.set_autoscaley_on(True)
        if fixylim:
            self.axes.set_ylim(-180,180)
        self.axes.legend()


    def call(self, x, y):
        self.axis_x.append(x)
        self.axis_y.append(y)
        self.lineplot.set_data(self.axis_x, self.axis_y)
        self.axes.set_xlim(self.axis_x[0], self.axis_x[-1] + 1e-15)
        self.axes.relim(); self.axes.autoscale_view() # rescale the y-axis
        plt.pause(0.001)

    def animate(self, callback, interval = 50):
        def wrapper(frame_index):
            self.call(*callback(frame_index))
            self.axes.relim(); self.axes.autoscale_view() # rescale the y-axis
            return self.lineplot
        animation.FuncAnimation(self.fig, wrapper, interval=interval)
