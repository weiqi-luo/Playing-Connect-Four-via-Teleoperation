"""
This module implements a tf coordinate frame like in ROS.
It can be updated and used for animations.
Dependency:
    numpy-quaternion
"""

import numpy as np
import quaternion
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D


class Arrow3D(FancyArrowPatch):
    """
    Plot a 3D arrow by extending the 2D FancyArrowPatch. For animated plots
    the arrow coordinates can be updated by set_coord()
    >>> a = Arrow3D([0,1], [0,0], [0,0], mutation_scale=10, lw=1, arrowstyle="-|>")
    >>> ax.add_artist(a)
    """

    def __init__(self, xs, ys, zs, *args, **kwargs):
        """@param xs, ys, zs   2D vector each, start and end point for x, y, z"""
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def set_coord(self, xs, ys, zs):
        """Update the arrow coordinates"""
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        """Draw the 3D arrow"""
        xs, ys, zs = proj3d.proj_transform(*self._verts3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


class TfAxis:
    """
    Visualize a homogeneous transform with origin and the orientation from a quaternion
    The TfAxis pose (origin and quat) can be updated by set_pose() that allows for animated plots.
    If FuncAnimation() is not used, the plot must be updated [fig.canvas.draw()].
    >>> fig = plt.figure()
    >>> ax = Axes3D(fig)
    >>> tf = TfAxis(origin=[0.2, 0.2, 0.2], quat=[1,0,0,0], scale=0.4)
    >>> tf.plot(ax)
    """
    DEFAULT_COLOR = ('red', 'forestgreen', 'mediumblue')
    GRAY_COLOR = ('gray', 'gray', 'gray')

    def __init__(self, origin=[0, 0, 0], quat=np.quaternion(1), *,
                 scale=0.4, ax=None, color=DEFAULT_COLOR, **kwargs):
        """
        @param origin           iterable [x, y, z]
        @param quat             np.quaternion or iterable [qw, qx, qy, qz]
        @param scale            length of a coordinate axis (float)
        @param ax               mpl_toolkits.mplot3d.Axes3D object
        @param color            color for x, y and z axis
        @param mutation_scale   kwargs  scale of the arrows (default 10)
        @param lw               kwargs  line width          (default 1)
        """
        # plot handle for the TfAxis origin
        self._origin_plt = None
        self._scale = float(scale)

        self._origin, self._quat = self._check_transform(origin, quat)
        coord_axis = self._make_coord_axis()

        # args for Arrow3D by kwargs or default values
        args = {'arrowstyle': '-|>'}
        for key, default in zip(['mutation_scale', 'lw'], [10, 1]):
            args[key] = default if key not in kwargs.keys() else kwargs[key]

        # TfAxis consists of three Arrow3D
        self.axis = [Arrow3D(*axis, **args, color=c)
                     for axis, c in zip(coord_axis, color)]

        if ax is not None:
            self.plot(ax)

    def __repr__(self):
        return '{}(origin({}), {})'.format(type(self).__name__, self._origin, self._quat)

    def plot(self, ax):
        """Plots the tf coordinate system on a Axes3D plot"""
        assert type(ax) == Axes3D, "TfAxis only defined for Axes3D plot."
        for arrow in self.axis:
            ax.add_artist(arrow)
        self._plot_origin(ax)

    def set_pose(self, origin, quat=np.quaternion(1)):
        """
        Update the TfAxis pose
        @param origin   iterable [x, y, z]
        @param quat     np.quaternion or iterable [qw, qx, qy, qz]', mat)
        """

        self._origin, self._quat = self._check_transform(origin, quat)
        # compute and set new coord system
        coord_axis = self._make_coord_axis()
        for ax, axis in zip(self.axis, coord_axis):
            ax.set_coord(*axis)
        self._plot_origin()

    def _check_transform(self, origin, quat):
        """
        Verify the input for origin and quat and converts them into tuples
        @param origin   iterable [x, y, z]
        @param quat     np.quaternion or iterable [qw, qx, qy, qz]
        @return         np.array(origin), np.quaternion(quat)
        """
        origin = np.asarray(origin, dtype=np.float64)
        assert origin.shape == (3,),  'origin must be a 3D vector'
        if not isinstance(quat, np.quaternion):
            quat = quaternion.from_float_array(quat)
        return origin, quat

    def _make_coord_axis(self):
        """
        Generates a coordinate system with self._origin and the orientation from self._quat
        @return     list of 3 np.array(2,3), each encoding an Arrow3D
        """
        # rotation matrix from quat and scale the axis
        coord_ax = self._scale * quaternion.as_rotation_matrix(self.quat).T
        coord_axis = [np.vstack(([0, 0, 0], coord_ax[i])) for i in range(3)]
        return [np.add(axis, self.origin).T for axis in coord_axis]

    def _plot_origin(self, ax=None):
        """Plots/updates the origin of the coordinate frame"""
        if self._origin_plt is None:        # init plot
            self._origin_plt = ax.plot(*[[x] for x in self._origin],
                                       c='k', marker='o', markersize=3)[0]
        else:                               # change [x,y,z] data
            self._origin_plt.set_data(*self._origin[:2])
            self._origin_plt.set_3d_properties(self._origin[2])

    @property
    def origin(self):
        return self._origin

    @property
    def quat(self):
        return self._quat
