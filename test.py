import matplotlib.pyplot as plt

from pylab import get_current_fig_manager
import matplotlib
matplotlib.use( 'tkagg' )

# plt.figure()
# thismanager = get_current_fig_manager()
# thismanager.window.wm_geometry("-1000+0")

plt.figure()
thismanager = get_current_fig_manager()
thismanager.window.wm_geometry("+1000+0")

plt.figure()
thismanager = get_current_fig_manager()
thismanager.window.wm_geometry("+0+1000")

plt.figure()
thismanager = get_current_fig_manager()
thismanager.window.wm_geometry("+0-1000")

plt.show()