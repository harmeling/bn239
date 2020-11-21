## from https://vincentstudio.info/2019/02/08/012_Use_Qt5Agg_GUI_backend_for_matplotlib_on_MacOS/

import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
speed = 0.03
fig, ax = plt.subplots()
x = np.arange(0, 2 * np.pi, 0.01)
line, = ax.plot(x, np.sin(x))
ani = animation.FuncAnimation(
    fig=fig,
    func=lambda i: line.set_ydata(np.sin(x + i * speed)),
    frames=int(2 * np.pi / speed),
    init_func=lambda: line.set_ydata(np.sin(x)),
    interval=1,
    blit=False
)
plt.show()
