import numpy as np            #numpy for Sine function
import matplotlib.pyplot as plt    #matplotlib for plotting functions
x=np.linspace(0, 2*np.pi,1000)
fig = plt.figure()       
ax = fig.add_subplot(111)
for t in range(0,500):   #looping statement;declare the total number of frames
   y=np.sin(x-0.2*t)       # traveling Sine wave
   ax.clear()
   ax.plot(x,y)
   plt.pause(0.1)
