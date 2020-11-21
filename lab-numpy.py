# lab-numpy.py
#
# author: Stefan Harmeling
# date:   2019-03-22
#
# Goal: easy-to-use functions (inspired by matlab)
#       based on numpy

## scipy
from scipy.special import erf, erfinv

## numpy
import numpy as np
from numpy        import *
from numpy.fft    import fft, ifft, fft2, ifft2, fftn, ifftn
def rand(*args):
    if len(args) == 1 and iterable(args[0]): # allows rand([3,4])
        return np.random.rand(*args[0])      # allows rand(3,4)
    return np.random.rand(*args)
def randn(*args):
    if len(args) == 1 and iterable(args[0]):
        return np.random.randn(*args[0])  # allows randn([3,4])
    return np.random.randn(*args)         # allows randn(3,4)
def ones(*args):
    if len(args) == 1:
        return np.ones(args[0])   # allows ones([3,4])
    return np.ones(args)          # allow ones(3,4)
def zeros(*args):
    if len(args) == 1:
        return np.zeros(args[0])  # allows zeros([3,4])
    return np.zeros(args)         # allow zeros(3,4)

## matplotlib
from matplotlib.pyplot import *
# ion, figure, plot, axes, hist, imshow, plot, title, xlabel, ylabel, draw, pause, clf, subplot, cla, colorbar, gcf, gca, savefig, close, xlim, ylim
ion()       # switch on interactive plotting
###from matplotlib import rc
###rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
##### for Palatino and other serif fonts use:
####rc('font',**{'family':'serif','serif':['Palatino']})
###rc('text', usetex=True)
from matplotlib.patches import Circle

## debugger
import pdb
keyboard = pdb.set_trace     # matlab style "keyboard()" for debugging

## os
import os
pwd = os.getcwd

## subprocess
import subprocess
def ls(args=None):
    if args is None:
        subprocess.run(["ls"])
    else:
        subprocess.run(["ls", args])
#def open(args):
#    subprocess.run(["open", args])

### more tools!
    
## plotting discontinuous functions
def plotdis(x, y, xs, ys0, ys1):
    # xs, ys0, ys1 are equal sized list
    # e.g. xs[0] ys0[0] a circle
    #      xs[0] ys1[0] a disc
    #plot(x, y)
    xy = vstack([hstack([x,xs]),hstack([y,len(xs)*[nan]])]).T
    xy = xy[argsort(xy[:,0]),:]
    plot(xy[:,0],xy[:,1], color='orange')
    plot(xs, ys0, 'o', fillstyle='none', color='orange')
    plot(xs, ys1, 'o', fillstyle='full', color='orange')

## imshow and show value
def imshowvalue(a):
    imshow(a)
    sa = shape(a)
    for i in range(sa[1]):
        for j in range(sa[0]):
            if a[j][i] > 0.0:
                text(i, j, a[j][i], horizontalalignment='center', fontsize=5)

### circular convolution
#def cnv2(x, a):
#    # 'x' is image, 'a' is filter
#    return real(ifft2(fft2(x) * fft2(a, x.shape)))
#def cnv2tp(x, a):
#    return real(ifft2(fft2(x) * fft2(a, x.shape)))
