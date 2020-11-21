# lab.py
#
# author: Stefan Harmeling
# date:   2020-06-30
#
# Goal: easy-to-use functions (inspired by matlab)
#       based on pytorch (and not numpy)

## default stuff to make you feel comfy
from torch import zeros, ones, randn, rand, arange, stack, cat, linspace, zeros_like, roll, linspace
from torch import tensor, get_default_dtype
from torch import log, exp, norm, sqrt, norm, sqrt, exp
from torch.fft import fft, ifft, fftn, ifftn, rfft, irfft
from math import pi

## more missing functions
def fft_shift(x):
    # shift the zero frequency into the center
    return x.roll([i//2 for i in x.shape], list(range(x.dim())))

## flipping the phase
def phase_flip2(p):
    # flips the phase such that p + phase_flip(p) == 0
    return p.flip([0,1]).roll([1,1],[0,1])

def phase_flip(p):
    # flips the phase such that p + phase_flip(p) == 0
    d = p.dim()
    dims = list(range(d))
    return p.flip(dims).roll(d*[1], dims)


## useful for loading simple files with numbers
from numpy import loadtxt

# plotting matplotlib
import matplotlib
matplotlib.use("Qt5Agg")   # comment this out for notebook
from matplotlib.pyplot import *
ion()       # switch on interactive plotting
from matplotlib.patches import Circle

# debugger
import pdb
keyboard = pdb.set_trace     # matlab style "keyboard()" for debugging

# os
import os
pwd = os.getcwd

# subprocess
import subprocess
def ls(args=None):
    if args is None:
        subprocess.run(["ls"])
    else:
        subprocess.run(["ls", args])
#def open(args):
#    subprocess.run(["open", args])

# more tools!
    
# plotting discontinuous functions
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

# imshow and show value
def imshowvalue(a):
    imshow(a)
    sa = shape(a)
    for i in range(sa[1]):
        for j in range(sa[0]):
            if a[j][i] > 0.0:
                text(i, j, a[j][i], horizontalalignment='center', fontsize=5)

## useful images
def barbara():     return tensor(imread('images/barbara.png'),     dtype=float)
def boat():        return tensor(imread('images/boat.png'),        dtype=float)
def cameraman():   return tensor(imread('images/cameraman.tif'),   dtype=float)
def fingerprint(): return tensor(imread('images/fingerprint.png'), dtype=float)
def house():       return tensor(imread('images/house.png'),       dtype=float)
def lena():        return tensor(imread('images/lena.png'),        dtype=float)
def peppers256():  return tensor(imread('images/peppers256.png'),  dtype=float)
