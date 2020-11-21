## phase_align.py
#
# author: Stefan Harmeling
# date:   2020-11-21
#
# Goal: how to measure phase alignment

from lab import *

# generate the images of the base frequencies
def base_sin(sx, loc, p=None):
    # generate an image of size sx with a sinus
    # corresponding to the loc in Fourierspace
    X = 1j*zeros(sx)      # fourier transform of some x
    X[tuple(loc)] = 1  # set a single phase to one
    if p is not None:
        # phase is given as well, i.e. some value [0, 2*pi]
        X[tuple(loc)] *= exp(1j * p * ones(1,1))[0,0]
    X += phase_flip(X.conj())
    return ifftn(X, norm='forward').real

def base_cor(b1, b2):
    # calculate the correlation between two tensors
    return (b1 * b2).sum()
    
