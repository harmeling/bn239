# phase_quant.py
#
# author: Stefan Harmeling
# date:   2020-11-21
#
# Goal: how to discretize phases

import torch
import torch.fft

## discretizing the phase
# notes:
# * we have to ensure that also 0.0 is among the landmarks,
#   the following implementation has 0.0 always as label 0
# * the discretized phases shoule be symmetric, i.e. for each label but zero,
#   there should be a conjugated phase label
# * we should be independent on the implementation of atan2 (or angle)
#   i.e. we should accept any real number as phase not just [-pi,pi]

def phase_landmarks(c):
    # landmarks used in the following algorithm
    return torch.linspace(0, 2*pi, c+1)[0:c]
    
def phase_to_label(p, c):
    # c is number of discrete labels equispaced from [0,2*pi[
    p = torch.remainder(p, 2*pi)   # in [0,2*pi[
    p = c * p / (2*pi)             # in [0,c[
    lp = torch.round(p)            # in {0,1,...,c}
    lp[lp==c] = 0                  # in {0,1,...,c-1}
    lp = lp.to(torch.long)
    return lp

def label_to_phase(lp, c):
    # convert labels to phases
    p = torch.empty_like(lp, dtype=torch.float32)
    for li, pi in enumerate(phase_landmarks(c)):
        p[lp==li] = pi
    return p

def quant_phase(p, c):
    # quantize phases
    return label_to_phase(phase_to_label(p, c), c)

def quant_tensor(x, c):
    # quantize tensor by quantizing its phases
    X = torch.fft.fftn(x)
    m, p = X.abs(), quant_phase(X.angle(), c)
    return torch.fft.ifftn(m * exp(1j * p)).real

if __name__ == '__main__':
    from lab import *
    x = cameraman()
    rows, cols = 3, 5
    sp = 1
    subplot(rows, cols, sp)
    imshow(x)
    title('original')
    axis('off')
    for c in range(3, rows*cols+2):
        sp += 1
        subplot(rows, cols, sp)
        imshow(quant_tensor(x,c))
        title('c='+str(c))
        axis('off')
    waitforbuttonpress()
    
