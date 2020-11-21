## fourier transforms, phase, magnitude, reconstruction
# these are compatible with rfft, irfft, 
# i.e. the real and imaginary are combined

import torch
import torch.fft
from numpy import pi

## basics
onesided = False
def fft2(batch, paddings=None):
    if paddings:
        batch = torch.pad(batch, paddings)
    return torch.rfft(batch, 2, onesided=onesided)

def ifft2(batch_ft):
    return torch.irfft(batch_ft, signal_ndim=2, onesided=onesided)

def real(batch_ri):
    return batch_ri[...,0]

def imag(batch_ri):
    return batch_ri[...,1]

def phase(batch):
    return torch.atan2(imag(batch), real(batch))

def magnitude(batch):
    return torch.sqrt(torch.sum(batch**2, -1))

def mag(batch):
    return magnitude(batch)

def c2p(batch_ri):
    # cartesian to polar (real/imag to mag/phase)
    return magnitude(batch_ri), phase(batch_ri)

def p2c(batch_m, batch_p):
    # polar to cartesian (mag/phase to real/imag)
    batch_m  = batch_m.unsqueeze(-1)       # magnitude
    batch_p  = batch_p.unsqueeze(-1)       # phase
    batch_r  = batch_m*torch.cos(batch_p)  # real
    batch_i  = batch_m*torch.sin(batch_p)  # imag
    batch_ri = torch.cat((batch_r, batch_i), dim=-1)
    return batch_ri

def p2im(batch_m, batch_p):
    return ifft2(p2c(batch_m, batch_p))

## flipping the phase
def phase_flip2(p):
    "flips the phase such that p + phase_flip(p) == 0"
    return p.flip([0,1]).roll([1,1],[0,1])

## discretizing the phase
# notes:
# * we have to ensure that also 0.0 is among the landmarks,
#   the following implementation has 0.0 always as label 0
# * the discretized phases shoule be symmetric, i.e. for each label but zero,
#   there should be a conjugated phase label
# * we should be independent on the implementation of atan2, 
#   i.e. we should accept any real number as phase not just [-pi,pi]
def phase_landmarks(c):
    # landmarks used in the following algorithm
    return torch.linspace(0, 2*pi, c+1)[0:c]
    
def phases2labels(sample_p, c):
    # c is number of discrete labels equispaced from [0,2*pi[
    sample_p = torch.remainder(sample_p, 2*pi) # in [0,2*pi[
    sample_p = c * sample_p / (2*pi)           # in [0,c[
    label_p  = torch.round(sample_p)           # in {0,1,...,c}
    label_p[label_p==c] = 0                    # in {0,1,...,c-1}
    label_p  = label_p.to(torch.long)
    return label_p

def labels2phases(label_p, c):
    sample_p = torch.empty_like(label_p, dtype=torch.float32)
    for l, p in enumerate(phase_landmarks(c)):
        sample_p[label_p==l] = p
    return sample_p

def quant_phases(sample_p, c):
    return labels2phases(phases2labels(sample_p, c), c)

def quant_image(sample, c):
    sample_m, sample_p = c2p(fft2(sample))
    sample_ft = p2c(sample_m, quant_phases(sample_p, c))
    return ifft2(sample_ft)

# cropping
def crop2(x, sx, offset=None):
    pass

# convolution
def cnv2(x, a, shape=None):
    # 'x' is image, 'a' is filter
    return real(ifft2(fft2(x) * fft2(a, x.shape)))
def cnv2tp(x, a):
    return real(ifft2(fft2(x) * fft2(a, x.shape)))
