## 2020-09-25
#
# base causal inference on entropy?

from lab import *

## generate some simple data
nx = 50        # should be even number  
ny = 50

def simpleJoint(nx, ny):
    p = zeros(nx, ny)
    for i in range(nx):
        for j in range(ny):
            if i < nx/2:
                if 2*i * ny > j * nx:
                    p[i,j] = 1
            else:
                if (2*i - nx) * ny <= j * nx:
                    p[i,j] = 1
    return p

## generate some density from list of data points
n = 100
x = rand(n)       # cause
y = x + rand(n)   # effect
def hist(x, nx):
    # transform values into density
    x    = x.sort()
    minx = x[0]
    maxx = x[-1]
    hx   = zeros(nx)
    
    
def density(x, y, nx=100, ny=100):
    # transform a list of points into a density
    minx = x.min()
    maxx = x.max()
    



## row or column normalize
def normit(pxy, dim):
    ps = pxy.sum(dim)
    ps[ps==0.0] = 1.0    # do not normalize where there is no support
    if dim == 0:
        return pxy / ps
    elif dim == 1:
        return (pxy.T / ps).T
    else:
        raise("not implemented")


pxy  = simpleJoint(nx, ny)  # p(x,y)
px   = pxy.sum(0)           # p(x)
py   = pxy.sum(1)           # p(y)

pxgy = normit(pxy, dim=1)         # p(x|y)
pygx = normit(pxy, dim=0)         # p(y|x)
imshow(pxy)

## define an entropy on a two dim distribution
def log2(x):
    return log(x) / log(tensor([2.0]))
def entropy2(qxy, joint=None):
    if joint is None:
        qxy = qxy / qxy.sum()
        # after normalization there might be some Inf in there
        qxy[torch.isinf(qxy)] = 0.0
    lq = log2(qxy)
    lq[torch.isinf(lq)] = 0.0  # sets all nan to zero
    return -(lq * qxy).sum()

print(entropy2(pxy))
print(entropy2(pxgy))
print(entropy2(pygx))

