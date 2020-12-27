## phase_align.py
#
# author: Stefan Harmeling
# date:   2020-11-21
#
# Goal: how to measure phase alignment
# 1. phase alignments might help to get rid of local optima
# 2. what should we optimize?  non-gaussianity?  entropy?
#

from lab import *
from progress.bar import Bar

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

def edge_fft(n, i):
    # show the fft of an edge
    x = zeros(n)
    x[i:] = 1.0     # create an edge
    X = fftn(x)

def energy(x):
    # calculates the overall signal energy
    return (x*x).sum()
    
def en_cos(p):  # trash
    x = arange(1000);
    return energy(cos(2*pi*x/20) + cos(2*pi*(x+p)/22))

def calc_aligns(n, f1, f2):
    x = arange(n)
    y1 = cos((2*pi*(x/n)) * f1)
    y2 = cos((2*pi*(x/n)) * f2)
    # next move the phases and calculate a curve for it
    
def ituc_slow(X, maxK=None, doplot=False):
    "inverse discrete Fourier transform using cosines"
    if X.dim() != 2:
        raise NotImplementedError('so far only for two dimensions')
    N1, N2 = X.shape
    K1, K2 = (N1, N2) if maxK is None else maxK
    n1 = arange(N1).unsqueeze(1)    # a 1 x N1  matrix
    n2 = arange(N2).unsqueeze(0)    # a N2 x 1  matrix
    m  = X.abs()    # magnitudes
    p  = X.angle()  # phases
    bar = Bar('ituc_slow', max=N1)
    if doverb: fig = plt.figure(); ax = fig.add_subplot(111)
    x = zeros(N1, N2)
    for k1 in range(K1):
        if doverb: ax.clear(); ax.imshow(x); pause(0.1)
        for k2 in range(K2):
            x += m[k1,k2] * cos(2*pi*k1*n1/N1 + 2*pi*k2*n2/N2 + p[k1,k2])
        bar.next()
    bar.finish()
    x = x / K1 / K2
    return x
            
def ituc(X):
    "fast inverse discrete Fourier transform using cosines"
    allcosines = tucall(X)
    K1, K2 = allcosines.shape[2:4]
    x = allcosines.sum(dim=(2,3)) / K1 / K2
    return x
            
def tucall(X, maxK=None):
    "calculate all cosines"
    if X.dim() == 1:
        N1 = X.shape[0]
        K1 = N1 if maxK is None else maxK[0]
        print(K1)
        m  = X[:K1].abs().reshape(1, K1)    # magnitudes
        p  = X[:K1].angle().reshape(1, K1)  # phases
        n1 = arange(N1).reshape(N1, 1)
        k1 = arange(K1).reshape(1, K1)
        angles = k1*n1/N1  # shape == (N1, K1)
    elif X.dim() == 2:
        N1, N2 = X.shape
        K1, K2 = (N1, N2) if maxK is None else maxK
        m  = X[:K1,:K2].abs().reshape(1, 1, K1, K2)    # magnitudes
        p  = X[:K1,:K2].angle().reshape(1, 1, K1, K2)  # phases
        n1 = arange(N1).reshape(N1, 1, 1, 1)
        n2 = arange(N2).reshape(1, N2, 1, 1)
        k1 = arange(K1).reshape(1, 1, K1, 1)
        k2 = arange(K2).reshape(1, 1, 1, K2)
        angles = k1*n1/N1 + k2*n2/N2   # shape == (N1, N2, K1, K2)
    else:
        raise NotImplementedError('so far only for one and two dimensions')
    # we use: cos(alpha+beta) == cos(alpha)*cos(beta)-sin(alpha)*sin(beta)
    cosalpha = cos(2*pi*angles)
    sinalpha = sin(2*pi*angles)
    cosbeta  = cos(p)
    sinbeta  = sin(p)
    allcosines = m * (cosalpha * cosbeta - sinalpha * sinbeta)
    return allcosines

def tuccor1(X, maxK=None):
    if X.dim() != 1:
        raise NotImplementedError('implemented only for one dimensions')
    allcosines = tucall(X, maxK)
    m = X.abs()
    K1 = X.shape[0] if maxK is None else maxK
    allenergies = zeros(K1, K1)
    bar = Bar('tuccor1', max=K1)
    for k1 in range(K1):
        for l1 in range(K1):
            z = (allcosines[:,k1] + allcosines[:,l1]) / (m[k1]*m[l1])
            allenergies[k1,l1] = energy(z)
        bar.next()
    bar.finish()
    return allenergies


def tuccor2(X, maxK=None):
    if X.dim() != 2:
        raise NotImplementedError('so far only for one and two dimensions')
    allcosines = tucall(X, maxK)
    m = X.abs()
    K1, K2 = X.shape if maxK is None else maxK
    allenergies = zeros(K1, K2, K1, K2)
    bar = Bar('tuccor2', max=K1*K2)
    for k1 in range(K1):
        for k2 in range(K2):
            for l1 in range(K1):
                for l2 in range(K2):
                    z = (allcosines[:,:,k1,k2] + allcosines[:,:,l1,l2]) / (m[k1,k2]*m[l1,l2])
                    allenergies[k1,k2,l1,l2] = energy(z)
            bar.next()
    bar.finish()
    return allenergies

def mixfft(x, y):
    X = fftn(x)
    Y = fftn(y)
    xy = ifftn(X.abs() * exp(1j * Y.angle())).real
    yx = ifftn(Y.abs() * exp(1j * X.angle())).real
    return xy, yx



def distort_phase(x, alpha, y=None):
    """Add some alien phase.

    x     -- a tensor
    alpha -- a weight between 0 and 1
    y     -- another tensor, default rand(x.shape)
    alpha == 0 has no effect.
    """
    if not(0.0 <= alpha <= 1.0):
        raise RuntimeError('alpha must be between 0 and 1.')
    if y is None: y = rand(x.shape)
    X = fftn(x)
    Y = fftn(y)
    mixed_angle = (1-alpha)*X.angle() + alpha*Y.angle()
    return ifftn(X.abs() * exp(1j * mixed_angle)).real

def norm_tensor(x):
    """Remove the mean and set std to one."""
    return (x - x.mean()) / x.std()   # normalize it
    
## functions for image entropy
# https://stats.stackexchange.com/questions/235270/entropy-of-an-image
# https://www.tandfonline.com/doi/abs/10.1080/713821475?journalCode=tmop19
# https://arxiv.org/abs/1609.01117
# let's do the following:
# * calculate horizontal and vertical gradients
# * calculate smooth 2d histogram and normalize
# * calculate discrete entropy of the 2d histogram
def gaussian_kernel(x,y,h=0.1):
    """Compute the Gaussian kernel matrix for two data sets.
    
    x -- a data matrix of shape (d,nx)
    y -- another data matrix of shape (d,ny), defaults to x
    h -- kernel width of Gaussian kernel
    output matrix will have shape (nx, ny)
    """
    if y is None: y = x
    if x.dim()!=2 or y.dim()!=2:
        raise RuntimeError('2D tensors expected')
    # use (xi-yj)^2 = dot(xi,xi) + dot(yj,yj) - 2*dot(xi,yj)
    D = (x*x).sum(0).reshape(-1,1) + (y*y).sum(0).reshape(1,-1)
    D -= 2*matmul(x.T,y)
    return exp(-D/h)

def smooth_histogram1(x, nbin=100, h=0.01, lims=(0.0,1.0)):
    """Compute 1D histogram based on kernel density estimate.

    x    -- a 1D tensor of scalars
    nbin -- the number of kernels (aka bins), default 100
    h    -- the width of the Gaussian kernel, default 0.1
    lims -- 2-tuple of limits, default (0.0, 1.0)
    """
    if x.dim()!=1:
        raise RuntimeError('1D tensor expected')
    x = x.reshape(1,-1)                # turn it into a matrix
    z = linspace(lims[0],lims[1],nbin)      # kernel locations
    K = gaussian_kernel(x, z.view(1,-1), h)  # a kernel matrix
    p = K.sum(0)                     # sum out the data points
    p = p / p.sum()                             # normalize it
    return (p, z)

def smooth_histogram2(x, nbin=100, h=0.01, lims=(0.0,1.0)):
    """Compute 2D histogram based on kernel density estimate.

    x    -- a 2D tensor of shape (2,nx)
    nbin -- the number of kernels (aka bins), default 100
    h    -- the width of the Gaussian kernel, default 0.1
    lims -- 2-tuple of limits, default (0.0, 1.0)
    """
    if x.dim()!=2:
        raise RuntimeError('2D tensor expected')
    z = linspace(lims[0],lims[1],nbin)         # row vector
    z1, z2 = meshgrid(z, z)
    zz = vstack((z1.flatten(), z2.flatten())) # (2,nbin*nbin)
    K = gaussian_kernel(x, zz, h)           # a kernel matrix
    p = K.sum(0)                   # sum out the data points
    p = p.reshape(nbin,nbin) / p.sum()        # normalize it
    return (p, z)

def entropy(p):
    """Compute the entropy of a discrete distribution."""
    plogp = p * log(p)            # for p=0.0 this is nan
    plogp[plogp.isnan()] = 0.0          # set nan to zero
    return -plogp.sum()
    
def entropy2(x, nbin=100):
    """Estimate some reasonable image entropy.

    We assume that the pixel values are between 0 and 1.
    """
    if x.dim()!=2: raise RuntimeError('2D tensors expected')
    hx = x - x.roll(1,0)   # horizontal gradient
    vx = x - x.roll(1,1)   # vertical gradient
    hvx = vstack((hx.flatten(), vx.flatten()))
    lims = (-0.3, 0.3)    # should be symmetric
    h = 0.0001
    (p,z) = smooth_histogram2(hvx, nbin=nbin, h=h, lims=lims)
    return (entropy(p), p, z)

def gradient_var(x):
    """Estimate variance of gradient pixels."""
    if x.dim()!=2: raise RuntimeError('2D tensors expected')
    hx = x - x.roll(1,0)   # horizontal gradient
    vx = x - x.roll(1,1)   # vertical gradient
    hvx = vstack((hx.flatten(), vx.flatten()))
    return hvx

def frobenius_norm(x):
    return (x*x).sum()

def negentropy_hyvarinen(x):
    """Estimate negentropy like Hyvarinen."""
    z = randn(x.shape)  # a Gaussian sample of same size
    x = (x-x.mean())/x.std()       # remove mean and std
    # any non-squared function should do the job
    #G1 = lambda u: log(cosh(u))  # works ok
    G2 = lambda u: -exp(-u*u)   # WORKS BEST!
    #G3 = lambda u: u*u*u*u   # not working well
    G = G2
    return frobenius_norm(G(x) - G(z))
    
def kurt(x):
    """Estimate kurtosis.

    Based on the formula:
    kurt(x) = E(x^4) - 3(E(x^2))^2"""
    n = x.numel()
    return (x**4).sum()/n - 3*((x**2).sum()/n)**2

## negentropy_hyvarinen works!
#x = cameraman()[::2,::2]
x = boat()
y = barbara()
for alpha in linspace(0.0, 1.0, 11):
    hvx = gradient_var(distort_phase(x, alpha,y))
    print(alpha, negentropy_hyvarinen(hvx))
    
## kurtosis:
x = cameraman()[::2,::2]
y = barbara()[::4,::4]
for alpha in linspace(0.0, 1.0, 11):
    hvx = gradient_var(distort_phase(y,alpha,x))
    print(alpha, kurt(hvx))


## ok looks like std on gradient_var does not work!
x = cameraman()[::2,::2]
y = barbara()[::4,::4]
for alpha in linspace(0.0, 1.0, 11):
    print(alpha, gradient_var(distort_phase(x,alpha,y)).std())
# surprisingly it is constant!

## try third order moment
x = cameraman()[::2,::2]
y = barbara()[::4,::4]
for alpha in linspace(0.0, 1.0, 11):
    hvx = gradient_var(distort_phase(x,alpha,y))
    print(alpha, (hvx**4).sum())

    
## ok looks like entropy2 works!
x = cameraman()[::2,::2]
y = barbara()[::4,::4]
for alpha in linspace(0.0, 1.0, 11):
    print(alpha, entropy2(distort_phase(y,alpha,x),nbin=30)[0])

## test
(e,p,z) = entropy2(distort_phase(y, 0.0, x), nbin=17)
pcolor(z,z,p,shading='nearest')

## script to test stuff
x = cameraman()
y = barbara()[::2,::2]
K1, K2 = 4, 4
xy, yx = mixfft(x, y)
maxK = (8,8)
ex = tuccor2(fftn(x), maxK)
ey = tuccor2(fftn(y), maxK)
exy = tuccor2(fftn(xy), maxK)
eyx = tuccor2(fftn(yx), maxK)
clf()
pl(log( ex.reshape(64,64)), 221, 'x')
pl(log(eyx.reshape(64,64)), 222, 'yx')
pl(log(exy.reshape(64,64)), 223, 'xy')
pl(log( ey.reshape(64,64)), 224, 'y')

## script to run on mnist
x = mnist()[0]   # get a digit
y = mnist()[1]   # get another digit
xy, yx = mixfft(x, y)
ex = tuccor2(fftn(x))
ey = tuccor2(fftn(y))
exy = tuccor2(fftn(xy))
eyx = tuccor2(fftn(yx))
pl()
pl(log( ex.reshape(784,784)), 221, 'x')
pl(log(eyx.reshape(784,784)), 222, 'yx')
pl(log(exy.reshape(784,784)), 223, 'xy')
pl(log( ey.reshape(784,784)), 224, 'y')

## script to redo the experiments from mathematica
nn = 128;
def f(m1, m2, p1, p2, k1, k2, n):
    return m1*cos(2*pi*k1*n/nn+p1) + m2*cos(2*pi*k2*n/nn+p2)
((f(2,1.2,1.8,1.5,12,15, arange(128))**2).sum())


