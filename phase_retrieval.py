## phase_retrieval.py
#
# author: Stefan Harmeling
# date:   2020-12-08
#
# Goal: recover the phase

from lab import *
from phase_align import *
from progress.bar import Bar

## setup
images = [barbara,
          boat,
          cameraman,
          fingerprint,
          house,
          lena,
          peppers256]
x = images[0]()    # load an image
X = fftn(x)
magnitude = X.abs()
phase     = X.angle()

## recover the phase from the magnitudes
# plan: maximize negentropy_hyvarinen

## problem: exp and ifftn not working with autograd yet
# so rewrite them...

# for any real number x
euler = lambda x: cos(x) + 1j*sin(x)  # == exp(1j * x)

# inverse Fourier transform using cosines
def isftn_cos(m, p):
    N1, N2 = m.shape
    K1, K2 = N1, N2
    m  = m.reshape(1, 1, K1, K2)    # magnitudes
    p  = p.reshape(1, 1, K1, K2)  # phases
    n1 = arange(N1).reshape(N1, 1, 1, 1)
    n2 = arange(N2).reshape(1, N2, 1, 1)
    k1 = arange(K1).reshape(1, 1, K1, 1)
    k2 = arange(K2).reshape(1, 1, 1, K2)
    angles = k1*n1/N1 + k2*n2/N2   # shape == (N1, N2, K1, K2)
    # we use: cos(alpha+beta) == cos(alpha)*cos(beta)-sin(alpha)*sin(beta)
    cosalpha = cos(2*pi*angles)
    sinalpha = sin(2*pi*angles)
    cosbeta  = cos(p)
    sinbeta  = sin(p)
    allcosines = m * (cosalpha * cosbeta - sinalpha * sinbeta)
    return allcosines.sum(dim=(2,3))

def ifftn_cos(m, p, maxK=None):
    "inverse discrete Fourier transform using cosines"
    if X.dim() != 2:
        raise NotImplementedError('so far only for two dimensions')
    N1, N2 = X.shape
    K1, K2 = (N1, N2) if maxK is None else maxK
    n1 = arange(N1).unsqueeze(1)    # a 1 x N1  matrix
    n2 = arange(N2).unsqueeze(0)    # a N2 x 1  matrix
    bar = Bar('ifftn_cos', max=K1)
    x = zeros(N1, N2)
    for k1 in range(K1):
        for k2 in range(K2):
            x += m[k1,k2] * cos(2*pi*k1*n1/N1 + 2*pi*k2*n2/N2 + p[k1,k2])
        bar.next()
    bar.finish()
    x = x / K1 / K2
    return x

## first use easy starting points
alpha = 0.1
phase_hat = fftn(distort_phase(x, alpha)).angle()   # initial phases
phase_hat.requires_grad = True
learning_rate = 1e-8
for i in range(10):
    #x_hat = ifftn(magnitude * euler(phase_hat)).real
    x_hat = ifftn_cos(magnitude, phase_hat)
    loss = negentropy_hyvarinen(x_hat)
    loss.backward()
    with torch.no_grad():
        phase_hat += learning_rate * phase_hat.grad
        phase.grad = None
    print(loss)


## first use easy starting points
from torch.optim import SGD
x = barbara()[::8,::8]
alpha = 0.1
phase_hat = fftn(distort_phase(x, alpha)).angle()   # initial phases
phase_hat.requires_grad = True
learning_rate = 1e-8
opt = SGD([phase_hat], learning_rate)
opt.zero_grad()
for i in range(10):
    #x_hat = ifftn(magnitude * euler(phase_hat)).real
    x_hat = ifftn_cos(magnitude, phase_hat)
    loss = -negentropy_hyvarinen(x_hat)
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(loss)
