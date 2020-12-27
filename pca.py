## pca.py
#
# author: Stefan Harmeling
# date:   2020-12-01
#
# Goal: how to measure phase alignment

from lab import *

## load mnist data
X = mnist()
print(X.shape)
pl(X[0], 221, clear=True)
pl(X[1], 222)
pl(X[2], 223)
pl(X[3], 224)

## remove mean
N = X.shape[0]
Xm = X.sum(0)/N
pl(Xm, clear=True)
X  = X - Xm

## calculate covariance matrix
C = mm(X.view(N, -1).T, X.view(N, -1))
eig
