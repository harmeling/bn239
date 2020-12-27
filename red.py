# Peyman Milanfar's talk
# https://stats.hohoweiya.xyz/2019/03/11/Tweedie-and-selection-bias/
#
# Red for inverse problems
# try `minx (H*x-y)^2 + lambda/2 x'*(x-f(x))`
#
# The insight here is that an image that equals its denoised version is a real image and not a garbage image.
#
# FIRST idea
# For phase retrieval we might be able to rewrite `H*x-y` using some tricks, e.g.
# * use Cos for the rows of H (use phases of x for those Cos), this should calculate some correlations
#   and should correspond to the correct magnitude, once the phases are correct
# * updating x also updates the phases of the Cos rows of H
# * use a denoising network that works well on the problem domain
# * the denoising for the molecules must als be tested on images
# * y being magnitude, or use magnitude to weigh the more important frequencies
# OR
# * use 
#
# SECOND idea:
# * replace H*x-y by abs(F*x)-y, this looses some of the niceties, but is more correct
#   wrt to phase retrieval
#
# THIRD idea use
# * new variant of HIO/Gerchberg/Saxton:
#

