    # public void buildRing() {
    #     xRingCoordinates0 = new float[nRingCoordinates];
    #     yRingCoordinates0 = new float[nRingCoordinates];
    #     xRingCoordinates1 = new float[nRingCoordinates];
    #     yRingCoordinates1 = new float[nRingCoordinates];
    #     float angleStep = (PI * 2f) / (float) nRingCoordinates;
    #     for(int angleIter = 0; angleIter < nRingCoordinates; angleIter++){
    #         xRingCoordinates0[angleIter] = spatialRadius * cos(angleStep * angleIter);
    #         yRingCoordinates0[angleIter] = spatialRadius * sin(angleStep * angleIter);
    #         xRingCoordinates1[angleIter] = gradRadius * cos(angleStep * angleIter);
    #         yRingCoordinates1[angleIter] = gradRadius * sin(angleStep * angleIter);
    #     }
    # }

from scipy.ndimage.filters import convolve
from scipy import interpolate
from matplotlib import pyplot as plt
import numpy as np

import pims
from nd2reader import ND2Reader


def gradientXY(im, kx, ky):
    gx = convolve(im, kx, mode='wrap')
    gy = convolve(im, ky, mode='wrap')
    return (gx, gy)

def interpolate_gradient(grad, x_pos, y_pos, magn=5):
    size_y, size_x = grad.shape
    x, y = np.mgrid[0:size_x, 0:size_y]
    xnew, ynew = np.mgrid[0:(size_x * magn), 0:(size_y * magn)]
    tck = interpolate.bisplrep(x, y, grad, s=0)
    znew = interpolate.bisplev(xnew[:,0], ynew[0,:], tck)
    return znew

def run(movie_path, frame=0):
    m = pims.open(movie_path)
    im = m[frame]

    kx = np.ones((3, 5))
    kx[:, 2] = 0
    kx[:, 0:2] = -1
    ky = kx.T

    gx, gy = gradientXY(im, kx, ky)

    interp = interpolate_gradient(g, 0, 0)

    plt.figure()
    plt.imshow(g)
    
    # plt.figure()
    # plt.imshow(interp)

    plt.show()


if __name__=='__main__':
    run('./test_data/tubulin647 2d.nd2', frame=10000)