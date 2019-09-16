import numpy as np
from sklearn.neighbors import NearestNeighbors
from skimage import transform as tf
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift


def nearest_neighbours(pointsA, pointsB, col=0):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(pointsA)
    distances, indices = nbrs.kneighbors(pointsB)
    return distances[:, col]

def pad(coords):
    return np.hstack([coords, np.ones((coords.shape[0], 1))])
    
def unpad(coords):
    return coords[:,:-1]
    
def calc_transform(unreg, base):
    tform = tf.AffineTransform()
    tform.estimate(unreg, base)
    A = tform.params.copy()
    # reg = tform.inverse(unreg)
    # A, reg, rank, s = np.linalg.lstsq(pad(unreg), pad(base))
    A[np.abs(A) < 1e-10] = 0  # set really small values to zero
    # return (base, reg, A)
    return tform
    
# def do_transform(coords,A):
#     return unpad(np.dot(pad(coords), A))

def inverse_transform(coords, tform):
    return tform.inverse(coords)


def dist_search(A, B, dist_thresh=5):
    nn_dist = nearest_neighbours(A.T, B.T)
    idx = nn_dist < dist_thresh
    return idx

def warp_image(A, image):
    pass


def apply_shift(shift, image):
    offset_image = fourier_shift(np.fft.fftn(image), shift)
    offset_image = np.fft.ifftn(offset_image)
    return offset_image

def correlation_registration(src, target, factor=100):
    shift, error, diffphase = register_translation(src, target, upsample_factor=factor)
    return shift

    
if __name__=='__main__':
    import pims
    from nd2reader import ND2Reader

    m = pims.open('.\\test_data\\tetra_speck_beads_time_lapse.nd2')
    m.bundle_axes = 'cxy'
    im = m[0]
    print(im.shape)
    print(correlation_registration(im[0, :, :], im[1,:, :]))

    from matplotlib import pyplot as plt

    plt.figure()
    plt.imshow(im)
    # dst = np.array([[164.13039759598942, 138.51766442481798, 
    #                      78.089917747885394, 128.87601196416392, 
    #                      139.37950024523425, 77.761806849012856, 
    #                      167.39469512765191, 147.09465734762969, 
    #                      96.512258538802669, 88.666231753399572, 
    #                      38.051902605410596, 51.298347671406489], 
    #                      [39.183043028867175, 59.086259543442516, 
    #                       71.895008549258463, 130.67537915751086, 
    #                       164.16551437168738, 167.3162790741014, 
    #                       115.48545081050807, 84.927624224523186, 
    #                       110.39112844517994, 103.28833125750975, 
    #                       37.40010560414855, 66.60282609190854]]).T
    # src = np.zeros((dst.shape[0], 2))
    # src[:, 0] = dst[:, 0] + 10
    # src[:, 1] = dst[:, 1] + 20
    
    # # base, reg, A = calc_transform(primary, secondary)
    # tform = calc_transform(src, dst)
    
    # print("Target:")
    # print(src)
    # print("Result:")
    # print(inverse_transform(dst, tform))
    # print("Max error: {}".format(np.abs(src - inverse_transform(dst, tform)).max()))
    # print("transform:")
    # print(tform.params)