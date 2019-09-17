import datetime

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import label2rgb

from ..cluster import _focal_inner as focal
from ..utils.locs_hdfstore import ClustersHDFStore
from ..render.rendering import image_2d


def focal_clustering(xy, camera_format, camera_pixel, scale,
                     minL, min_area, show_plot):
    """
    Clustering using Focal analysis

    Parameters
    ----------
    xy : ndarray
        Localisation coordinates to be clustered
    camera_format : tuple
        Camera frame width, height.
    camera_pixel : float
        Size in nm of back projected pixel size.
    scale : int
        The scale at which to upsample the original image.
    minL : int
        Density threshold for clustering.
    min_area : int
        Retain clusters with area above this minimum value.

    Reference
    ---------
    A. Mazouchi et al. "Fast Optimized Cluster Algorithm for
    Localizations (FOCAL): a spatial cluster analysis for
    super-resolved microscopy", Bioinformatics, Volume 32, Issue 5,
    1 March 2016, Pages 747–754,
    https://doi.org/10.1093/bioinformatics/btv630
    """
    # create an image from the coordinates
    bins = [f * scale for f in camera_format]
    nm_scale = float(camera_pixel / scale)

    im = image_2d(xy[:, 0], xy[:, 1], bins[1], bins[0], nm_scale)
    
    # build the focal map, generate labels and
    # determine region region properties using 
    # skimage
    i_x = np.floor(xy[:, 0] / nm_scale).astype(np.int32)
    i_y = np.floor(xy[:, 1] / nm_scale).astype(np.int32)
    core_map = focal.density_map(i_x, i_y, im, minL)
    label_image = label(core_map)
    image_label_overlay = label2rgb(label_image, image=im)
    
    l = 0
    labels = np.full(xy.shape[0], -1, dtype=np.int32)
    core_samples_mask = np.full(xy.shape[0], 1, dtype=np.int32)
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= min_area**2:
            l += 1
            coords = region.coords
            indices = []
            for n in range(coords.shape[0]):                
                indices.append(np.where(
                    (i_y == coords[n, 0]) &
                    (i_x == coords[n, 1]))[0])

            indices = np.concatenate(indices)
            labels[indices] = l

    # any labels which are still -1 are not core samples
    core_samples_mask[labels == -1] = 0

    if show_plot:
        plt.figure()
        plt.imshow(im, interpolation='nearest', vmax=1)
        for m in np.unique(labels):
            if m != -1:
                cluster_points = xy[labels == m]
                plt.plot(
                    cluster_points[:, 0]/nm_scale,
                    cluster_points[:, 1]/nm_scale, 'rx')
        plt.show()        

    return (labels, core_samples_mask)


# need a streaming version of this to deal with massive datasets
def run_focal(fpath, camera_format=(256, 256),
              camera_pixel=160.0, scale=10, minL=9, min_area=3,
              show_plot=False):
    """
    Clustering using FOCAL analysis (see reference).

    Parameters
    ----------
    fpath : str
        Full path to the file being analysed.
    camera_format : tuple
        Camera frame width, height.
    camera_pixel : float
        Size in nm of back projected pixel size.
    scale : int
        The scale at which to upsample the original image.
    minL : int
        Density threshold for clustering.
    min_area : int
        Retain clusters with area above this minimum value.

    Reference
    ---------
    A. Mazouchi et al. "Fast Optimized Cluster Algorithm for
    Localizations (FOCAL): a spatial cluster analysis for
    super-resolved microscopy", Bioinformatics, Volume 32, Issue 5,
    1 March 2016, Pages 747–754,
    https://doi.org/10.1093/bioinformatics/btv630
    """
    with ClustersHDFStore(fpath) as ct:
        # get all xy coordinates out of hdf5 file
        index, xy = ct.get_points_for_clustering()

        labels, core_samples_mask = focal_clustering(
                xy, camera_format, camera_pixel, scale, minL, min_area, show_plot
        )

        # write clusters to hdf5
        now = datetime.datetime.now()
        ct_dict = {}
        ct_dict['clustering_method'] = 'focal'
        ct_dict['minL'] = minL
        ct_dict['min_area'] = min_area
        ct_dict['date_processed'] = now.strftime("%Y-%m-%d %H:%M")
        ct.add_clusters(index, labels, core_samples_mask, ct_dict)


if __name__ == "__main__":
    fpath = '.\\test_data\\ts.h5'
    source='thunderstorm'
    run_focal(fpath, minL=20, show_plot=True)