import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.color import label2rgb

from smlm_analysis.clustering import _focal_inner as focal
from smlm_analysis.utils.localisations import ClustersTable
from smlm_analysis.utils.rendering import hist_2d


# need a streaming version of this to deal with massive datasets
def run_focal(fpath, filter_by=None, camera_format=(256, 256),
              camera_pixel=160.0, scale=10, block_shape=(4, 4), minL=9):
    # should use LocsTable class since it requires building an
    # image from all the coordinates
    with ClustersTable(fpath) as ct:
        _, xy = ct.get_cluster_points(filter_by=filter_by)

        bins = [f * scale for f in camera_format] # num pixels in reconstructed image
        nm_scale = float(camera_pixel / scale)

        i_x = np.floor(xy[:, 0]/nm_scale).astype(np.int32)
        i_y = np.floor(xy[:, 1]/nm_scale).astype(np.int32)
        n = xy.shape[0]
        im = np.zeros(bins, dtype=np.int32)
        hist_2d(i_x, i_y, n, im)
        
        core_map = focal.density_map(i_x, i_y, im, minL)
        label_image = label(core_map)
        image_label_overlay = label2rgb(label_image, image=im)
        
        l = 0
        labels = np.full(n, -1, dtype=np.int32)
        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 3*3+1: # this should be an input
                l += 1
                coords = region.coords
                indices = []
                for n in range(coords.shape[0]):                
                    indices.append(np.where(
                        (i_y == coords[n, 0]) &
                        (i_x == coords[n, 1]))[0])

                indices = np.concatenate(indices)
                labels[indices] = l

        # this uses the image so won't return a cluster id
        # for each localisation, just an overall number of
        # clusters found - maybe save centroids, area, pixel
        # coordinates? use regionprops

        # now = datetime.datetime.now()
        # ct_dict = {}
        # ct_dict['clustering_method'] = 'hdbscan'
        # ct_dict['min_samples'] = min_samples
        # ct_dict['date_processed'] = now.strftime("%Y-%m-%d %H:%M")
        # ct.add_clusters(cluster_id, index, labels, ct_dict)

    plt.figure()
    plt.imshow(im, interpolation='nearest', vmax=1)
    # for m in np.unique(labels):
    #     if m != -1:
    #         cluster_points = xy[labels == m]
    #         plt.plot(cluster_points[:, 0]/nm_scale, cluster_points[:, 1]/nm_scale, 'rx')
    plt.plot(xy[:, 0]/nm_scale, xy[:, 1]/nm_scale, 'rx')
    plt.show()

if __name__ == "__main__":
    fpath = '.\\test_data\\ts.h5'
    source='thunderstorm'
    run_focal(fpath, minL=20)