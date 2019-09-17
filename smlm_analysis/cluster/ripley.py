from astropy.stats import RipleysKEstimator
import numpy as np

from ..utils.locs_hdfstore import ClustersHDFStore


def ripley_function(locs_path, max_dist=200, method='ripley'):
    """
    Wrapper around astropy RipleyKEstimator class
    """
    with ClustersHDFStore(locs_path) as ct:
        _, xy = ct.get_points_for_clustering()
        x_min = np.min(xy[:, 0])
        x_max = np.max(xy[:, 0])
        y_min = np.min(xy[:, 1])
        y_max = np.min(xy[:, 1])
        area = (x_max - x_min) * (y_max - y_min)
        kest = RipleysKEstimator(
            area, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max
        )
        radii = np.linspace(0, max_dist, 100)
        rip = kest.Hfunction(xy, radii, mode=method)    
    return rip


def ripley_ci(locs_path, num_simulations=100, max_dist=200, method='ripley'):
    """
    Use Monte Carlo to estimate the confidence interval
    """
    with ClustersHDFStore(locs_path) as ct:
        _, xy = ct.get_points_for_clustering()
        n_locs = xy.shape[0]
        x_min = np.min(xy[:, 0])
        x_max = np.max(xy[:, 0])
        y_min = np.min(xy[:, 1])
        y_max = np.min(xy[:, 1])
        area = (x_max - x_min) * (y_max - y_min)
        box = [x_min, x_max, y_min, y_max]
        dist_scale = np.linspace(0, max_dist, 100)
        lrand = np.zeros((dist_scale.shape[0], num_simulations))
        kest = RipleysKEstimator(
            area, x_min=box[0], x_max=box[1], y_min=box[2], y_max=box[3]
        )
        for s in range(num_simulations):
            rand_datax = np.random.uniform(box[0], box[1], n_locs)
            rand_datay = np.random.uniform(box[2], box[3], n_locs)
            rand_xy = np.stack((rand_datax.T, rand_datay.T), axis=-1)
            lrand[:, s] = kest.Hfunction(rand_xy, dist_scale, mode=method)

        meanl = np.mean(lrand, axis=1)
        stdl = np.std(lrand, axis=1)
        ci_plus = meanl + 2*stdl
        ci_minus = meanl - 2*stdl

    return (meanl, ci_plus, ci_minus)