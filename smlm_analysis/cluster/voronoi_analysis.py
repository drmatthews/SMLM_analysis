"""
This is a modified version of code written by Hazen Babcock from the
Zhuang lab at Harvard Medical School.

This has been modified to work on localisations saved in a Pandas
DataFrame. The localisations are parsed using the ClustersTable class
in "localisations.py".

Dan 23/11/18
"""
import os
import datetime
import math
from collections import defaultdict
import itertools
import time

import numpy as np
import pandas as pd
from descartes import PolygonPatch
from matplotlib import pylab as plt
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
from scipy.spatial import Delaunay
from shapely.geometry import Polygon
from scipy.spatial import Delaunay
from scipy.stats import norm

from smlm_analysis.clustering._voronoi_inner import voronoi_inner
from smlm_analysis.utils.localisations import ClustersTable
from smlm_analysis.utils.plotting import plot_voronoi_diagram, plot_cluster_polygons
import smlm_analysis.utils.triangulation as tri


# required
# 1. refactor so that the methods work on arrays of (x,y) coordinates
# and have separate methods for reading the input data and writing results
# back to the data structure - done
# 2. some way of marking the data so that multiple passes
# through the clustering method can be made - done
# 3. a method for batch analysis - done
# 4. a collection of methods to calculate cluster statistics (separate module)
# 5. a method for saving clusters back to the hdf5 file - done see add_clusters method



def monte_carlo_threshold(points, confidence=99.0, iterations=100, show_plot=False):
    """
    Use a Monte Carlo simulation to determine the density level
    at which the input points are more clustered than a random
    distribution of points at the same density.

    Arguments
    ---------
    points (ndarray): 2d numpy array of (x,y) localisation coordinates
    confidence (float): confidence interval
    iterations (int): number of Monte Carlo iterations
    show_plot (bool): plot histrograms of the distribution of Voronoi polygon
    areas

    Returns
    -------
    Dict
    """
    locs_density = voronoi_density(points)
    locs_area = np.divide(1.0, locs_density)
    locs_area = locs_area[~np.isnan(locs_area)]

    z = norm.ppf(1 - float(100 - confidence) * 0.01 / 2.0)
    # the following is broken
    ####################
    density = locs.density
    area = locs.area
    width = locs.fov[0]
    height = locs.fov[1]
    #####################
    num_rand_locs = int(density * area)
    bins = round(2 * locs_area.shape[0]**(1/3))
    mc_counts = np.zeros((iterations, bins))

    for i in range(0, iterations):
        x = height * np.random.random((num_rand_locs, 1))
        y = width * np.random.random((num_rand_locs, 1))
        xy = pd.DataFrame(np.hstack((x, y)))
        xy.columns = ['x', 'y']
        rand_vor_density = voronoi_density(xy)
        area = np.divide(1.0, rand_vor_density)
        area = area[~np.isnan(area)]
        if i == 0:
            lim = 3 * np.median(area)
        c, _ = np.histogram(area, bins=bins, range=(0, lim))
        mc_counts[i, :] = c[:]

    counts, edges = np.histogram(locs_area, bins=bins, range=(0, lim))
    centers = (edges[:-1] + edges[1:]) / 2
    mean_mc_counts = np.mean(mc_counts, axis=0)
    std_mc_counts = np.std(mc_counts, axis=0)
    upper_mc_counts = np.add(mean_mc_counts, z * std_mc_counts)
    lower_mc_counts = np.subtract(mean_mc_counts, z * std_mc_counts)

    ind = np.where((counts - mean_mc_counts) < 0.0)[0][0]
    x1 = np.array((centers[ind - 1], centers[ind]))
    y1 = np.array((counts[ind - 1], counts[ind]))
    x2 = x1
    y2 = np.array((mean_mc_counts[ind - 1], mean_mc_counts[ind]))

    inters = tri.intersection(x1, y1, x2, y2)

    results = {}
    results['counts'] = counts
    results['upper_mc_counts'] = upper_mc_counts
    results['lower_mc_counts'] = lower_mc_counts
    results['intersection'] = inters

    if show_plot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(centers, counts, 'k-', label='Data')
        plt.plot(centers, mean_mc_counts, 'r-', label='MC mean')
        plt.plot(centers, upper_mc_counts, 'b-', label='MC mean + 2*std')
        plt.plot(centers, lower_mc_counts, 'b-', label='MC mean - 2*std')
        plt.plot(inters[0], inters[1], 'g*')
        ax.set_xlim([0, np.max(centers)])
        ax.set_ylim([0, np.max(counts) + 0.1 * np.max(counts)])
        ax.tick_params(labelsize='large')
        plt.xlabel('Polygon area [nm^2]', fontsize=18)
        plt.ylabel('Counts', fontsize=18)
        plt.legend()
        plt.show()        
    return results


def find_neighbours(points):
    """
    Determine the indices of the neighbouring Voronoi
    polygons for each input point.

    Arguments
    ---------
    points (ndarray): 2d numpy array of (x,y) localisation coordinates

    Return
    ------
    Tuple of ndarray of neighbours and neighbour_counts
    """
    tri = Delaunay(points)
    neighbour_list = defaultdict(set)
    neighbour_counts = np.zeros(points.shape[0])
    for p in tri.vertices:
        for i,j in itertools.combinations(p, 2):
            neighbour_list[i].add(j)
            neighbour_list[j].add(i)

    neighbours = np.zeros(points.shape[0], dtype='object') 
    for key, value in sorted(neighbour_list.items()):
        neighbours[key] = np.array(list(value), dtype='object')
        neighbour_counts[key] = len(neighbours[key])

    return (neighbours, neighbour_counts)


def voronoi_density(points):
    """
    Determine the local density for each Voronoi polygon for
    a set of localisation coordinates.

    Arguments
    ---------
    points (ndarray): 2d numpy array of (x,y) localisation coordinates

    Returns
    -------
    ndarray of density values for each localisation
    """
    n_locs = points.shape[0]
    vor = Voronoi(points)
    density = np.empty(n_locs)
    density[:] = np.nan
    for i, region_index in enumerate(vor.point_region):

        vertices = []
        for vertex in vor.regions[region_index]:

            if (vertex == -1):
                vertices = []
                break

            vertices.append(vor.vertices[vertex])

        if (len(vertices) > 0):
            density[i] = 1.0 / Polygon(vertices).area
    return density


def find_clusters(points, thresh_method='median', min_samples=5,
                  density_factor=0.001, show_plot=False):

    """
    Segments a set of localisation coordinates into clusters based
    on local density determined by forming a Voronoi tessellation.

    Arguments
    ---------
    points (ndarray): 2d numpy array of (x,y) localisation coordinates
    thresh_method (str): the method used to determine the density
    threshold for cluster formation
    density_factor (float): a multiplication factor used to refine
    the threshold
    min_samples (int): there should be at least this many
    localisations in a cluster
    show_plot (bool): plot localisation coordinates, Voronoi polygons
    and clusters

    Returns
    -------
    ndarray of cluster labels
    """

    n_locs = points.shape[0]
    nlist, _ = find_neighbours(points)
    density = voronoi_density(points)

    if thresh_method == 'median':
        thresh_density = np.median(density[~np.isnan(density)]) * density_factor
    elif thresh_method == 'monte_carlo':
        mc_data = monte_carlo_threshold(points, iterations=10)
        thresh_density = 1.0 / mc_data['intersection'][0]

    labels = np.full(n_locs, -1, dtype=np.intp)
    labels = voronoi_inner(n_locs, nlist, density, thresh_density, min_samples)

    if show_plot:
        vor = Voronoi(points)
        v2d = voronoi_plot_2d(vor, show_points=False, show_vertices=False)
        ax = v2d.axes[0]
        for m in np.unique(labels):
            if m != -1:
                cluster_points = points[labels == m]
                concave_hull, _ = (
                        tri.alpha_shape(cluster_points, alpha=0.00001)
                )
                patch = PolygonPatch(concave_hull, fc='#999999',
                                    ec='#000000', fill=True,
                                    zorder=-1)
                ax.add_patch(patch)

                ax.plot(cluster_points[:, 0], cluster_points[:, 1], 'rx')    
        plt.show()

    return labels

def run(locs_path, cluster_id='cluster_id', filter_by=None,
        thresh_method='median', min_samples=10,
        density_factor=0.01, show_plot=False):

    with ClustersTable(locs_path) as ct:
        index, xy = ct.get_cluster_points(filter_by=filter_by)
        labels = find_clusters(
            xy, thresh_method=thresh_method,
            min_samples=min_samples,
            density_factor=density_factor,
            show_plot=show_plot
        )
        now = datetime.datetime.now()
        ct_dict = {}
        ct_dict['clustering_method'] = 'voronoi'
        ct_dict['threshold_method'] = thresh_method
        ct_dict['min_samples'] = min_samples
        ct_dict['density_factor'] = density_factor
        ct_dict['date_processed'] = now.strftime("%Y-%m-%d %H:%M")
        ct.add_clusters(cluster_id, index, labels, ct_dict)

def batch(folder, cluster_id='cluster_id', filter_by=None,
          thresh_method='median', min_samples=10, density_factor=0.01):

    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            if filename.endswith("h5"):
                print("Running voronoi clustering on file {0}".format(filename))
                fpath = os.path.join(folder, filename)
                run(fpath, cluster_id=cluster_id, filter_by=filter_by,
                    thresh_method=thresh_method, min_samples=min_samples,
                    density_factor=density_factor)


if __name__ == '__main__':
    fpath = '.\\test_data\\ts_small.h5'
    source='thunderstorm'
    # locs = LocsTable(filepath=fpath, source=source)
    start = time.time()
    run(fpath, cluster_id='object_id', thresh_method='median',
        density_factor=0.01, show_plot=False)
    end = time.time()
    print(end - start)

    start = time.time()
    run(fpath, cluster_id='cluster_id', filter_by='object_id',
        thresh_method='median', density_factor=0.1, show_plot=False)
    end = time.time()
    print(end - start)
    # vor = Voronoi(locs.get_points())
    # voronoi_plot_2d(vor, show_points=True, show_vertices=False)
    # plt.show()

    # points=[[0.0, 0.0], [0.0, 1.0], [0.2, 0.5], [0.3, 0.6], [0.4, 0.5], [0.6, 0.3], [0.6, 0.5], [1.0, 0.0], [1.0, 1.0]]
    # nlist, ncounts = neighbours(points)
    # print(nlist[0])