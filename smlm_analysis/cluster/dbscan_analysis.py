import os
import datetime
import time

import numpy as np
import pandas as pd
import scipy.special as special
from sklearn import cluster
import hdbscan

from smlm_analysis.utils.locs_hdfstore import ClustersHDFStore
import smlm_analysis.utils.triangulation as tri


# def find_clusters(points, eps=None, min_samples=1, metric="euclidean"):
#     # Note to self
#     # as it stands this is unfinished
#     # partly adapted from original optics clustering method
#     # optics algorithm is not included in the current release of sklearn
#     # but is included on master branch
#     # be aware that sklearn DBSCAN has poor memory performance unless
#     # a bunch of optimisations (which I couldn't understand from an
#     # initial reading of the docs) are made
#     # there is a option of using HDBSCAN which is a separate package
#     # but is sklearn compatible
#     # HDBSCAN can be installed from conda-forge

#     optics_clusters = {}
#     noise = {}
#     for roi in points.keys():
#         xy = points[roi]

#         if xy.shape[0] == 0:
#             print("no coords in roi")
#             continue

#         if eps is None:
#             eps = _epsilon(xy, min_samples)

#         clusters = DBSCAN(eps=0.3, min_samples=10, metric=metric).fit(xy)

#         core_samples_mask = np.zeros_like(clusters.labels_, dtype=bool)
#         core_samples_mask[clusters.core_sample_indices_] = True
#         unique_labels = set(clusters.labels_)

#         cluster_list = ClusterList()
#         for m in unique_labels:

#             class_member_mask = (clusters.labels_ == m)
#             if m != -1:
#                 coords = xy[class_member_mask & core_samples_mask]
#                 if coords.shape[0] > 3:
#                     cluster_ = Cluster(coords, m)
#                     if cluster_.is_valid:
#                         cluster_list.append(Cluster(coords, m))
#                         optics_clusters[roi] = cluster_list
#             else:
#                 noise_ = xy[class_member_mask & ~core_samples_mask]
#                 cluster_list.noise = noise_
#                 noise[roi] = noise_

#     return (optics_clusters, noise)


def _epsilon(x, k):
    if len(x.shape) > 1:
        m, n = x.shape
    else:
        m = x.shape[0]
        n == 1
    prod = np.prod(x.max(axis=0) - x.min(axis=0))
    gamma = special.gamma(0.5 * n + 1)
    denom = (m * np.sqrt(np.pi**n))
    eps = ((prod * k * gamma) / denom)**(1.0 / n)

    return eps


def run_dbscan(locs_path, cluster_id='cluster_id',
               eps=None, min_samples=10, show_plot=False):

    with ClustersHDFStore(locs_path) as chdf:
        index, xy = chdf.get_points_for_clustering()

        if eps is None:
            eps = _epsilon(xy, min_samples)
        print(eps)
        db = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(xy)
        core_samples_mask = np.zeros_like(db.labels_, dtype=np.int32)
        core_samples_mask[db.core_sample_indices_] = 1
        labels = db.labels_

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        now = datetime.datetime.now()
        ct_dict = {}
        ct_dict['clustering_method'] = 'dbscan'
        ct_dict['min_samples'] = min_samples
        ct_dict['eps'] = eps
        ct_dict['date_processed'] = now.strftime("%Y-%m-%d %H:%M")
        chdf.add_clusters(cluster_id, index, labels, core_samples_mask, ct_dict)

    # if show_plot:
    #     unique_labels = set(labels)
    #     colors = [plt.cm.Spectral(each)
    #             for each in np.linspace(0, 1, len(unique_labels))]
    #     for k, col in zip(unique_labels, colors):
    #         if k == -1:
    #             # Black used for noise.
    #             col = [0, 0, 0, 1]

    #         class_member_mask = (labels == k)

    #         points = xy[class_member_mask & core_samples_mask]
    #         plt.plot(points[:, 0], points[:, 1], 'o', markerfacecolor=tuple(col),
    #                 markeredgecolor='k', markersize=8)

    #         points = xy[class_member_mask & ~core_samples_mask]
    #         plt.plot(points[:, 0], points[:, 1], 'o', markerfacecolor=tuple(col),
    #                 markeredgecolor='k', markersize=4)

    #     plt.title('Estimated number of clusters: %d' % n_clusters_)
    #     plt.show()


def batch_dbscan(folder, cluster_id='cluster_id', filter_by=None, min_samples=10, eps=0.5):
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            if filename.endswith("h5"):
                print("Running voronoi clustering on file {0}".format(filename))
                fpath = os.path.join(folder, filename)
                run_dbscan(fpath, cluster_id=cluster_id,
                    filter_by=filter_by, eps=eps, min_samples=min_samples)


def run_hdbscan(locs_path, cluster_id='cluster_id', filter_by=None,
                min_cluster_size=10, min_samples=10, show_plot=False):

    with ClustersHDFStore(locs_path) as chdf:
        index, xy = chdf.get_cluster_points(filter_by=filter_by)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples).fit(xy)
        labels = clusterer.labels_
        probabilities = clusterer.probabilities_            

        now = datetime.datetime.now()
        ct_dict = {}
        ct_dict['clustering_method'] = 'hdbscan'
        ct_dict['min_samples'] = min_samples
        ct_dict['date_processed'] = now.strftime("%Y-%m-%d %H:%M")
        chdf.add_clusters(cluster_id, index, labels, ct_dict)

    # if show_plot:
    #     plt.figure()
    #     color_palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    #     cluster_colors = [color_palette[x] if x >= 0
    #                     else (0.5, 0.5, 0.5)
    #                     for x in labels]
    #     cluster_member_colors = [sns.desaturate(x, p) for x, p in
    #                             zip(cluster_colors, probabilities)]
    #     plt.scatter(xy[:,0], xy[:,1], s=50, linewidth=0, c=cluster_member_colors, alpha=0.25)
    #     plt.show()


def batch_hdbscan(folder, cluster_id='cluster_id', filter_by=None,
                  min_cluster_size=10, min_samples=10):
    if os.path.isdir(folder):
        for filename in os.listdir(folder):
            if filename.endswith("h5"):
                print("Running voronoi clustering on file {0}".format(filename))
                fpath = os.path.join(folder, filename)
                run_hdbscan(fpath, cluster_id=cluster_id,
                            filter_by=filter_by,
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples)


if __name__ == '__main__':
    fpath = '.\\test_data\\ts_small.h5'
    source='thunderstorm'
    # locs = LocsTable(filepath=fpath, source=source)
    # start = time.time()
    # run_hdbscan(fpath, cluster_id='object_id', min_cluster_size=20, min_samples=20, show_plot=True)
    # end = time.time()
    # print(end - start)

    start = time.time()
    run_dbscan(fpath, cluster_id='object_id', min_samples=10)
    end = time.time()
    print(end - start)