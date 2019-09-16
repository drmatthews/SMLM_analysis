import os
import math
import numpy as np

from scipy.spatial import voronoi_plot_2d
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, MultiPoint, MultiLineString
from shapely.ops import cascaded_union, polygonize
from descartes import PolygonPatch
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from read_roi import read_roi_file
from read_roi import read_roi_zip

import smlm_analysis.utils.triangulation as tri


def plot_clusters_in_roi(roi_path,
                         pixel_size,
                         cluster_list,
                         noise=None,
                         save=False,
                         filename=None,
                         show_plot=True,
                         new_figure=True,
                         cluster_marker_size=4):

    filename, ext = os.path.splitext(roi_path)
    print(ext)
    if 'zip' in ext:
        rois = read_roi_zip(roi_path)
    else:
        rois = read_roi_file(roi_path)

    for roi_id, roi in rois.items():
        for k, v in roi.items():
            if not isinstance(v, str):
                roi[k] = float(v) * pixel_size

        plot_optics_clusters(
                cluster_list,
                noise=noise,
                save=False,
                filename=None,
                show_plot=True,
                new_figure=True,
                cluster_marker_size=4,
                roi=roi)


def plot_optics_clusters(cluster_list,
                         noise=None,
                         save=False,
                         filename=None,
                         show_plot=True,
                         new_figure=True,
                         cluster_marker_size=4,
                         roi=None):

    if new_figure:
        plt.figure()

    colors = plt.cm.Spectral(np.linspace(0, 1, cluster_list.n_clusters))
    num_clusters = cluster_list.n_clusters
    if roi is not None:
        num_clusters = 0

    for c, color in zip(cluster_list, colors):
        if roi is not None:
            # plot clusters and annotate centroids
            coords = np.hstack((c.x, c.y))
            coords = coords[(coords[:, 0] > roi['left']) &
                            (coords[:, 0] < roi['left'] + roi['width']) &
                            (coords[:, 1] > roi['top']) &
                            (coords[:, 1] < roi['top'] + roi['height'])]
            if coords.shape[0] > 0:
                num_clusters += 1
                x = coords[:, 0]
                y = coords[:, 1]
            else:
                continue
        else:
            x = c.x
            y = c.y

        plt.plot(x, y, 'o',
                 markerfacecolor=color,
                 markeredgecolor='k',
                 markersize=cluster_marker_size,
                 alpha=0.5)

        plt.annotate('%s' % str(c.cluster_id),
                     xy=(c.center[0:2]),
                     xycoords='data')

    # plot noise
    if noise is not None:
        plt.plot(noise[:, 0], noise[:, 1], 'o', markerfacecolor='k',
                 markeredgecolor='k', markersize=1, alpha=0.5)

    plt.title('Estimated number of clusters: %d' % num_clusters)
    plt.xlabel('X [nm]')
    plt.ylabel('Y [nm]')
    plt.gca().invert_yaxis()

    if save and filename is not None:
        plt.savefig(filename)

    if show_plot:
        plt.show()


def plot_voronoi_clusters(v, vor, save=False, filename=None, show_plot=True):
    voronoi_plot_2d(vor, show_points=True, show_vertices=False)
    cluster_locs_df = v[v['lk'] != -1]
    labels = cluster_locs_df['lk'].unique()

    for m in labels:
        cluster_points = cluster_locs_df[
            cluster_locs_df['lk'] == m].as_matrix(columns=['x', 'y'])
        hull = ConvexHull(cluster_points)
        plt.plot(cluster_points[:, 0], cluster_points[:, 1], 'ko')
        for simplex in hull.simplices:
            plt.plot(
                cluster_points[simplex, 0], cluster_points[simplex, 1], 'r-')

    if save and filename is not None:
        plt.savefig(filename)

    if show_plot:
        plt.show()


def plot_polygon(polygon, figure=None):
    if figure is None:
        from matplotlib import pylab as plt
        fig = plt.figure(figsize=(10, 10))
    else:
        fig = figure
    ax = fig.add_subplot(111)
    margin = .3
    if polygon.bounds:
        x_min, y_min, x_max, y_max = polygon.bounds
        ax.set_xlim([x_min-margin, x_max+margin])
        ax.set_ylim([y_min-margin, y_max+margin])
        patch = PolygonPatch(polygon, fc='#999999',
                             ec='#000000', fill=True,
                             zorder=-1)
        ax.add_patch(patch)
    return fig


def plot_voronoi_diagram(vor, cluster_column='lk', locs_df=None):
    v2d = voronoi_plot_2d(vor, show_points=False, show_vertices=False)

    if locs_df is not None:
        cluster_locs_df = locs_df.copy()
        cluster_locs_df = (
            cluster_locs_df[cluster_locs_df[cluster_column] != -1]
        )
        labels = cluster_locs_df[cluster_column].unique()

        ax = v2d.axes[0]
        # figure = plt.figure()
        # ax = figure.add_subplot(111)
        for m in labels:
            cluster_points = (
                cluster_locs_df[cluster_locs_df[cluster_column] == m].
                as_matrix(columns=['x [nm]', 'y [nm]'])
            )
            concave_hull, edge_points = (
                    tri.alpha_shape(cluster_points, alpha=0.00001))
            patch = PolygonPatch(concave_hull, fc='#999999',
                                 ec='#000000', fill=True,
                                 zorder=-1)
            ax.add_patch(patch)

            ax.plot(cluster_points[:, 0], cluster_points[:, 1], 'rx')
            # lines = LineCollection(
            #    edge_points,color=mcolors.to_rgba('b'),linestyle='solid')
            # ax.add_collection(lines)
    return v2d


def plot_cluster_polygons(locs,
                          figure=None,
                          cluster_column='lk',
                          area_filter=0.0,
                          patch_colour='#303F9F'):

    if cluster_column not in locs:
        raise ValueError("Run clustering first")

    locs = locs[locs[cluster_column] != -1]
    labels = locs[cluster_column].unique()
    if figure:
        ax = figure.axes[0]
    else:
        figure = plt.figure()
        ax = figure.add_subplot(111)

    for m in labels:
        points = (
            locs[locs[cluster_column] == m].
            as_matrix(columns=['x [nm]', 'y [nm]'])
        )
        concave_hull, edge_points = (
                tri.alpha_shape(points, alpha=0.01)
        )
        if 'GeometryCollection' not in concave_hull.geom_type:
            if concave_hull.area > area_filter:
                patch = PolygonPatch(concave_hull, fc=patch_colour,
                                     ec='#000000', fill=True,
                                     zorder=-1)
                ax.add_patch(patch)
        plt.plot(points[:, 0], points[:, 1], 'bo', alpha=.5)
    return figure


def polygon_area(points):
    concave_hull, edge_points = (
            tri.alpha_shape(points, alpha=0.01)
    )
    return concave_hull.area


def polygon_perimeter(points):
    concave_hull, edge_points = (
            tri.alpha_shape(points, alpha=0.01)
    )
    return concave_hull.length


def import_voronoi_clusters(path, sheetname=None, column='object_id'):
    if path.endswith('xlsx'):
        sn = 'localisations'
        if sheetname:
            sn = '{0} localisations'.format(sheetname)

        data = pd.read_excel(path, sheet_name=sn)
        df = data[data[column] > -1]
        return df
    else:
        raise ValueError("Input data must be in Excel format")

