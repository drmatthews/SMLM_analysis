import os
import math

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from shapely.geometry import Polygon, MultiPoint, MultiLineString
from shapely.ops import cascaded_union, polygonize
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans

import smlm_analysis.utils.triangulation as tri

class Cluster(object):
    def __init__(self, coordinates, cluster_id):
        N = coordinates.shape[0]
        self.x = np.reshape(coordinates[:, 0], (N, 1))
        self.y = np.reshape(coordinates[:, 1], (N, 1))
        self.cluster_id = cluster_id
        id_array = np.array(
            [cluster_id for i in range(coordinates.shape[0])],
            dtype='int32')
        self.id_array = np.reshape(id_array, (N, 1))
        self.occupancy = coordinates.shape[0]
        self.alpha = 0.01
        self.hull, edges = self._alpha_shape(coordinates, self.alpha)
        self.pc_ratio = None
        self.area = None
        self.perimeter = None
        self.nn = None
        self.center = None
        self.is_valid = False
        if edges:
            self.edges = np.vstack([np.reshape(a, (1, 4)) for a in edges])
            self.pc_ratio = self._pc(edges)
            self.area = self.hull.area
            self.perimeter = self.hull.length
            self.nn = np.reshape(self._near_neighbours(coordinates), (N, 1))
            self.center = self._center(np.hstack((self.x, self.y)))
            self.is_valid = True
        else:
            self.edges = edges

    def _alpha_shape(self, points, alpha):
        """
        Compute the alpha shape (concave hull) of a set
        of points.
        @param points: Numpy array of object coordinates.
        @param alpha: alpha value to influence the
            gooeyness of the border. Smaller numbers
            don't fall inward as much as larger numbers.
            Too large, and you lose everything!
        """
        if len(points) < 4:
            # When you have a triangle, there is no sense
            # in computing an alpha shape.
            return MultiPoint(list(points)).convex_hull

        def add_edge(edges, edge_points, coords, i, j):
            """
            Add a line between the i-th and j-th points,
            if not in the list already
            """
            if (i, j) in edges or (j, i) in edges:
                # already added
                return

            edges.add((i, j))
            edge_points.append(coords[[i, j]])

        # coords = numpy.array([point.coords[0] for point in points])
        coords = points
        tri = Delaunay(coords)
        edges = set()
        edge_points = []
        # loop over triangles:
        # ia, ib, ic = indices of corner points of the
        # triangle
        for ia, ib, ic in tri.vertices:
            pa = coords[ia]
            pb = coords[ib]
            pc = coords[ic]

            # Lengths of sides of triangle
            a = math.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)
            b = math.sqrt((pb[0] - pc[0])**2 + (pb[1] - pc[1])**2)
            c = math.sqrt((pc[0] - pa[0])**2 + (pc[1] - pa[1])**2)
            # Semiperimeter of triangle
            s = (a + b + c) / 2.0
            # Area of triangle by Heron's formula
            area = math.sqrt(s * (s - a) * (s - b) * (s - c))

            if area > 0.0:
                circum_r = a * b * c / (4.0 * area)
                # Here's the radius filter.
                # print circum_r
                if circum_r < 1.0 / alpha:
                    add_edge(edges, edge_points, coords, ia, ib)
                    add_edge(edges, edge_points, coords, ib, ic)
                    add_edge(edges, edge_points, coords, ic, ia)
            else:
                continue
        m = MultiLineString(edge_points)
        triangles = list(polygonize(m))
        return cascaded_union(triangles), edge_points

    def _near_neighbours(self, points):
        tree = KDTree(points, leaf_size=2)
        dist, ind = tree.query(points[:], k=2)
        return dist[:, 1]

    def _pc(self, edges):
        e = np.vstack(edges)
        Y = np.c_[e[:, 0], e[:, 1]]
        pcr = tri.pc_ratio(Y)
        return pcr

    def _center(self, xy):
        kmeans = KMeans(n_clusters=1, random_state=0).fit(xy)
        return kmeans.cluster_centers_[0]


class ClusterList(object):
    def __init__(self):
        self.clusters = []
        self.n_clusters = len(self.clusters)
        self.noise = None

    def __getitem__(self, i):
        return self.clusters[i]

    def __len__(self):
        return self.n_clusters

    def append(self, item):
        self.clusters.append(item)
        self.n_clusters = len(self.clusters)

    def save(self, path, sheetname='image'):

        ext = os.path.splitext(path)[1]
        if ('xlsx' in ext):
            if self.clusters:

                writer = pd.ExcelWriter(path, engine='openpyxl')
                if os.path.exists(path):
                    book = load_workbook(path)
                    writer.book = book
                    writer.sheets = dict(
                        (ws.title, ws) for ws in book.worksheets
                    )

                coords = []
                stats = []
                hulls = []
                for c in self.clusters:
                    coords.append(np.hstack([c.x, c.y, c.nn, c.id_array]))
                    stats.append(np.hstack(
                        [c.area, c.perimeter, c.pc_ratio, c.occupancy]))
                    hull_cluster_id = np.array(
                        [c.cluster_id
                         for i in range(c.edges.shape[0])], dtype='int32')
                    hull_cluster_id = np.reshape(
                        hull_cluster_id, (c.edges.shape[0], 1))
                    hulls.append(np.hstack([c.edges, hull_cluster_id]))

                c_df = pd.DataFrame(np.vstack(coords))
                c_df.columns = [
                    'x [nm]', 'y [nm]', 'nn distance [nm]', 'cluster_id']
                s_df = pd.DataFrame(np.vstack(stats))
                s_df.columns = [
                    'area [nm^2]', 'perimeter [nm]', 'pc_ratio', 'occupancy']
                h_df = pd.DataFrame(np.vstack(hulls))
                h_df.columns = [
                    'x0 [nm]', 'y0 [nm]', 'x1 [nm]', 'y1 [nm]', 'cluster_id']

                c_df.to_excel(
                    writer,
                    sheet_name='{} cluster coordinates'.format(sheetname),
                    index=False
                )
                s_df.to_excel(
                    writer,
                    sheet_name='{} cluster statistics'.format(sheetname),
                    index=False
                )
                h_df.to_excel(
                    writer,
                    sheet_name='{} convex hulls'.format(sheetname),
                    index=False
                )

                if self.noise is not None:
                    noise_df = pd.DataFrame(self.noise)
                    noise_df.columns = ['x [nm]', 'y [nm]']
                    noise_df.to_excel(
                        writer,
                        sheet_name='{} noise'.format(sheetname),
                        index=False
                    )
                writer.save()
            else:
                print("no clusters to save")
        else:
            print("file path for saving must contain extension xlsx")
