import math
import numpy as np
from scipy.spatial import Delaunay, ConvexHull
from sklearn.decomposition import PCA
from shapely.geometry import Polygon, MultiPoint, MultiLineString
from shapely.ops import cascaded_union, polygonize

def triangulate(xy):
    """
    Returns the Delaunay triangulation object.
    """
    return Delaunay(xy)


def hull_vertices(xy):
    """
    Returns the vertices from a Hull object.
    """
    hull = ConvexHull(xy)
    return xy[hull.vertices]


def hull_area(xy):
    """
    Returns the area from the Hull object. For
    a 2D shape this is the perimeter.
    """
    hull = ConvexHull(xy)
    return hull.area
    

def poly_area2D(pts):
    """
    Returns a calculation of the 2D polygon area.
    """
    lines = np.hstack([pts,np.roll(pts,-1,axis=0)])
    area = 0.5*abs(sum(x1*y2-x2*y1 for x1,y1,x2,y2 in lines))
    return area


def pc_ratio(pts):
    """
    Returns the ratio of the first two principle components.
    Uses the scikit-learn package to calculate PCA.
    """
    pca = PCA(n_components=2)
    pca.fit(pts)
    pc = pca.components_[0]
    return max(abs(pc[0]),abs(pc[1])) / min(abs(pc[0]),abs(pc[1]))


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.

    Parameters
    ----------
    points: Numpy array
        Oject coordinates.
    alpha: alpha value
        Will influence the gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers. Too large, and
        you lose everything!

    Returns
    -------
    The object border.
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

        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])

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
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        #print circum_r
        if circum_r < 1.0/alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles), edge_points


def intersection(x1, y1, x2, y2):
#      y = a*x + b
    a1 = (y1[1] - y1[0]) / (x1[1] - x1[0])
    a2 = (y2[1] - y2[0]) / (x2[1] - x2[0])
    b1 = y1[0] - a1 * x1[0]
    b2 = y2[0] - a2 * x2[0]
    xi = (b2 - b1) / (a1 - a2)
    yi = a1 * xi + b1
    return (xi, yi)