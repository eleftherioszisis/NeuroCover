import numpy
import scipy.spatial


def vector_projection(vectors1, vectors2):
    """ a . b * b / (|b| . |b|)
    """
    ab = numpy.sum(vectors1 * vectors2, axis=1)
    bb = numpy.sum(vectors2 * vectors2, axis=1)
    return  (ab / bb)[:, numpy.newaxis] * vectors2 


def rowwise_dot(vectors1, vectors2):
    """ line by line dot product
    """
    return np.sum(vectors1 * vectors2, axis=1)


def distance_point_to_edge(points, edge_starts, edge_ends):
    """ calculates the distance to the edge
    """
    ABs = edge_ends - edge_starts
    APs = points - edge_starts
    BPs = points - edge_ends

    cprod = numpy.cross(APs, ABs)

    distx =  numpy.linalg.norm(cprod, axis=1) / numpy.linalg.norm(ABs, axis=1)

    mask = (rowwise_dot(APs, ABs) <= 0.) | (rowwise_dot(BPs, ABs) >= 0.)
    distx[mask] = np.inf

    return distx
