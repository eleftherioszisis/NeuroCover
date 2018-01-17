import numpy
from .math import vectorized_dot


def points_to_segment(points, p0, p1, return_t=False):

    edge_vector = p1 - p0

    # t for closest distance between line and point
    ts = vectorized_dot(edge_vector, points - p0) / numpy.dot(edge_vector, edge_vector)

    # clamp for segment extent
    ts = numpy.clip(ts, 0., 1.)

    # closest point on capsule axis to point
    p_t = p0 + ts[:, numpy.newaxis] * edge_vector

    distx = numpy.linalg.norm(p_t - points, axis=1)

    if return_t:

        return distx, ts

    else:

        return distx

