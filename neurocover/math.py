import numpy
import scipy.spatial

from .spatial import is_point_inside_segment_bb
from .bounding_box import vectorized_AABB_tapered_capsule

def vector_projection(vectors1, vectors2):
    """ a . b * b / (|b| . |b|)
    """
    ab = numpy.sum(vectors1 * vectors2, axis=1)
    bb = numpy.sum(vectors2 * vectors2, axis=1)
    return  (ab / bb)[:, numpy.newaxis] * vectors2 

def vectorized_dot_product(vector, vectors):
    return numpy.einsum('i,ji->j', vector, vectors)

def rowwise_dot(vectors1, vectors2):
    """ line by line dot product
    """
    return numpy.sum(vectors1 * vectors2, axis=1)


def is_inside(node_data, edges, sample_points, inflation_coefficient):

    points = node_data[:, (0, 1, 2)]
    radii = node_data[:, 3]

    starts = points[edges[:, 0]]
    ends = points[edges[:, 1]]

    radii_starts = radii[edges[:, 0]]
    radii_ends = radii[edges[:, 1]]

    edge_vectors = ends - starts
    edge_radii = 0.5 * (radii_starts + radii_ends)

    mask = numpy.zeros(len(sample_points), dtype=numpy.bool)

    edge_lengths_squared = rowwise_dot(edge_vectors, edge_vectors) ** 2

    factor = (1. + inflation_coefficient) 
    factor_radii_squared = (factor * edge_radii) ** 2

    for i, point in enumerate(sample_points):
        for j in xrange(len(edges)):

                AP = point - starts[j]
                AB = edge_vectors[j]

                if numpy.dot(AP, AB) <= 0.:
                    continue

                BP = point - ends[j]

                if numpy.dot(BP, AB) >= 0.:
                    continue

                proj = numpy.dot(AP, AB) * AB / edge_lengths_squared[j]

                distance2 = numpy.dot(proj - AP, proj - AP)

                if distance2 <=  factor_radii_squared[j]:

                    mask[i] = True
                    break

    return mask


def is_inside2(node_data, edges, sample_points, inflation_coefficient):

    points = node_data[:, (0, 1, 2)]
    radii = node_data[:, 3]

    starts = points[edges[:, 0]]
    ends = points[edges[:, 1]]

    radii_starts = radii[edges[:, 0]] * (1. + inflation_coefficient)
    radii_ends = radii[edges[:, 1]] * (1. + inflation_coefficient)

    edge_vectors = ends - starts
    edge_radii = 0.5 * (radii_starts + radii_ends)

    mask = numpy.zeros(len(sample_points), dtype=numpy.bool)

    edge_lengths_squared = rowwise_dot(edge_vectors, edge_vectors) ** 2
    factor_radii_squared = edge_radii ** 2

    bbs = vectorized_AABB_tapered_capsule(starts, ends, radii_starts, radii_ends)

    mask_inside = numpy.zeros(len(sample_points), dtype=numpy.bool)
    idx = numpy.arange(len(sample_points), dtype=numpy.int)

    for n in xrange(len(edges)):

        xmin, ymin, zmin, xmax, ymax, zmax = bbs[n]

        pidx = idx[~mask_inside]

        # point inside bb
        mask = (xmin <= sample_points[pidx, 0]) & (sample_points[pidx, 0] <= xmax) & \
               (ymin <= sample_points[pidx, 1]) & (sample_points[pidx, 1] <= ymax) & \
               (zmin <= sample_points[pidx, 2]) & (sample_points[pidx, 2] <= zmax)

        if not mask.any():
            continue

        pidx = pidx[mask]

        APs = sample_points[pidx] - starts[n]

        mask_in = vectorized_dot_product(edge_vectors[n], APs) >= 0.

        if not mask_in.any():
            continue

        pidx = pidx[mask_in]

        BPs = sample_points[pidx] - ends[n]

        mask_in = vectorized_dot_product(edge_vectors[n], BPs) <= 0.

        if not mask_in.any():
            continue

        pidx = pidx[mask_in]

        APs = sample_points[pidx] - starts[n]

        projs = (vectorized_dot_product(edge_vectors[n], APs) / edge_lengths_squared[n])[:, numpy.newaxis] * edge_vectors[n]

        distx2 = rowwise_dot(projs - APs, projs - APs)

        mask_in = distx2 <= factor_radii_squared[n]

        if not mask_in.any():
            continue

        pidx = pidx[mask_in]

        mask_inside[pidx] = True

    return mask_inside
