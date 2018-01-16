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
            BP = point - ends[j]
            AB = edge_vectors[j]

            if numpy.dot(AP, AB) <= 0. or numpy.dot(BP, AB) >= 0.:
                continue

            proj = numpy.dot(AP, AB) * AB / edge_lengths_squared[j]

            distance2 = numpy.dot(proj - AP, proj - AP)

            if distance2 <=  factor_radii_squared[j]:

                mask[i] = True
                break

    return mask
