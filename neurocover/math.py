import numpy
import scipy.spatial


def vector_projection(vectors1, vectors2):
    """ a . b * b / (|b| . |b|)
    """
    ab = numpy.sum(vectors1 * vectors2, axis=1)
    bb = numpy.sum(vectors2 * vectors2, axis=1)
    return  (ab / bb)[:, numpy.newaxis] * vectors2 


def vectorized_dot(vector, vectors):
    return numpy.einsum('i,ji->j', vector, vectors)


def rowwise_dot(vectors1, vectors2):
    """ line by line dot product
    """
    return numpy.sum(vectors1 * vectors2, axis=1)


def skew_symmetric_matrix(v):
    return numpy.array(((0., - v[2], v[1]), (v[2], 0., -v[0]), (- v[1], v[0], 0.)))


def rotate_from_unit_vector_to_another(u_a, u_b):

    v = numpy.cross(u_a, u_b)

    v_x = skew_symmetric_matrix(v)

    return numpy.identity(3) + v_x + numpy.linalg.matrix_power(v_x, 2) * (1.  / (1. + numpy.dot(u_a, u_b)))


def apply_rotation_to_points(points, rotation_matrix):
    return numpy.einsum('ij,kj->ik', points, rotation_matrix)
