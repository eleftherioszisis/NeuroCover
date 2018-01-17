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

