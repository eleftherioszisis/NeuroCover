import sys
import numpy
import logging
import scipy.spatial
import neurocover
import neurocover.math
import neurocover.spatial
import neurocover.convert
import neurocover.animation
import neurocover.point_generators


import matplotlib.pylab
from mpl_toolkits.mplot3d import Axes3D

logging.basicConfig(level=logging.INFO)


L = logging.getLogger(__name__)


def _generate_points(dl, r):

    length = 2.

    angle = numpy.pi * 2. / 3.

    dir1 = numpy.array((0., numpy.cos(0.), numpy.sin(0.)))
    dir2 = numpy.array((0., numpy.cos(angle), numpy.sin(angle)))
    dir3 = numpy.array((0., numpy.cos(-angle), numpy.sin(-angle)))


    points = numpy.array([dl * dir1, length * dir1,
                          dl * dir2, length * dir2,
                          dl * dir3, length * dir3])

    return points

def create_shapes(dl, length):

    radii = numpy.array([1., 1., 1., 1., 1., 1.])

    points = _generate_points(dl, length)

    edges = numpy.array([[0, 1], [2, 3], [4, 5]])

    return numpy.column_stack([points, radii]), edges


def axisEqual3D(ax):
    extents = numpy.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = numpy.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


if __name__ == '__main__':


    node_data1, edges1 = create_shapes(0., 3.)

    dl = numpy.cos(numpy.pi / 3.) * node_data1[0, 3]

    node_data2, edges2 = create_shapes(dl, 3.)

    bb_point_min = numpy.array([-3., -3., -3.])
    bb_point_max = numpy.array([3., 3., 3.])

    bb_volume = numpy.prod(bb_point_max - bb_point_min)

    N = 200000

    sample_points = neurocover.point_generators.uniform(N, bb_point_min, bb_point_max)


    own1 = neurocover.spatial.ownership(node_data1, edges1, sample_points,
                                                inclusion_func=neurocover.spatial.points_inside_cylinder)

    own2 = neurocover.spatial.ownership(node_data2, edges2, sample_points,
                                                inclusion_func=neurocover.spatial.points_inside_cylinder)

    all_included_mask = own1[0] | own1[1] | own1[2]

#    intersection_mask = own[0] ^ ((own[0] & own[1]) | (own[0] & own[2])) 

    intersection_mask = all_included_mask ^ (own2[0] | own2[1] | own2[2])
    #intersection_mask = numpy.logical_xor(all_included_mask, intersection_mask)

    """
    Nh = float(all_included_mask.sum())

    V = bb_volume * Nh / float(N)

    dV = bb_volume * numpy.sqrt(Nh - Nh / N) / N
    rel_dV = numpy.sqrt((1. - Nh / N) / Nh)

    string = 'Volume: {} +- {}'.format(V, dV)

    L.info(string)
    L.info('Relative Error: {}'.format(rel_dV))


    # --------------------------------------------

    shared_points_mask = (own[0] & own[1]) | \
                         (own[0] & own[2]) | \
                         (own[1] & own[2]) | \
                         (own[0] & own[1] & own[2])
    Nh = float(shared_points_mask.sum())

    print Nh
    V = bb_volume * Nh / float(N)

    dV = bb_volume * numpy.sqrt(Nh - Nh / N) / N
    rel_dV = numpy.sqrt((1. - Nh / N) / Nh)

    string = 'Intersection Volume: {} +- {}'.format(V, dV)

    L.info(string)
    L.info('Relative Error: {}'.format(rel_dV))
    """
    f = matplotlib.pylab.figure()
    ax = f.add_subplot(111, projection='3d')

    ax.set_xlim([-4, 4])
    ax.set_ylim([-4, 4])
    ax.set_zlim([-4, 4])

    rest_points = sample_points[all_included_mask & ~intersection_mask]

    ax.scatter(rest_points[:, 0], rest_points[:, 1], rest_points[:, 2], color='b', alpha=0.1)

    #ax.
    #intx_points = sample_points[shared_points_mask]


    mask = intersection_mask 

    intx_points = sample_points[mask] 
    ax.scatter(intx_points[:, 0], intx_points[:, 1], intx_points[:, 2], color='r')
    axisEqual3D(ax)
    matplotlib.pylab.show()

