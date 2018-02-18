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

angle = numpy.pi * 2. / 3.

def _generate_points(dl, length):

    dir1 = numpy.array((0., numpy.cos(0.), numpy.sin(0.)))
    dir2 = numpy.array((0., numpy.cos(angle), numpy.sin(angle)))
    dir3 = numpy.array((0., numpy.cos(-angle), numpy.sin(-angle)))


    points = numpy.array([dl * dir1, length * dir1,
                          dl * dir2, length * dir2,
                          dl * dir3, length * dir3])

    return points

def create_shapes(dl, length, R):

    radii = numpy.array([1., 1., 1., 1., 1., 1.]) * R

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


def _bounding_box(R, length):

    d = numpy.array((0., numpy.cos(-angle), numpy.sin(-angle)))

    ds = R / numpy.cos(numpy.pi / 2. - angle)

    bmin = + d * (length + ds)
    bmax = - d * (length + ds)

    bmin[0] = - R
    bmax[0] = R

    return bmin - 0.2, bmax + 0.2


def intersection_surface(node_data, edges, cylinder_radius, number_of_points):

    u_a = numpy.array([0., 0., 1.])

    cylinders = []

    for v0, v1 in edges:

        p0 = node_data[v0, :3]
        p1 = node_data[v1, :3]

        cylinder_height = numpy.linalg.norm(p1 - p0)

        u_b = (p1 - p0) / cylinder_height

        points = neurocover.point_generators.on_cylinder_surface(number_of_points, cylinder_height, cylinder_radius)
        # local ref system. need to e rotated, translated
        R = neurocover.math.rotate_from_unit_vector_to_another(u_a, u_b)

        # rotation
        new_points = neurocover.math.apply_rotation_to_points(points, R)

        # translation
        new_points += (p0 + p1) * 0.5

        cylinders.append(new_points)

    total_area = 0.

    for i in xrange(len(edges)):

        surface_points = cylinders[i]

        mask = numpy.ones(len(points), dtype=numpy.bool)

        for j in xrange(len(edges)):

            if i != j:

                v0, v1 = edges[j]

                p0 = node_data[v0, :3]
                p1 = node_data[v1, :3]

                r0 = node_data[v0, 3]
                r1 = node_data[v1, 3]

                mask &= ~neurocover.spatial.points_inside_cylinder(surface_points, p0, p1, r0, r1)


        total_area += (float(mask.sum()) / float(mask.size)) * 2. * numpy.pi * cylinder_radius * cylinder_height

    return total_area, cylinders



if __name__ == '__main__':

    R = 0.5
    length = 1.

    N = 1000

    bb_point_min, bb_point_max = _bounding_box(R, length)

    bb_volume = numpy.prod(bb_point_max - bb_point_min)

    L.info('Bounding Box volume: {}'.format(bb_volume))

    sample_points = neurocover.point_generators.uniform(N, bb_point_min, bb_point_max)

    ########

    node_data1, edges1 = create_shapes(0., length, R)

    own1 = neurocover.spatial.ownership(node_data1, edges1, sample_points,
                                                inclusion_func=neurocover.spatial.points_inside_cylinder)

    ########

    dl = numpy.cos(numpy.pi / 3.) * R
    node_data2, edges2 = create_shapes(dl, length, R)

    own2 = neurocover.spatial.ownership(node_data2, edges2, sample_points,
                                                inclusion_func=neurocover.spatial.points_inside_cylinder)

    # union of points inside cylinders for two cases
    all_included_mask1 = own1[0] | own1[1] | own1[2]
    all_included_mask2 = own2[0] | own2[1] | own2[2]

    intersection_mask = all_included_mask1 ^ all_included_mask2

    Nh = float(intersection_mask.sum())
    V = bb_volume * Nh / float(N)
    dV = bb_volume * numpy.sqrt(Nh - Nh / N) / N
    rel_dV = numpy.sqrt((1. - Nh / N) / Nh)

    string = 'Intersection Volume: {} +- {}'.format(V, dV)

    L.info(string)
    L.info('Relative Error: {}'.format(rel_dV))

    Nh = float(all_included_mask1.sum())
    V = bb_volume * Nh / float(N)
    dV = bb_volume * numpy.sqrt(Nh - Nh / N) / N

    string = 'Total Volume: {} +- {}'.format(V, dV)

    L.info(string)
    L.info('Relative Error: {}'.format(rel_dV))


    # -----




    total_area, cylinders1 = intersection_surface(node_data1, edges1, R, N)

    total_area += 3. * numpy.pi * R ** 2

    inter_area, cylinders2 = intersection_surface(node_data2, edges2, R, N)

    inter_area += 3. * numpy.pi * R ** 2

    print "Total Area: ", total_area
    print "Inter Area: ", inter_area


    f = matplotlib.pylab.figure()
    ax = f.add_subplot(111, projection='3d')

    ax.set_xlim([bb_point_min[0], bb_point_max[0]])
    ax.set_ylim([bb_point_min[1], bb_point_max[1]])
    ax.set_zlim([bb_point_min[2], bb_point_max[2]])

    for cylinder in cylinders1:
        ax.scatter(cylinder[:, 0], cylinder[:, 1], cylinder[:, 2], color='b', alpha=0.1)

    # ---------------------

    f = matplotlib.pylab.figure()
    ax = f.add_subplot(111, projection='3d')

    ax.set_xlim([bb_point_min[0], bb_point_max[0]])
    ax.set_ylim([bb_point_min[1], bb_point_max[1]])
    ax.set_zlim([bb_point_min[2], bb_point_max[2]])

    rest_points = sample_points[all_included_mask1 & ~intersection_mask]

    ax.scatter(rest_points[:, 0], rest_points[:, 1], rest_points[:, 2], color='b', alpha=0.1)

    mask = intersection_mask 

    intx_points = sample_points[mask] 
    ax.scatter(intx_points[:, 0], intx_points[:, 1], intx_points[:, 2], color='r')
    axisEqual3D(ax)

    ax.view_init(elev=0, azim=-179)
    matplotlib.pylab.show()

