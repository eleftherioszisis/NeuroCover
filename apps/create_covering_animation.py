import sys
import numpy
import logging
import neurom
import neurom.geom
import scipy.spatial
import neurocover
import neurocover.math
import neurocover.spatial
import neurocover.convert
import neurocover.animation
import neurocover.point_generators


logging.basicConfig(level=logging.INFO)


L = logging.getLogger(__name__)


def data_generator(node_data, edges, sample_points, dstep):

    inflation_coefficient = 0.

    N = len(sample_points)

    percentage = 0.

    mask_inside = numpy.zeros(len(sample_points), dtype=numpy.bool)
    idx = numpy.arange(len(sample_points), dtype=numpy.int)

    while abs(percentage - 1.0) > 0.001:

        # don't use the points that have already been found inside the morphology from
        # the previous iteration
        idx_out = idx[~mask_inside]

        mask_idx_in = neurocover.spatial.is_inside(node_data, edges, sample_points[idx_out], inflation_coefficient)

        # add the newfound points to the mask
        mask_inside[idx_out[mask_idx_in]] = True

        percentage = float(mask_inside.sum()) / float(N)

        L.info('{0}, {1}'.format(inflation_coefficient, percentage))
        yield sample_points[mask_inside], inflation_coefficient, percentage

        inflation_coefficient += dstep


if __name__ == '__main__':

    filename = sys.argv[1]
    if len(sys.argv) > 2:
        output = sys.argv[2]
    else:
        output = 'test.mp4'
    if len(sys.argv) > 3:
        neurite_type = getattr(neurom.NeuriteType, sys.argv[3])
    else:
        neurite_type = neurom.NeuriteType.basal_dendrite

    neuron = neurom.load_neuron(filename)

    node_data, edges = neurocover.convert.morphology(neuron, neurite_type=neurite_type)

    bb_point_min, bb_point_max = neurom.geom.bounding_box(neuron)

    N = 200000

    sample_points = neurocover.point_generators.uniform(N, bb_point_min, bb_point_max)

    mask = neurocover.spatial.points_inside_convex_hull(sample_points, scipy.spatial.ConvexHull(node_data[:, :3]))

    sample_points = sample_points[mask]

    dstep = 1.

    it = data_generator(node_data, edges, sample_points, dstep)

    results = list(it)

    neurocover.animation.space_filling_inflation(results, node_data, edges, bb_point_min, bb_point_max, out_file=output)
