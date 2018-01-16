import sys
import numpy
import logging
import neurom
import neurom.geom
import neurocover
import neurocover.math
import neurocover.convert
import neurocover.animation
import neurocover.point_generators


logging.basicConfig(level=logging.INFO)

L = logging.getLogger(__name__)

def data_generator(node_data, edges, sample_points, dstep):

    inflation_coefficient = 0.

    percentage = 0.

    while abs(percentage - 1.0) > 0.01:

        mask = neurocover.math.is_inside(node_data, edges, sample_points, inflation_coefficient)

        percentage = float(mask.sum()) / float(N)

        yield sample_points[mask], inflation_coefficient, percentage

        inflation_coefficient += dstep


def data_generator2(node_data, edges, sample_points, dstep):

    inflation_coefficient = 0.

    percentage = 0.

    mask_inside = numpy.zeros(len(sample_points), dtype=numpy.bool)
    idx = numpy.arange(len(sample_points), dtype=numpy.int)

    while abs(percentage - 1.0) > 0.01:

        idx_out = idx[~mask_inside]

        mask_idx_in = neurocover.math.is_inside2(node_data, edges, sample_points[idx_out], inflation_coefficient)

        mask_inside[idx_out[mask_idx_in]] = True

        percentage = float(mask_inside.sum()) / float(N)

        L.info('{0}, {1}'.format(inflation_coefficient, percentage))
        yield sample_points[mask_inside], inflation_coefficient, percentage

        inflation_coefficient += dstep


if __name__ == '__main__':

    filename = sys.argv[1]

    neuron = neurom.load_neuron(filename)

    node_data, edges = neurocover.convert.morphology(neuron)

    bb_point_min, bb_point_max = neurom.geom.bounding_box(neuron)

    N = 100000

    sample_points = neurocover.point_generators.uniform(N, bb_point_min, bb_point_max)

    dstep = 1.

    it = data_generator2(node_data, edges, sample_points, dstep)

    results = list(it)

    neurocover.animation.space_filling_inflation(results, node_data, edges, bb_point_min, bb_point_max, out_file='test.mp4')
