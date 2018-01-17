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


if __name__ == '__main__':

    filename = sys.argv[1]

    neuron = neurom.load_neuron(filename)

    node_data, edges = neurocover.convert.morphology(neuron)

    bb_point_min, bb_point_max = neurom.geom.bounding_box(neuron)

    bb_volume = numpy.prod(bb_point_max - bb_point_min)

    N = 500000

    sample_points = neurocover.point_generators.uniform(N, bb_point_min, bb_point_max)

    mask_idx_in = neurocover.spatial.is_inside(node_data, edges, sample_points, 0.)

    Nh = float(mask_idx_in.sum())

    V = bb_volume * Nh / float(N)

    dV = bb_volume * numpy.sqrt(Nh - Nh / N) / N
    rel_dV = numpy.sqrt((1. - Nh / N) / Nh)

    string = 'Volume: {} +- {}'.format(V, dV)

    L.info(string)
    L.info('Relative Error: {}'.format(rel_dV))
