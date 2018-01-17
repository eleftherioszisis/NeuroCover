import sys
import numpy
import logging
import neurom
import neurom.geom
import scipy.spatial
import matplotlib.pylab

import neurocover
import neurocover.math
import neurocover.spatial
import neurocover.convert
import neurocover.parallel
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
        yield inflation_coefficient, percentage

        inflation_coefficient += dstep


def apply_to_morphology(morphology):

    node_data, edges = neurocover.convert.morphology(morphology)

    bb_point_min, bb_point_max = neurom.geom.bounding_box(morphology)

    N = 200000

    sample_points = neurocover.point_generators.uniform(N, bb_point_min, bb_point_max)

    mask = neurocover.spatial.points_inside_convex_hull(sample_points, scipy.spatial.ConvexHull(node_data[:, :3]))

    sample_points = sample_points[mask]

    dstep = 1.

    results = list(data_generator(node_data, edges, sample_points, dstep))

    return numpy.asarray(results)

if __name__ == '__main__':

    dirpath = sys.argv[1]

    population = neurom.load_neurons(dirpath)

    results_list = neurocover.parallel.multiprocessing_map(apply_to_morphology, population)

    f, ax = matplotlib.pylab.subplots(1,2)

    for i, morphology in enumerate(population):

        name = morphology.name

        coeffs, percs = results_list[i].T

        ax[0].plot(coeffs, percs, label=name)
        ax[0].legend()
        dstep = 1.
        dpercs = (percs[1:-1] - percs[:-2]) / (2. * dstep)
        ax[1].plot(coeffs[1:-1], dpercs)


    matplotlib.pylab.show()

