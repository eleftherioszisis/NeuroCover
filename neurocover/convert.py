import numpy
import neurom as nm
from neurom.core import iter_sections


def morphology(neuron, neurite_type=None, reset_radii=True):

    pdata = []
    edges = []

    if neurite_type is None:
        it = iter_sections(neuron)
    else:
        it = iter_sections(neuron, neurite_filter=lambda n: n.type == neurite_type)

    for section in it:

        section_points = section.points[:, :4]

        if reset_radii:
            section_points[:, 3] = 0.01

        section_points = section_points.tolist()

        N = len(pdata)

        neurite_edges = [[i, i + 1] for i in range(N, len(section_points) + N - 1)]

        pdata.extend(section_points)
        edges.extend(neurite_edges)

    return numpy.asarray(pdata), numpy.asarray(edges)
