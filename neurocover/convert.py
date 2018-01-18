import numpy
import neurom as nm
from neurom.core import iter_sections

def morphology(neuron, neurite_type=nm.NeuriteType.basal_dendrite):

    pdata = []
    edges = []

    filter = lambda n : n.type == neurite_type

    for section in iter_sections(neuron, neurite_filter=filter):

        section_points = section.points.tolist()

        N = len(pdata)

        neurite_edges = [[i, i + 1] for i in xrange(N, len(section_points) + N - 1)]

        pdata.extend(section_points)
        edges.extend(neurite_edges)

    return numpy.asarray(pdata), numpy.asarray(edges)
