import numpy


def morphology(neuron):

    pdata = []
    edges = []

    for section in neuron.sections:

        section_points = section.points.tolist()

        N = len(pdata)

        neurite_edges = [[i, i + 1] for i in xrange(N, len(section_points) + N - 1)]

        pdata.extend(section_points)
        edges.extend(neurite_edges)

    return numpy.asarray(pdata), numpy.asarray(edges)
