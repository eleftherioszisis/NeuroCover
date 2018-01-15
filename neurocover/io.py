import numpy
import neurom


def load_morphology(filepath):

    neuron = neurom.load_neuron(filepath)

    pdata = []
    edges = []

    for neurite in neuron.neurites:

        neurite_points = neurite.points.tolist()

        N = len(pdata)

        neurite_edges = [[i, i + 1] for i in xrange(N, len(neurite_points) + N - 1)]

        pdata.extend(neurite_points)
        edges.extend(neurite_edges)

    return numpy.asarray(pdata), numpy.asarray(edges)
