import numpy


def scale(sample_points, pmin, pmax):

    xmin, ymin, zmin = pmin
    xmax, ymax, zmax = pmax

    sample_points[:, 0] = xmin + (xmax - xmin) * sample_points[:, 0]
    sample_points[:, 1] = ymin + (ymax - ymin) * sample_points[:, 1]
    sample_points[:, 2] = zmin + (zmax - zmin) * sample_points[:, 2]

    return sample_points


def uniform(number_of_points, pmin, pmax):


    sample_points = numpy.random.random(size=(number_of_points, 3))

    return scale(sample_points, pmin, pmax)
