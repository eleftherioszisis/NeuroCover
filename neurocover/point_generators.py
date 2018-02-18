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


def on_cylinder_surface(number_of_points, height, radius):

	phis = numpy.random.uniform(low=0., high=2. * numpy.pi, size=number_of_points)
	zs = numpy.random.uniform(low=- 0.5 * height, high=0.5 * height, size=number_of_points)

	points = numpy.empty((number_of_points, 3), dtype=numpy.float)

	points[:, 0] = radius * numpy.cos(phis)
	points[:, 1] = radius * numpy.sin(phis)
	points[:, 2] = zs

	return points