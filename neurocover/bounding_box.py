import numpy

def AABB_sphere(center, radius):
    ''' Returns the bounding box of a sphere given its center and radius.
    (xmin, ymin, zmin, xmax, ymax, zmax)
    '''
    return (center[0] - radius, center[1] - radius, center[2] - radius,
            center[0] + radius, center[1] + radius, center[2] + radius)


def AABB_tapered_capsule(cap1_center, cap2_center, cap1_radius, cap2_radius):

    xmin1, ymin1, zmin1, xmax1, ymax1, zmax1 = AABB_sphere(cap1_center, cap1_radius)
    xmin2, ymin2, zmin2, xmax2, ymax2, zmax2 = AABB_sphere(cap2_center, cap2_radius)

    return (min(xmin1, xmin2),
            min(ymin1, ymin2),
            min(zmin1, zmin2),
            max(xmax1, xmax2),
            max(ymax1, ymax2),
            max(zmax1, zmax2))


def vectorized_AABB_sphere(centers, radii):

    expanded_radii = numpy.tile(radii, (3, 1)).T

    return centers - expanded_radii, centers + expanded_radii


def vectorized_AABB_tapered_capsule(cap1_centers, cap2_centers, cap1_radii, cap2_radii):

    C1_min, C1_max = vectorized_AABB_sphere(cap1_centers, cap1_radii)
    C2_min, C2_max = vectorized_AABB_sphere(cap2_centers, cap2_radii)

    A = numpy.empty((len(cap1_centers), 6))

    A[:, 0] = numpy.min((C1_min[:, 0], C2_min[:, 0]), axis=0)
    A[:, 1] = numpy.min((C1_min[:, 1], C2_min[:, 1]), axis=0)
    A[:, 2] = numpy.min((C1_min[:, 2], C2_min[:, 2]), axis=0)
    A[:, 3] = numpy.max((C1_max[:, 0], C2_max[:, 0]), axis=0)
    A[:, 4] = numpy.max((C1_max[:, 1], C2_max[:, 1]), axis=0)
    A[:, 5] = numpy.max((C1_max[:, 2], C2_max[:, 2]), axis=0)

    return A
