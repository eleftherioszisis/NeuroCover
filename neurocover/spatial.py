import numpy
import scipy.spatial
from .bounding_box import vectorized_AABB_tapered_capsule
from .distance import points_to_segment
from .math import vectorized_dot, rowwise_dot

def is_point_inside_segment_bb(point, seg_start, seg_end, seg_start_r, seg_end_r):

    xmin, ymin, zmin, xmax, ymax, zmax = AABB_tapered_capsule(seg_start, seg_end, seg_start_r, seg_end_r)

    return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax and zmin <= point[2] <= zmax


def points_inside_convex_hull(points, convex_hull):

    dl = scipy.spatial.Delaunay(convex_hull.points[convex_hull.vertices])

    return dl.find_simplex(points) >= 0


def points_inside_capsule(points, p0, p1, r0, r1):

    distx, ts = points_to_segment(points, p0, p1, return_t=True)

    # varying radius capsule for projection to point p_t
    r_ts = r0  + (r1 - r0) * ts

    return (distx < r_ts) & ~numpy.isclose(distx, r_ts)


def points_inside_cylinder(points, p0, p1, r0, r1):

    seg_vector = p1 - p0
    pnt_vectors = points - p0

    dots = vectorized_dot(pnt_vectors, seg_vector)

    seg_length_sq = numpy.dot(seg_vector)

    return points_inside_capsule(points, p0, p1, r0, r1) & (dots >= 0.) & (dots <= seg_length_sq)


def is_inside(node_data, edges, sample_points, inflation_coefficient, inclusion_func=points_inside_capsule):
    """ Given the node_data and edges for segments it checks which
    sample points are inside inflated by the coefficient
    """
    points = node_data[:, (0, 1, 2)]
    radii = node_data[:, 3] + inflation_coefficient

    starts = points[edges[:, 0]]
    ends = points[edges[:, 1]]

    radii_starts = radii[edges[:, 0]]
    radii_ends = radii[edges[:, 1]]

    mask = numpy.zeros(len(sample_points), dtype=numpy.bool)

    bbs = vectorized_AABB_tapered_capsule(starts, ends, radii_starts, radii_ends)

    mask_inside = numpy.zeros(len(sample_points), dtype=numpy.bool)
    idx = numpy.arange(len(sample_points), dtype=numpy.int)

    for n in xrange(len(edges)):

        xmin, ymin, zmin, xmax, ymax, zmax = bbs[n]

        pidx = idx[~mask_inside]

        # find points inside edge bounding box
        mask = (xmin <= sample_points[pidx, 0]) & (sample_points[pidx, 0] <= xmax) & \
               (ymin <= sample_points[pidx, 1]) & (sample_points[pidx, 1] <= ymax) & \
               (zmin <= sample_points[pidx, 2]) & (sample_points[pidx, 2] <= zmax)

        if not mask.any():
            continue

        pidx = pidx[mask]

        mask_in = inclusion_func(sample_points[pidx], starts[n], ends[n], radii_starts[n], radii_ends[n])

        if not mask_in.any():
            continue

        pidx = pidx[mask_in]

        mask_inside[pidx] = True

    return mask_inside
