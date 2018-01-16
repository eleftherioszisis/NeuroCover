from .bounding_box import AABB_tapered_capsule


def is_point_inside_segment_bb(point, seg_start, seg_end, seg_start_r, seg_end_r):

    xmin, ymin, zmin, xmax, ymax, zmax = AABB_tapered_capsule(seg_start, seg_end, seg_start_r, seg_end_r)

    return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax and zmin <= point[2] <= zmax
