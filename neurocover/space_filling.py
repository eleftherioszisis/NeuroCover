import numpy
import rtree
from itertools import izip
import neurom
import matplotlib.pyplot as plt
from collections import namedtuple
from scipy.spatial import cKDTree

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation

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

def _bulk_loader_generator_tapered_capsules(p0s, p1s, r0s, r1s):

    for n, (p0, p1, r0, r1) in enumerate(izip(p0s, p1s, r0s, r1s)):
        yield (n, AABB_tapered_capsule(p0, p1, r0, r1), (p0, p1, r0, r1))

def _make_property(dims):

    p = rtree.index.Property()

    p.dimension=dims

    return p

def spatial_index(p0s, p1s, r0s, r1s):

    p = _make_property(3)

    gen_data = _bulk_loader_generator_tapered_capsules(p0s, p1s, r0s, r1s)

    index = rtree.index.Index(gen_data, properties=p)

    return index


def vector_projection(vectors1, vectors2):
    """ a . b * b / (|b| . |b|)
    """
    ab = numpy.sum(vectors1 * vectors2, axis=1)
    bb = numpy.sum(vectors2 * vectors2, axis=1)
    return  (ab / bb)[:, numpy.newaxis] * vectors2 

def rowwise_dot(vectors1, vectors2):

    return np.sum(vectors1 * vectors2, axis=1)

def closest_edges_to_points(starts, ends, points):

    edge_centers = 0.5 * (ends + starts)

    t = cKDTree(edge_centers)

    return t.query(points)


def distance_point_to_edge(points, edge_starts, edge_ends):

    ABs = edge_ends - edge_starts
    APs = points - edge_starts
    BPs = points - edge_ends

    cprod = numpy.cross(APs, ABs)

    distx =  numpy.linalg.norm(cprod, axis=1) / numpy.linalg.norm(ABs, axis=1)

    mask = (rowwise_dot(APs, ABs) <= 0.) | (rowwise_dot(BPs, ABs) >= 0.)
    distx[mask] = np.inf

    return distx

def is_inside(node_data, edges, sample_points, inflation_coefficient):

    points = node_data[:, (0, 1, 2)]
    radii = node_data[:, 3]

    starts = points[edges[:, 0]]
    ends = points[edges[:, 1]]

    radii_starts = radii[edges[:, 0]]
    radii_ends = radii[edges[:, 1]]

    edge_vectors = ends - starts

    edge_radii = 0.5 * (radii_starts + radii_ends)

    #_, edge_idx = closest_edges_to_points(starts, ends, sample_points)

    mask = numpy.zeros(len(sample_points), dtype=numpy.bool)

    edge_lengths = np.linalg.norm(edge_vectors, axis=1)
    edge_lengths_squared = edge_lengths ** 2

    factor = (1. + inflation_coefficient) 

    factor_radii_squared = (factor * edge_radii) ** 2

    """
    index = spatial_index(starts, ends, radii_starts, radii_ends)

    for i, point in enumerate(sample_points):

        try:

            edge_index = index.intersection((point[0], point[1], point[2], point[0], point[1], point[2])).next()

            p0 = starts[edge_index]
            p1 = ends[edge_index]
            r = edge_radii[edge_index]

            AP = point - p0
            BP = point - p1
            AB = p1 - p0

            APAB = np.dot(AP, AB)b 

            if APAB >= 0. and np.dot(BP, AB) <= 0.:

                proj = APAB * AB / edge_lengths_squared[edge_index]
                distance = np.linalg.norm(proj - AP)

                mask[i] = distance <= factor * r

        except StopIteration:
            pass




    """
    for i, point in enumerate(sample_points):
        for j in xrange(len(edges)):

            AP = point - starts[j]
            BP = point - ends[j]
            AB = edge_vectors[j]

            if np.dot(AP, AB) <= 0. or np.dot(BP, AB) >= 0.:
                continue

            proj = np.dot(AP, AB) * AB / edge_lengths_squared[j]

            distance2 = np.dot(proj - AP, proj - AP)

            #distance = np.linalg.norm(np.cross(AP, AB)) / edge_lengths[j]

            if distance2 <=  factor_radii_squared[j]:

                mask[i] = True
                break

    """
    mask = numpy.zeros(len(sample_points), dtype=numpy.bool)
    #edge_idx = numpy.full(len(sample_points), fill_value=n, dtype=numpy.int)
    edges_template = np.ones(len(sample_points), dtype=np.int)

    for n in xrange(len(edges)):

        edge_idx = edges_template * n
        
        pvecs = sample_points - starts[n]
        evecs = ends[edge_idx] - starts[edge_idx]

        distx = distance_point_to_edge(sample_points, starts[edge_idx], ends[edge_idx])

        mask |= distx <= (1 + inflation_coefficient) * edge_radii[edge_idx]
    """
    return mask


def gen_points_in_bb(N, pmin, pmax):

    xmin, ymin, zmin = pmin
    xmax, ymax, zmax = pmax

    sample_points = numpy.random.random(size=(N, 3))

    sample_points[:, 0] = xmin + (xmax - xmin) * sample_points[:, 0]
    sample_points[:, 1] = ymin + (ymax - ymin) * sample_points[:, 1]
    sample_points[:, 2] = zmin + (zmax - zmin) * sample_points[:, 2]

    return sample_points


def bb_nrn(neuron):
    points = neuron.points

    xmin = points[:, 0].min()
    ymin = points[:, 1].min()
    zmin = points[:, 2].min()

    xmax = points[:, 0].max()
    ymax = points[:, 1].max()
    zmax = points[:, 2].max()

    return numpy.array([xmin, ymin, zmin]), numpy.array([xmax, ymax, zmax])


def test(filename):

    """
    points = numpy.array([[0., 0., 0.], [0.5, 0.5, 0.5], [1., 1., 1.], [0.5, 0.5, 0.], [0.5, -0.5, 0.]])

    node_data = numpy.array([[0., 0., 0., 0.1],
                              [0.5, 0.5, 0.5, 0.1],
                              [1., 1., 1., 0.05], 
                              [0.5, 0.5, 0., 0.02], 
                              [0.5, -0.5, 0., 0.02]])

    edges = numpy.array([[0, 1], [1, 2], [1, 3], [1, 4]])
    pmin, pmax = np.array([0.,0.,0.]), np.array([1.,1.,1.])
    """
    nrn = neurom.load_neuron(filename)

    node_data, edges = h5_conversion(nrn)

    

    pmin, pmax = bb_nrn(nrn)

    N = 10000
    sample_points = gen_points_in_bb(N, pmin, pmax)

    volume = numpy.prod(pmax - pmin)


    radii = []
    coverage = []

    percentage = 0.
    r = 0.

    results = []
    while np.abs(percentage - 1.0) > 0.01:

        mask = is_inside(node_data, edges, sample_points, r)

        percentage = float(mask.sum()) / float(N)

        radii.append(r)
        coverage.append(percentage)

        results.append((sample_points[mask], r, percentage))

        r += 3.

        print r, percentage, mask.sum()

    scatter_anim(results, r, node_data, edges)


def h5_conversion(nrn):

    import numpy as np

    all_pdata = []
    all_edges = []

    for neurite in nrn.neurites:

        neu_points = neurite.points.tolist()

        N = len(all_pdata)

        neu_edges = [[i, i + 1] for i in xrange(N, len(neu_points) + N - 1)]

        all_pdata.extend(neu_points)
        all_edges.extend(neu_edges)

    all_pdata = np.asarray(all_pdata)

    all_edges = np.asarray(all_edges)
    """
    f = plt.figure()
    ax = f.add_subplot(111, projection='3d')


    segments = [[all_pdata[i, :3].tolist(), all_pdata[j, :3].tolist()] for i, j in all_edges]

    lc = Line3DCollection(segments)

    ax.add_collection3d(lc)

    plt.show()
    """

    return all_pdata, all_edges


def scatter_anim(results, max_r, node_data, edges):

    xmax = len(results)
    """
    def update_graph(num):

        x, y, z = results[num].T

        graph._offsets3d = (x, y, z)
        title.set_text('3D Test, time={}'.format(num))
        return title, graph, 
    """

    def update_graph(num):
        x, y, z = results[num][0].T

        r = results[num][1]
        coverage = results[num][2]

        scatter_3d.set_data (x, y)
        scatter_3d.set_3d_properties(z)

        line.set_xdata(numpy.append(line.get_xdata(), r))
        line.set_ydata(numpy.append(line.get_ydata(), coverage))

        #title.set_text('3D Test, time={}'.format(num))
        return line, scatter_3d, 

    print results[0]
    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Inflation Coefficient')
    ax2.set_ylabel('Volume Coverage %')

    ax2.set_xlim(0., max_r)
    ax2.set_ylim(0., 1.)
    #title = ax

    x, y, z = results[0][0].T

    segments = [[node_data[i, :3].tolist(), node_data[j, :3].tolist()] for i, j in edges]

    lc = Line3DCollection(segments)

    scatter_3d, = ax1.plot(x, y, z, color='k', linestyle="", marker='o', alpha=0.5, markersize=1)

    ax1.add_collection3d(lc)
    line,  = ax2.plot(results[0][1], results[0][2], color='k')

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, frames=range(0, xmax), blit=True)

    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=2000)

    ani.save('test2.mp4', writer=writer)










