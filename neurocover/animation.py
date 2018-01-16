import numpy
import matplotlib.pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

def space_filling_inflation(results, node_data, edges, bb_point_min, bb_point_max, out_file='animation.mp4'):

    xmax = len(results)
    rmax = results[-1][1]

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

    fig = matplotlib.pylab.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_xlim(bb_point_min[0], bb_point_max[0])
    ax1.set_ylim(bb_point_min[1], bb_point_max[1])
    ax1.set_zlim(bb_point_min[2], bb_point_max[2])

    ax2 = fig.add_subplot(122)
    ax2.set_xlabel('Inflation Coefficient')
    ax2.set_ylabel('Volume Coverage %')

    ax2.set_xlim(0., rmax)
    ax2.set_ylim(0., 1.)

    x, y, z = results[0][0].T

    segments = [[node_data[i, :3].tolist(), node_data[j, :3].tolist()] for i, j in edges]

    lc = Line3DCollection(segments)

    scatter_3d, = ax1.plot(x, y, z, color='k', linestyle="", marker='o', alpha=0.5, markersize=1)


    ax1.add_collection3d(lc)
    line,  = ax2.plot(results[0][1], results[0][2], color='k')

    ani = matplotlib.animation.FuncAnimation(fig, update_graph, frames=range(0, xmax), blit=True)
    #Writer = matplotlib.animation.writers['ffmpeg']
    Writer = matplotlib.animation.writers['avconv']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=2000)

    ani.save(out_file, writer=writer)
