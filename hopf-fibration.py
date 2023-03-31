import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation

'''
Coordinate stuff
'''

# stereographic_projection: point_s3 = (a, b, c, d) --> (x, y, z)
# point is given by Euclidean coordinates in R^4, output is in R^3 or infinity
# Takes a point in the 3-sphere and applies a stereographic projection to R^3, with (1, 0, 0, 0) being mapped to infinity
def stereographic_projection(point_s3):
    a, b, c, d = point_s3
    if a == 1:
        return (np.inf, np.inf, np.inf)
    else:
        return (b/(1-a), c/(1-a), d/(1-a))
        
# spherical_to_euclidean: point = (phi, theta) --> (x, y, z)
# For use on S^2
# Maps spherical coordinates to euclidean coordinates
def spherical_to_euclidean(point):
    phi, theta = point
    x = np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta) * np.sin(phi)
    return (x, y, z)

# # euclidean_to_spherical: point = (x0, y0, z0) --> (phi, theta)
# # Do we even use this here?
# # Maps euclidean coordinates to spherical coordinates.
# def euclidean_to_spherical(point):
#     x0, y0, z0 = point
#     norm = np.sqrt(x0**2 + y0**2 + z0**2)
#     x = x0 / norm
#     y = y0 / norm
#     z = z0 / norm
#     phi = np.arccos(x)
#     if 1 - x**2 <= 0:
#         return (phi, 0)
#     else:
#         ratio_0 = y / np.sin(phi)
#         ratio_1 = z / np.sin(phi)
#         if ratio_1 > 1:
#             theta = np.pi / 2
#         elif ratio_1 < -1:
#             theta = -np.pi / 2
#         else:
#             theta = np.arcsin(ratio_1)
#         if ratio_0 > 0:
#             return (phi, theta)
#         elif ratio_0 < 0:
#             return (phi, np.pi - theta)


'''
Preimage of the Hopf map
'''

# fiber: point_euc = (x, y, z) --> (t |--> (a,b,c,d))
# point_euc in Euclidean coordinates for a point on S^2, output is a function that maps an angle t to a point in the preimage of point_euc
# Takes a point on the 2-sphere (with its standard embedding in R^3) and looking at its preimage under the Hopf map, which will be a circle in the 3-sphere as realized as in its standard embedding in R^4.
def fiber(point_euc):
    
    def circle(point, t):
        x, y, z = point
        if x >= 1:
            a = np.cos(t)
            b = np.sin(t)
            c = 0
            d = 0
        elif x <= -1:
            a = 0
            b = 0
            c = np.cos(t)
            d = np.sin(t)
        else:
            a = np.sqrt((1+x) / 2) * np.cos(t)
            b = np.sqrt((1+x) / 2) * np.sin(t)
            c = (1 / np.sqrt(2 * (1+x))) * (-y * np.cos(t) + z * np.sin(t))
            d = (1 / np.sqrt(2 * (1+x))) * (z * np.cos(t) + y * np.sin(t))
        return (a, b, c, d)
    
    return lambda t : circle(point_euc, t)

# s2_to_fiber: point = (x, y, z) --> (t --> (x0, y0, z0))
# Input is from S^2, output is a function that maps t to the stereographic projection of fiber(point)(t) in R^3
# Takes the output of fiber(point) and applies a stereographic projection to it, with (1, 0, 0, 0) being mapped to infinity
def s2_to_fiber(point):
    return lambda t : stereographic_projection(fiber(point)(t))


'''
Plotting stuff
'''

# default_color: 0 <= color_parameter <= 1 --> rgb color
# Sets the hue of the point to the given color parameter
def default_color(color_param):
    return colors.hsv_to_rgb([color_param % 1, 1, 1])

# plot_hopf_fiber: point = (x0, y0, z0) OR point = (phi, theta)
# Defaults to assuming either spherical or euclidean coordinates. May update to accomodate other coordinate systems.
def plot_hopf_fiber(point, fig, color_param, color_function=default_color):
    if len(point) == 2:
        new_point = spherical_to_euclidean(point)
    elif len(point) == 3:
        new_point = point
    else:
        print(point)
        print("IDK man :/")
    x0,y0,z0 = new_point
    ax_s3 = fig.get_axes()[0]
    ax_s2 = fig.get_axes()[1]
    circle = np.linspace(0, 2 * np.pi, num=360)
    plot_color = color_function(color_param)
    x = [s2_to_fiber(new_point)(t)[0] for t in circle]
    y = [s2_to_fiber(new_point)(t)[1] for t in circle]
    z = [s2_to_fiber(new_point)(t)[2] for t in circle]
    ax_s3.plot(x, y, z, color=plot_color)
    ax_s2.plot(x0,y0,z0, color=plot_color, marker='o', markersize=6)

# plot_hopf_fibers: test_points = [(point, color_parameter)], point = spherical or euclidean, 0 <= color_parameter <= 1, color_function = function(color_parameter)
# For each point, color_parameter in test_points, plots point with color color_function(color_parameter)
# Good for single frames
def plot_hopf_fibers(test_points, fig, color_function=default_color):
    ax_s3 = fig.get_axes()[0]
    ax_s2 = fig.get_axes()[1]
    for point, color_parameter in test_points:
        plot_hopf_fiber(point, fig, color_parameter, color_function)

# create_fig(): --> fig
# makes two figures
def create_fig():
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax_s3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax_s2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    return fig

# scale_axes: fig; OPTIONAL: R=4 and r=2 for bounds of S^3 and S^2 viewing respectively, clear axes = True
# clears and resizes axes, sets autoscale to False
def scale_axes(fig, R=4, r=2, clear_axes=True):
    ax_s3 = fig.get_axes()[0]
    ax_s2 = fig.get_axes()[1]
    
    if clear_axes:
        ax_s3.cla()
        ax_s2.cla()
    
    ax_s3.set_autoscale_on(False)
    ax_s2.set_autoscale_on(False)
    
    ax_s3.set_xlim((-R, R))
    ax_s3.set_ylim((-R, R))
    ax_s3.set_zlim((-R, R))
    
    ax_s2.set_xlim((-r,r))
    ax_s2.set_ylim((-r,r))
    ax_s2.set_zlim((-r,r))

# legend_method: ?
# Idk
def legend_method(fig):
    for ax in fig.get_axes():
        ax.legend()


# init_fig: OPTIONAL: R=4 and r=2 for bounds of S^3 and S^2 viewing respectively --> fig
# creates, clears, and resizes axes, sets autoscale to False, returns fig
def init_fig(R=4, r=2):
    fig = create_fig()
    scale_axes(fig, R=R, r=r)
    legend_method(fig)
    return fig

# int_to_string: n = int, max_digits = int
# returns a string of n with 0's in front if number has fewer digits than max_digits
def int_to_string(n, max_digits):
    my_string = str(n)
    return "0"*(max_digits-len(my_string)) + my_string

# frames_to_files: my_frames = [[(point, color_param)]], name = string; OPTIONAL: color_function = default_color
# Each element of my_frames is an array of test_points for plot_hopf_fibers, saved with filename name and followed by frame number
def frames_to_files(my_frames, name, color_function=default_color):
    n = len(my_frames)
    max_digits = int(np.log10(n)+1)
    for i in range(n):
        frame = my_frames[i]
        fig = create_fig()
        scale_axes(fig)
        plot_hopf_fibers(frame, fig, color_function)
        legend_method(fig)
        file_name = name + "_" + int_to_string(i, max_digits) + ".png"
        fig.savefig(file_name)


'''
Point generation
'''

# projection: u, v = length n arrays
# Orthogonal projection of u onto v
def projection(u, v):
    u_dot_v = np.dot(u, v)
    norm_v = LA.norm(v)
    c = u_dot_v / (norm_v**2)
    return c * v

# gram_schmidt: basis = array of arrays --> orthonormal basis
# Applies Gram Schmidt process to basis
def gram_schmidt(basis):
    new_basis = []
    
    for u in basis:
        if len(new_basis)==0:
            v = np.multiply(1 / LA.norm(u), u)
            new_basis.append(v)
        else:
            v = u
            for e in new_basis:
                v = v - projection(v, e)
            v = np.multiply(1 / LA.norm(v), v)
            new_basis.append(v)
    
    return new_basis

# great_circle: a, b, c = real numbers, t real number (in 0 to 2*pi); OPTIONAL: r=1 --> (x, y, z)
# (a, b, c) is the vector of the axis that determines this great circle centered at the origin
def great_circle(a, b, c, t, r = 1):
    if a != 0:
        basis = [[a,b,c], [0,1,0],[0,0,1]]
    elif b != 0:
        basis = [[a,b,c], [1,0,0],[0,0,1]]
    else:
        basis = [[a,b,c], [1,0,0],[0,1,0]]
    new_basis = np.multiply(r, gram_schmidt(basis))
    u = new_basis[1]
    v = new_basis[2]
    vector = np.multiply(np.cos(t), u) + np.multiply(np.sin(t), v)
    
    return tuple(vector)

# great_circle_fn: a, b, c, (r=1) --> (t --> (x, y, z))
# function from t to great_circle(a, b, c, t, r)
def great_circle_fn(a, b, c, r = 1):
    return lambda t : great_circle(a, b, c, t, r = r)


'''
Frame generation
'''

# Returns a set of great circles intersecting the axis given by (a, b, c)
def great_circles_axis(a, b, c, frame_no, n):
    my_circles = [great_circle(a, b, c, 2*np.pi*i / frame_no) for i in range(0, frame_no)]
    
    frames = [[(great_circle(a0, b0, c0, 2*np.pi*i / n), i / n) for i in range(0, n)] for a0, b0, c0 in my_circles]
    
    return frames

# Returns the latitudes
def latitudes(frame_no, n):
    k = int(frame_no / 2)
    phis = [np.pi * (i / k) for i in range(k)] + [np.pi * (1 - i / k) for i in range(k)]
    thetas = [2 * np.pi * (i / n) for i in range(n)]
    
    frames = [ [ (spherical_to_euclidean((phi, theta)), i / n) for i, theta in enumerate(thetas)] for phi in phis ]
    
    return frames
    
# Returns the longitudes
def longitudes(frame_no, n):
    frames = great_circles_axis(1, 0, 0, frame_no, n)
    return frames
    




'''
Animation function
'''

# fig is the figure that you are animating
# my_points is the set of frames
# color_function = color function you wish to pass
def gen_animation(i, fig, my_points, color_function=default_color):
    scale_axes(fig)
    
    length = len(my_points)
    j = i % length
    
    test_points = my_points[j]
    plot_hopf_fibers(test_points, fig, color_function)


'''
Test function
'''

def test_function(fig):
    
    # Add your test stuff here
    
    return fig


'''
Basic settings
'''

frame_no = 60

n = 24

a0 = 1
b0 = 2
c0 = -1

latitude_points = latitudes(frame_no, n)
longitude_points = longitudes(frame_no, n)
my_points = great_circles_axis(a0, b0, c0, frame_no, n)

examples = [latitude_points, longitude_points, my_points]
example_names = ["latitudes/latitudes", "longitudes/longitudes", "my_points/my_points"]
example_coloring = [default_color for example in examples]

test_environment = False
save_frames = True
show_animation = False



'''
Runs
'''


if test_environment:
    fig = init_fig()
    test_function(fig)
    plt.show()


if save_frames:
    for i in range(len(examples)):
        frames_to_files(examples[i], example_names[i], color_function=example_coloring[i])


if show_animation:
    figs = [init_fig(R=4, r=2) for i in range(len(examples))]
    anis = [animation.FuncAnimation(figs[i], gen_animation, fargs=[figs[i], examples[i], default_color]) for i in range(len(examples))]
    plt.show()