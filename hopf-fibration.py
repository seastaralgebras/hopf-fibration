import pdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation

def fiber(point_euc):
    
    def circle(point, t):
        x, y, z = point
        if x == 1:
            a = np.cos(t)
            b = np.sin(t)
            c = 0
            d = 0
        elif x == -1:
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

def stereographic_projection(point_s3):
    a, b, c, d = point_s3
    if a == 1:
        return (np.inf, np.inf, np.inf)
    else:
        return (b/(1-a), c/(1-a), d/(1-a))
        

def spherical_to_euclidean(phi, theta):
    x = np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta) * np.sin(phi)
    return (x, y, z)

def s2_to_fiber(phi, theta):
    return lambda t : stereographic_projection(fiber(spherical_to_euclidean(phi, theta))(t))

def default_color(phi, theta):
    return colors.hsv_to_rgb([.5, 1, 1])

def theta_color(phi, theta):
    return colors.hsv_to_rgb([(theta/(2 * np.pi)) % 1, 1, 1])

def phi_color(phi, theta):
    return colors.hsv_to_rgb([(phi/(np.pi)) % 1, 1, 1])

def plot_hopf_fiber(phi, theta, fig, color_function=default_color):
    ax_s3 = fig.get_axes()[0]
    ax_s2 = fig.get_axes()[1]
    circle = np.linspace(0, 2 * np.pi, num=360)
    plot_color = color_function(phi, theta)
    x = [s2_to_fiber(phi, theta)(t)[0] for t in circle]
    y = [s2_to_fiber(phi, theta)(t)[1] for t in circle]
    z = [s2_to_fiber(phi, theta)(t)[2] for t in circle]
    ax_s3.plot(x, y, z, color=plot_color)
    x0, y0, z0 = spherical_to_euclidean(phi, theta)
    ax_s2.plot(x0,y0,z0, color=plot_color, marker='o', markersize=6)

def plot_hopf_fibers(test_points, fig, color_function=default_color):
    ax_s3 = fig.get_axes()[0]
    ax_s2 = fig.get_axes()[1]
    for point in test_points:
        phi, theta = point
        plot_hopf_fiber(phi, theta, fig, color_function)


def create_fig():
    fig = plt.figure(figsize=plt.figaspect(0.5))

    ax_s3 = fig.add_subplot(1, 2, 1, projection='3d')
    ax_s2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    return fig


def scale_axes(fig):
    ax_s3 = fig.get_axes()[0]
    ax_s2 = fig.get_axes()[1]
    
    ax_s3.cla()
    ax_s3.set_autoscale_on(False)
    R = 4
    ax_s3.set_xlim((-R, R))
    ax_s3.set_ylim((-R, R))
    ax_s3.set_zlim((-R, R))
    
    ax_s2.cla()
    r = 2
    ax_s2.set_xlim((-r,r))
    ax_s2.set_ylim((-r,r))
    ax_s2.set_zlim((-r,r))

    
def legend_method(fig):
    for ax in fig.get_axes():
        ax.legend()

def int_to_string(n, max_digits, base=10):
    my_string = str(n)
    return "0"*(max_digits-len(my_string)) + my_string

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


def euclidean_to_spherical(point):
    x0, y0, z0 = point
    norm = np.sqrt(x0**2 + y0**2 + z0**2)
    x = x0 / norm
    y = y0 / norm
    z = z0 / norm
    phi = np.arccos(x)
    if 1 - x**2 <= 0:
        return (phi, 0)
    else:
        ratio_0 = y / np.sin(phi)
        ratio_1 = z / np.sin(phi)
        if ratio_1 > 1:
            theta = np.pi / 2
        elif ratio_1 < -1:
            theta = -np.pi / 2
        else:
            theta = np.arcsin(ratio_1)
        if ratio_0 > 0:
            return (phi, theta)
        elif ratio_0 < 0:
            return (phi, np.pi - theta)
        

def great_circle(a, b, c, t):
    x = (c / np.sqrt(a**2 + c**2))*np.cos(t) - ((c * (a**2))/(a**2 + c**2))*np.sin(t)
    y = b * np.sin(t)
    z = -(a / np.sqrt(a**2 + c**2))*np.cos(t) - ((a * c**2) / (a**2 + c**2))*np.sin(t)
    return (x, y, z)

def great_circle_spherical(a, b, c, t):
    return euclidean_to_spherical(great_circle(a, b, c, t))

def great_circle_fn(a, b, c):
    return lambda t : great_circle(a, b, c, t)

def great_circle_spherical_fn(a, b, c):
    return lambda t : great_circle_spherical(a, b, c, t)





k = 12
n = 2*k

phi_values = [np.pi * (i / k) for i in range(k)]
theta_values = [2 * np.pi * (i / n) for i in range(n)]
longitudes = [[(phi_values[i], theta) for i in range(k)] + [(np.pi - phi_values[i], theta + np.pi) for i in range(k)] for theta in theta_values]
latitudes = [ [(phi, theta) for theta in theta_values] for phi in phi_values] + [ [(np.pi - phi, theta) for theta in theta_values] for phi in phi_values]

a0 = 1
b0 = 1
c0 = 1
frame_no = 100
my_circles = [great_circle(a0, b0, c0, 2*np.pi*i / frame_no) for i in range(1, frame_no)]
my_points = [[great_circle_spherical(a, b, c, 2*np.pi*i / n) for i in range(1, n)] for a, b, c in my_circles]



# print(my_points)

# frames_to_files(latitudes, "latitudes/latitudes", theta_color)
# frames_to_files(longitudes, "longitudes/longitudes", phi_color)
frames_to_files(my_points, "great-circles-01/frames", color_function = theta_color)



# fig1 = create_fig()
# fig2 = create_fig()

# legend_method(fig1)
# legend_method(fig2)






def gen_animation(i, fig, my_points, color_function=default_color):
    scale_axes(fig)
    
    length = len(my_points)
    j = i % length
    
    test_points = my_points[j]
    plot_hopf_fibers(test_points, fig, color_function)


# ani1 = animation.FuncAnimation(fig1, gen_animation, fargs=[fig1, latitudes, theta_color])
# ani2 = animation.FuncAnimation(fig2, gen_animation, fargs=[fig2, longitudes, phi_color])

# ani1.save('latitudes', fps=6)
# ani2.save('longitudes', fps=6)
# ani1.to_html5_video(embed_limit=100)
# ani1.to_jshtml(embed_frames=100)

# plt.show()

