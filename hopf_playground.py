import imageio.v3 as iio
from pathlib import Path


from hopf_fibration import *





'''
File-saving stuff
'''

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
    files = []
    for i in range(n):
        frame = my_frames[i]
        fig = create_fig()
        scale_axes(fig)
        plot_hopf_fibers(frame, fig, color_function)
        file_name = name + "_" + int_to_string(i, max_digits) + ".png"
        files.append(file_name)
        fig.savefig(file_name)
        plt.close()
    return files



'''
Runs
'''

def runs(test_environment, show_animations, save_frames, make_gifs, example_data):
    
    examples = example_data[0]
    example_dir = example_data[1]
    example_names = example_data[2]
    example_coloring = example_data[3]
    example_filenames = [example_dir[i]+"/"+example_names[i] for i in range(len(examples))]
    

    if test_environment:
        fig = init_fig()
        thing = test_function(fig)
        plt.show()


    if save_frames:
        for i in range(len(examples)):
            frames_to_files(examples[i], example_filenames[i], color_function=example_coloring[i])


    if show_animations:
        figs = [init_fig(R=4, r=2) for i in range(len(examples))]
        anis = [make_animation(figs[i], examples[i], default_color) for i in range(len(examples))]
        plt.show()
        

    # Makes gif
    if make_gifs:
        for i in range(len(examples)):
            files = frames_to_files(examples[i], example_filenames[i], color_function=example_coloring[i])

            frames = np.stack([iio.imread(file_name) for file_name in files], axis=0)
            iio.imwrite(example_names[i]+".gif", frames, loop=0)





'''
Test function
'''

def test_function(fig):
    
    # Add your test stuff here to figure fig
    frame_no = 24
    n = 24
    a, b, c = (1,2,-1)
    
    more_points = great_circles_axis(a, b, c, frame_no, n)
    
    my_ani = make_animation(fig, more_points)
    
    
    return my_ani
    
    



'''
Basic settings
'''


frame_no = 60

n = 24

a0 = 0
b0 = 0
c0 = 1

latitude_points = latitudes(frame_no, n)
longitude_points = longitudes(frame_no, n)
my_points = great_circles_axis(a0, b0, c0, frame_no, n)

examples = [latitude_points, longitude_points, my_points]
example_dir = ["latitudes", "longitudes", "my_points"]
example_names = ["latitudes", "longitudes", "my_points"]
example_coloring = [default_color for example in examples]

example_data = (examples, example_dir, example_names, example_coloring)

test_environment = False
show_animations = False
save_frames = False
make_gifs = False

runs(test_environment, show_animations, save_frames, make_gifs, example_data)
