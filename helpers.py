import numpy as np
import numpy.linalg as npl
from matplotlib import pyplot as plt
from matplotlib import cbook

# Returns a 2d image that encodes a light direction from light_dir within its RGB components
# Light 2d will have the same dimensions as original_image
#
# light_dir: a 3 dimensional vector representing the desired light direction, passed in as a list i.e. [0, 0, 1]
# original_image: the image that we are trying to relight
def get_new_light_dir(light_dir, original_image):
    light_dir = np.array(light_dir)
    
    # y axis is inverted for some reason
    light_dir[1] = -light_dir[1]
    
    # normalize the input light vector
    magnitude = np.sum(light_dir ** 2)
    normalized_light_dir = light_dir / magnitude
    
    new_light_dir = np.zeros(np.shape(original_image))
    new_light_dir[:] = normalized_light_dir
    
    return new_light_dir

# Returns shading for a single light source
#
# light_dir: a 2d image that encodes a light direction within its RGB components, must be the same dimension as the image
# path: the folder path to where the image components (like normals and albedo) are stored
def get_shading(light_dir, path):
    shading_original = plt.imread('./church/ours_shd.png')
    
    # Read and normalize the normals
    normals = plt.imread(path + 'normals.png')
    normals = (normals * 2) - 1
    
    # Dot product between normals and light dir to create shading 
    # (np.dot doesn't work for more than 2 dimensions)
    new_shading = normals * light_dir
    new_shading = np.sum(new_shading, axis=2)
    
    # Mask out the sky
    normals_for_mask = np.round(np.abs((normals)))
    mask = np.all(normals_for_mask == [0, 0, 0], axis=2)

    new_shading[mask] = shading_original[mask]
    new_shading = np.dstack([new_shading, new_shading, new_shading])
    
    return new_shading

# Combines multiple shadings into one normalized shading to allow for ambient light and secondary light sources
# IMPORTANT: if the shadings are directly opposite of each other (like strict left + strict right + strict top + strict bottom),
# they will CANCEL EACH OTHER OUT
#
# shading_list: the list of all shadings that we want to combine
# multiplier_list: the list of multipliers for each shading that represent how much of that shading we want in our image
# indexes of the multipliers and shading lists must align (i.e., shading at index 1 will look for a multiplier at index 1)
def combine_shading(shading_list, multiplier_list):
    if len(shading_list) != len(multiplier_list):
        print("Error in combine_shading(..): shading list and multiplier list have different sizes")
        return
    
    total_multipliers = sum(multiplier_list)
    
    new_shading = np.zeros(np.shape(shading_list[0]));
    for i in range(len(shading_list)):
        new_shading += (shading_list[i] * multiplier_list[i])
    new_shading = new_shading / total_multipliers
    
    return new_shading

# Creates a new image using one shading
#
# shading: new shading for the image, either calculated or provided
# path: the folder path to where the image components (like normals and albedo) are stored
def get_image(shading, path):
    albedo = plt.imread(path + 'ours_alb.png')
    new_image = albedo * shading
    new_image = new_image ** 0.4545 # gamma correct
    new_image = np.nan_to_num(new_image) # get rid of nan values to get rid of blurriness where NaNs are
    
    return new_image

def main():
    # this is just an example of how we can use the above helpers to create a new relit image
	# right now it just creates an ambient light from all directions

	path = ('./church/')
	img = plt.imread(path + 'input.png')

	# Get the lights for each direction we want to use for ambient light
	top_light = get_new_light_dir([0, -1, 0], img)
	bottom_light = get_new_light_dir([0, 1, 0.3], img)
	left_light = get_new_light_dir([1, 0.3, 0.3], img)
	right_light = get_new_light_dir([-1, 0.3, 0.3], img)
        
	# Get the shading with each new light direction
	top_shading = get_shading(top_light, path)
	bottom_shading = get_shading(bottom_light, path)
	left_shading = get_shading(left_light, path)
	right_shading = get_shading(right_light, path)
        
	# Combine each shading (this is where you would include the main light source as well)
	shading = [top_shading, bottom_shading, left_shading, right_shading]
	multipliers = [0.2, 0.2, 0.5, 0.5]

	shading = combine_shading(shading, multipliers)
        
	# Finally, create the image
	new_image = get_image(shading, path)	
        
	f = plt.figure()

    # set width, height, dpi
	f.set_dpi(120)
	plt.axis('off')
	plt.imshow(new_image)
        
main()
