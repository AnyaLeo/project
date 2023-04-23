import time

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
    shading_original = plt.imread('./data/church/ours_shd.png')

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
    new_image = new_image ** 0.4545  # gamma correct
    new_image = np.nan_to_num(new_image)  # get rid of nan values to get rid of blurriness where NaNs are

    return new_image


# Creates a 3D meshgrid compatible with the provided depth image
#
# depth_map: Depth image
# z_axis_resolution: The number of points along the z-axis (hyper-parameter)
def create_meshgrid(depth_map, z_axis_resolution=128):
    if np.max(depth_map) != z_axis_resolution and np.min(depth_map) != 0:
        # Note: depth_map should be read using "cv2.imread('.', cv2.IMREAD_ANYDEPTH)"
        depth_map = np.array(depth_map / np.max(depth_map) * z_axis_resolution, dtype='uint8')
        depth_map = z_axis_resolution - depth_map  # Convert disparity map to depth map
    x = np.arange(0, depth_map.shape[1])
    y = np.arange(0, depth_map.shape[0])
    z = np.arange(0, z_axis_resolution)

    xx, yy, zz = np.meshgrid(x, y, z, sparse=True)
    return xx, yy, zz


# Fills a 3D meshgrid with light direction vectors stored at each 3D coordinate. Also returns the magnitude of the
# light vector at each point (magnitude := distance from origin)
#
# (x_point, y_point, z_point): Origin of the light
# (xx, yy, zz): Sparse meshgrid
def unit_direction_3d(x_point, y_point, z_point, xx, yy, zz):
    t0 = time.perf_counter()
    point_origin = np.array([x_point, y_point, z_point])
    points_mesh = np.array([xx, yy, zz])
    magnitude = np.linalg.norm(points_mesh - point_origin)
    t1 = time.perf_counter()
    print(f"Time to calculate magnitudes = {t1 - t0}")
    t0 = time.perf_counter()
    directions_meshgrids = (points_mesh - point_origin)
    direction_x = np.ones_like(magnitude) * directions_meshgrids[0]
    direction_y = np.ones_like(magnitude) * directions_meshgrids[1]
    direction_z = np.ones_like(magnitude) * directions_meshgrids[2]
    direction_vectors = np.stack((direction_x, direction_y, direction_z), axis=-1)
    t1 = time.perf_counter()
    print(f"Time to stack direction vectors = {t1 - t0}")
    t0 = time.perf_counter()
    unit_direction_vectors = direction_vectors / magnitude[..., np.newaxis]
    t1 = time.perf_counter()
    print(f"Time to make direction vectors unit length = {t1 - t0}")
    return unit_direction_vectors, magnitude


# Get image surface in 3D
#
# image: Source image or normals
# depth: Depth map
# light_vector_field: Dense 3D light vector field (from unit_direction_3D)
# light_vector_magnitude: Distance of each light_vector from the light source (from unit_direction_3D)
def get_surface_lighting(image, depth, light_vector_field, light_vector_magnitude, spotlight_radius=0):
    num_rows, num_cols, _ = image.shape
    surface_light_directions = np.zeros_like(image, dtype=float)
    for i in range(num_rows):
        for j in range(num_cols):
            d = depth[i, j] - 1
            m = light_vector_magnitude[i, j, d]
            # Adjust c1 and c2 to control attenuation, f: 1 / 1.0 + c1*m + c2*m^2
            attenuation = 1.0 / 1.0 + 0.00005 * m + 0 * m ** 2
            surface_light_directions[i, j] = light_vector_field[i, j, d] * attenuation
    return surface_light_directions


def get_surface_lighting_vectorized(image, depth, light_vector_field, light_vector_magnitude, spotlight_radius=0):
    def cast_shadows(lvf, lvm):
        pass

    num_rows, num_cols, _ = image.shape
    row_idxs = np.arange(num_rows)[:, None]
    col_idxs = np.arange(num_cols)[None, :]
    depth_vals = depth - 1
    light_mags = light_vector_magnitude[row_idxs, col_idxs, depth_vals]
    attenuation = 1.0 / 1.0 + 0.00005 * light_mags
    surface_light_directions = light_vector_field[row_idxs[:, None], col_idxs[None, :], depth_vals] * attenuation[..., None]
    return surface_light_directions


def apply_surface_lighting(light_vectors, magnitudes, normals, albedo, intensity=20, light_colour=None):
    if light_colour is None:
        light_colour = [1, 1, 1]
    light_direction_times_normals = light_vectors * normals
    new_shading = np.sum(light_direction_times_normals, axis=2) / 3
    new_shading = np.dstack([new_shading, new_shading, new_shading]) * light_colour
    new_image = albedo * new_shading * intensity
    return new_image
