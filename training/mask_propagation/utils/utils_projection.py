import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def show_projection(image_path,x_projected, y_projected):
      # Load the image
    image = cv2.imread(image_path)

    # Visualize the projected point on the image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(x_projected, y_projected, c='red', marker='o', label='Projected Point')
    plt.legend()
    plt.show()
    
def project_3d_point_to_image(point_3d, K, C2W):
    # Create a 4x4 transformation matrix
    extrinsic_matrix = np.linalg.inv(C2W)
    # Project the 3D point onto the image
    homogeneous_point_3d = np.append(point_3d, 1)
    projected_point_homogeneous = K @ extrinsic_matrix @ homogeneous_point_3d

    # Normalize the homogeneous coordinates
    projected_point = projected_point_homogeneous[:2] / projected_point_homogeneous[2]

    # Extract the x and y coordinates of the projected point
    x_projected, y_projected = projected_point
    return int(y_projected),int(x_projected)

def draw_points(img_path, x_y_points):
    img1 = cv2.imread(img_path)[...,::-1]
    plt.clf()
    plt.imshow(img1)

    for point in x_y_points:
        x, y = point[0].__floor__(), point[1].__floor__()
        plt.scatter(x,y, color="blue", s=5)

    plt.show()

def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def sort_by_distance(points, cnt):
    points_copy = points[:]
    # Calculate the average of all points
    average_point = (sum(x for x, y in points_copy) / len(points_copy),
                    sum(y for x, y in points_copy) / len(points_copy))
    
    # Initialize a list to store the sorted points_copy
    selected_points = []

    while len(selected_points) < cnt:
        if not selected_points:
            # If the sorted_points list is empty, choose the closest point to the average point
            next_point = min(points_copy, key=lambda p: distance(average_point, p))
        else:
            # Find the point that is farthest from any point in sorted_points
            next_point = max(points_copy, key=lambda p: min(distance(p, sp) for sp in selected_points))

        # Append the next_point to the sorted_points list
        selected_points.append(next_point)

        # Remove the next_point from the list of points
        points_copy.remove(next_point)

    return selected_points

def sample_points_along_ray(K, extrinsics, point_3d, num_samples=10):
    # Transform 3D point to camera coordinates
    point_homogeneous = np.append(point_3d, 1)
    point_camera = np.dot(extrinsics, point_homogeneous)

    # Calculate the ray direction (unit vector) from camera to the point
    ray_direction = point_camera[:3] / np.linalg.norm(point_camera[:3])

    # Calculate step size for sampling
    step_size = np.linalg.norm(point_camera[:3]) / num_samples

    sampled_points = []

    for i in range(num_samples):
        sample_point = point_camera - i * step_size * np.array([ray_direction[0], ray_direction[1], ray_direction[2], 0])
        sampled_points.append(sample_point)

    # Convert the sampled points back to world coordinates
    sampled_points_world = np.array([np.dot(np.linalg.inv(extrinsics), point) for point in sampled_points])[:,:3]

    return sampled_points_world

# def project_all(points_3d, camera_dict_json, H, W):
#     K, C2W = np.array(camera_dict_json["K"]).reshape((4, 4)),np.array(camera_dict_json["C2W"]).reshape((4, 4))
#     proj = set()
#     for point_3d in points_3d:
#         p_img = project_3d_point_to_image(point_3d, K, C2W)
#         y, x = p_img
#         if 0<x<W and 0<y<H:
#             proj.add((int(x),int(y)))
#     return proj

def project_all(points_3d, camera_dict_json, H, W):
    K = np.array(camera_dict_json["K"]).reshape((4, 4))
    C2W = np.array(camera_dict_json["C2W"]).reshape((4, 4))

    # Create a 4x4 transformation matrix
    extrinsic_matrix = np.linalg.inv(C2W)

    # Homogeneous coordinates of all 3D points
    homogeneous_points_3d = np.hstack((points_3d, np.ones((len(points_3d), 1))))

    # Project all 3D points onto the image
    projected_points_homogeneous = np.dot(np.dot(K, extrinsic_matrix), homogeneous_points_3d.T)

    # Normalize the homogeneous coordinates
    projected_points = projected_points_homogeneous[:2] / projected_points_homogeneous[2]

    # Filter points within the image boundaries
    mask = (projected_points[0] >= 0) & (projected_points[0] < W) & (projected_points[1] >= 0) & (projected_points[1] < H)

    # Extract the x and y coordinates of the projected points
    x_projected, y_projected = projected_points[0, mask], projected_points[1, mask]

    proj = set([(int(x), int(y)) for x, y in zip(x_projected, y_projected)])

    return proj

def generate_rays_single(pixel_coords, K, C2W):
    num_pixels = pixel_coords.shape[0]

    pixels = np.column_stack((pixel_coords, np.ones(num_pixels)))
    pixels = pixels.T

    p = np.dot(np.linalg.inv(K)[:3, :3], pixels).squeeze()
    p /= np.linalg.norm(p, ord=2, axis=0)
    rays_v = np.dot(C2W[:3, :3], p).T

    return rays_v

def generate_rays(pixel_coords, K, C2W):
    num_pixels = pixel_coords.shape[0]

    pixels = np.column_stack((pixel_coords, np.ones(num_pixels)))
    pixels = pixels.T

    p = np.dot(np.linalg.inv(K)[:3, :3], pixels).squeeze()
    p /= np.linalg.norm(p, ord=2, axis=0)
    rays_v = np.dot(C2W[:3, :3], p).T

    return rays_v


def distance_3d(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

def sort_by_distance_3d(points, cnt):
    points_copy = points[:]
    
    # Calculate the average of all points
    average_point = (
        sum(x for x, y, z in points_copy) / len(points_copy),
        sum(y for x, y, z in points_copy) / len(points_copy),
        sum(z for x, y, z in points_copy) / len(points_copy)
    )
    
    # Initialize a list to store the sorted points_copy
    selected_points = []

    while len(selected_points) < cnt:
        if not selected_points:
            # If the sorted_points list is empty, choose the closest point to the average point
            next_point = min(points_copy, key=lambda p: distance_3d(average_point, p))
        else:
            # Find the point that is farthest from any point in sorted_points
            next_point = max(points_copy, key=lambda p: min(distance_3d(p, sp) for sp in selected_points))

        # Append the next_point to the sorted_points list
        selected_points.append(next_point)

        # Remove the next_point from the list of points
        points_copy.remove(next_point)

    return selected_points