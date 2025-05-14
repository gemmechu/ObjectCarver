import numpy as np
import os
import json
import glob
import cv2
import torch
import utils.utils_sdf as utils_sdf
import utils.utils_projection as utils_projection
import scipy
import ImageSegmentation as ImageSegmentation
import utils.utils_common as utils_common
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
import trimesh
import open3d as o3d
from datetime import datetime
import yaml
# import pyembree
from scipy import stats

class AutoSAM:
    def __init__(self, parent_dir, anchors,image_list, camera_dict_json,instance_category,sdf_network,thresh_ratio, mesh, surface_type="sdf"):
        self.parent_dir = parent_dir
        self.anchors = anchors
        self.image_list = image_list
        self.camera_dict_json = camera_dict_json
        self.instance_category = instance_category
        self.sdf_network = sdf_network 
        self.thresh_ratio = thresh_ratio
        self.H, self.W, _ = cv2.imread(image_list[0]).shape
        self.mesh = mesh
        self.surface_type = surface_type


    def get_annotated_data(self,anchor_dir):
        pts_annotated_by_category = {key: [] for key in self.instance_category.keys()}
        for anchor in self.anchors:
            mask_path = os.path.join(anchor_dir, anchor)
            K, C2W = np.array(self.camera_dict_json[anchor]["K"]).reshape((4, 4)), np.array(self.camera_dict_json[anchor]["C2W"]).reshape((4, 4))
            for obj in self.instance_category:
                try:
                    binary_mask = (utils_common.convert_image_to_binary(cv2.imread(mask_path), self.instance_category[obj]) / 255).squeeze()
                    binary_distance = scipy.ndimage.distance_transform_edt(binary_mask)
                    v_coords, u_coords = np.where(binary_distance > self.thresh_ratio)
                    pixel_coords = np.stack((u_coords, v_coords)).T
                    ray_directions = utils_projection.generate_rays(pixel_coords, K, C2W)
                    ray_origins = torch.tensor(C2W[:, 3][:3], dtype=torch.float)[:3].unsqueeze(0)
                    
                    if self.surface_type == "mesh":
                        ray_origins = ray_origins.repeat(ray_directions.shape[0],1)
                        index_tri, index_ray, points_itx = self.mesh.ray.intersects_id(ray_origins, ray_directions, return_locations=True, multiple_hits=False)
                    elif self.surface_type == "sdf":
                        ray_directions_torch = torch.tensor(ray_directions)
                        p, mask = utils_sdf.sphere_trace(self.sdf_network, ray_origins, ray_directions_torch, max_steps=20, eps=1e-3)
                        points_itx = p[np.where(mask == True)[0]].numpy()
                        
                    pts_annotated_by_category[obj].extend(points_itx)

                except:
                    print("mask_path: ", mask_path)
             
        return pts_annotated_by_category
    def get_seed(self, name, pts_annotated_by_category, thresh_proj, thresh_distance=1e-3, seed_cnt=15):
        seed = {}
        K, C2W = np.array(self.camera_dict_json[name]["K"]).reshape((4, 4)), np.array(self.camera_dict_json[name]["C2W"]).reshape((4, 4))
        origin = C2W[:, 3][:3]
        ray_origins = torch.tensor(C2W[:, 3][:3], dtype=torch.float)[:3].unsqueeze(0)
        
        for obj in self.instance_category:
            points3D = np.array(pts_annotated_by_category[obj]) 
            points_in_front = points_in_front_of_camera(C2W, points3D)
            if len(points_in_front) == 0:
                seed[obj] = []
                continue
            
            ray_directions = points_in_front - origin[:3]
            ray_directions /= np.linalg.norm(ray_directions, ord=2, axis=1, keepdims=True)
            ray_directions_torch = torch.tensor(ray_directions)
            if self.surface_type == "sdf":
                points_itx, mask = utils_sdf.sphere_trace(self.sdf_network, ray_origins, ray_directions_torch, max_steps=20, eps=1e-3)
            elif self.surface_type == "mesh":
                ray_origins_reshaped = ray_origins.repeat(ray_directions.shape[0],1)
                index_tri, index_ray, points_itx = self.mesh.ray.intersects_id(ray_origins_reshaped, ray_directions, return_locations=True, multiple_hits=False)
            pts = np.array(points_in_front)
            A, B = pts, np.array(points_itx)
            if self.surface_type == "mesh" and A.shape != B.shape: 
                A = A[index_ray] 
                
            distances = np.linalg.norm(A - B, axis=1)
            mask = distances < thresh_distance
            selected = A[mask]
            proj = utils_projection.project_all(selected, self.camera_dict_json[name], self.H, self.W)
            thresh_percentage = (len(proj)/len(points_in_front)) *100
            if thresh_percentage > thresh_proj and len(proj)>seed_cnt:
                proj = utils_projection.sort_by_distance(list(proj), cnt=seed_cnt)
                seed[obj] = proj
            else:
                proj = []
            seed[obj] = proj
        return  seed
    def get_seeds(self, pts_annotated_by_category, thresh_proj=15):
        sam_seeds_by_category = {}
        anchor_set =set(self.anchors)
        for img in self.image_list:
            name = img.split("/")[-1]
            sam_seeds_by_category[name] = self.get_seed(name, pts_annotated_by_category, thresh_proj=thresh_proj)

        return sam_seeds_by_category

    def export_pts_annotated(self, pts_annotated_by_category, instance_category, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for obj in instance_category:
            pts = np.array(pts_annotated_by_category[obj])
            if pts.shape[-1] != 3:
                continue
            utils_sdf.export_to_ply_trimesh(pts, os.path.join(out_dir, obj + ".ply"))
    
    def create_bounding_box_with_margin(self, points, margin, shape):
        h,w = shape
        if not points:
            return None

        min_x = max(min(points, key=lambda point: point[0])[0] - margin, 0)
        max_x = min(max(points, key=lambda point: point[0])[0] + margin, w)
        min_y = max(min(points, key=lambda point: point[1])[1] - margin, 0)
        max_y = min(max(points, key=lambda point: point[1])[1] + margin, h)

        bounding_box = [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

        return bounding_box

    def create_binary_image(self, h, w, bounding_box):
        # Create an empty black image with the specified height and width
        image = np.zeros((h, w,3), dtype=np.uint8)
        if not bounding_box:
            return image
        min_x, min_y = bounding_box[0]
        max_x, max_y = bounding_box[2]

        # Create a mask for the inside of the bounding box
        mask = np.zeros_like(image)
        mask[min_y:max_y, min_x:max_x] = (255,255,255)  

        return mask

    def create_amodal_mask(self, pts_annotated_by_category, outdir):
        for img_path in self.image_list:
            for category in self.instance_category:
                outpath = os.path.join(outdir,category)
                name = img_path.split("/")[-1]
                outpath_img = os.path.join(outpath, name)
                if os.path.exists(outpath_img):
                    continue
                points3D = np.array(pts_annotated_by_category[category]) 
                K, C2W = np.array(self.camera_dict_json[name]["K"]).reshape((4, 4)), np.array(self.camera_dict_json[name]["C2W"]).reshape((4, 4))
                points_in_front = points_in_front_of_camera(C2W, points3D)
                proj_points = utils_projection.project_all(points_in_front,self.camera_dict_json[name], self.H, self.W)
                
                bounding_box = self.create_bounding_box_with_margin(proj_points, margin=5,shape=(self.H,self.W))
                binary_image = self.create_binary_image(self.H, self.W, bounding_box)
                
                outpath = os.path.join(outdir,category)
                if not os.path.exists(outpath):
                    os.makedirs(outpath)
                
                plt.imsave(outpath_img, binary_image)


   

def points_in_front_of_camera(extrinsic_matrix, points):
    extrinsic_matrix = np.linalg.inv(extrinsic_matrix)
    # Append [0, 0, 0, 1] to make it a 4xN homogeneous coordinate
    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Transform points to camera coordinates
    points_camera = np.dot(extrinsic_matrix, points_homogeneous.T).T
    
    # Check if the Z-coordinate (depth) is negative for each point
    is_behind_camera = points_camera[:, 2] < 0
    
    # Return points that are in front of the camera
    return points[~is_behind_camera]
def remove_outliers(data, z_thresh=2.5):
    z_scores = np.abs(stats.zscore(data))
    filtered_data = data[(z_scores < z_thresh).all(axis=1)]
    return filtered_data

def load_pts(instance_category,out_dir_parent, iter):
    pts_annotated_by_category = {key: [] for key in instance_category.keys()}
    for obj in instance_category:
        try:
            pts_path = os.path.join(out_dir_parent, "pts_"+str(iter), obj+".ply")
            pts = trimesh.load(pts_path).vertices
        
            pts_annotated_by_category[obj].extend(pts)
        except:
            pts_path = os.path.join(out_dir_parent, "pts_"+str(iter-1), obj+".ply")
            pts = trimesh.load(pts_path).vertices
        
            pts_annotated_by_category[obj].extend(pts)
            
    return pts_annotated_by_category
def subsample(instance_category,out_dir_parent,num_samples,i):
    pts_annotated_by_category = {key: [] for key in instance_category.keys()}
    for obj in instance_category:
        pts_path = os.path.join(out_dir_parent, "pts_"+str(i), obj+".ply")
        pts = o3d.io.read_point_cloud(pts_path)
        sampled_points = pts.voxel_down_sample(num_samples)
        pts = np.asarray(sampled_points.points)
        pts_annotated_by_category[obj].extend(pts)
    return pts_annotated_by_category
def subsample_array(numpy_array_pts, num_samples):
    pts = o3d.geometry.PointCloud()
    # Assign the numpy array as points to the Open3D point cloud object
    pts.points = o3d.utility.Vector3dVector(numpy_array_pts)
    sampled_points = pts.voxel_down_sample(num_samples)
    pts = np.asarray(sampled_points.points)
    return pts
def remove_intersection_points(points1, points2):
    # Find intersection points
    intersection_indices_1 = []
    # intersection_indices_2 = []
    
    # Loop through points in the first point cloud and check if they exist in the second point cloud
    for i, point in enumerate(points1):
        if np.any(np.all(np.isclose(points2, point, atol=5e-3), axis=1)):
            intersection_indices_1.append(i)  # Add the index of the intersection point
    
    # Remove intersection points from the original point clouds
    points1 = np.delete(points1, intersection_indices_1, axis=0)
    # points2 = np.delete(points2, intersection_indices_2, axis=0)
    
    return points1
def remove_intersection(filtered_data):
    
    keys = list(filtered_data.keys())
    data_copy = filtered_data.copy()
    for i in range(len(keys)):
        key = keys[i]
        other_arrays = np.vstack([data_copy[k] for k in data_copy.keys() if k != key])
        filtered_data[key] = remove_intersection_points(filtered_data[key], other_arrays)
       
    return filtered_data
def start_segmentation(parent_dir,predictor,anchors,output_dir,thresh_ratio = 3.5, n_pass=1, surface_type="sdf", sdf_ckpt=None, sdf_conf=None, mesh_path=None, num_samples=0.005):
   
    #------------------------------------------------------#
    #                  Camera Json                    
    #------------------------------------------------------#
    camera_json_path = os.path.join(parent_dir, "cam_dict.json") 
    camera_dict_json = json.load(open(camera_json_path))
    instance_category = json.load(open(os.path.join(parent_dir, "instance_segmentation.json")))

    #------------------------------------------------------#
    #                   Get Surface Mesh or SDF            #
    #------------------------------------------------------#
    grand_dir, scene_name = "/".join(parent_dir.split("/")[:-1]), parent_dir.split("/")[-1]
    mesh,sdf_network = None,None
    if surface_type == "sdf":
        
        sdf_network = utils_sdf.get_sdf_network(sdf_ckpt, sdf_conf, "sdf_network_fine")

    elif surface_type == "mesh":
        mesh = trimesh.load_mesh(mesh_path)
        thresh_ratio = 4
    print("surface_type: ", surface_type)
    #------------------------------------------------------#
    #                  Anchor
    #------------------------------------------------------#

    image_list = sorted(glob.glob(os.path.join(parent_dir, "images/*.png")))
    print(image_list[0])
    H,W, _ = cv2.imread(image_list[0]).shape
    print(H,W)
    
    out_dir_parent = os.path.join(parent_dir, output_dir) 
    print("out_dir_parent: ",out_dir_parent)
    
    anchors_orig = anchors
   
    
    autoSam = AutoSAM(parent_dir, anchors_orig,image_list, camera_dict_json,instance_category,sdf_network,thresh_ratio, mesh, surface_type)
    # Record start time
    def do_first_pass(n_pass):
        start_time = datetime.now()

        
        for i in range(n_pass):
            print("get_annotated_data started")
            
            out_dir = os.path.join(out_dir_parent, "pts_"+str(i))
            
            anchor_dir =os.path.join(parent_dir, "anchor")  if i==0 else os.path.join(out_dir_parent, "mask_single_"+str(i-1))
            
            pts_annotated_by_category = autoSam.get_annotated_data(anchor_dir) if not os.path.exists(out_dir) else {key: [] for key in instance_category.keys()}
            
            if os.path.exists(out_dir):
                pts_annotated_by_category = subsample(instance_category,out_dir_parent,num_samples,i)
                
                filtered_data = pts_annotated_by_category
            else:
                filtered_data = {}
                for category in pts_annotated_by_category:
                    data = np.array(list(map(list, pts_annotated_by_category[category])))
                    # print("data: ",data)
                    numpy_array_pts = remove_outliers(data, z_thresh=2.5)
                    filtered_data[category] = subsample_array(numpy_array_pts, num_samples)

                    # Create Open3D point cloud object
                filtered_data_non_overlapping = remove_intersection(filtered_data)
                autoSam.export_pts_annotated(filtered_data_non_overlapping, instance_category, out_dir)
            
            print("get_annotated_data finished")
            
            if i < n_pass-1:
                outpath = os.path.join(parent_dir, output_dir, "seed_"+str(i)+".json")
                if not os.path.isfile(outpath):
                    sampled_filtered_data = load_pts(instance_category,out_dir_parent, iter=i)
                    sam_seeds_by_category = autoSam.get_seeds(sampled_filtered_data, thresh_proj = 0)
                    # return sam_seeds_by_category
                    print("Finished sam_seeds_by_category: ")
                    
                    utils_common.save_json(sam_seeds_by_category, outpath)
                sam_seeds_by_category = json.load(open(outpath))
                print("get_seed finished")
                imgSeg = ImageSegmentation.ImageSegmentation(predictor, instance_category,sam_seeds_by_category, H, W)
                img_dir = os.path.join(parent_dir, "images")
                out_dir = os.path.join(parent_dir, output_dir, "mask_single_"+str(i))
                if not os.path.exists(out_dir):
                    print("segment_all_images started")
                    imgSeg.segment_all_images(img_dir, out_dir)
                    
                else:
                    print("skipping segment_all_images")
                seeds = sam_seeds_by_category
                for obj in instance_category:
                    new_anchors = [key for key, value in seeds.items() if value.get(obj) and isinstance(value.get(obj), list) and len(value.get(obj)) > 0]
                    anchors = set(autoSam.anchors).union(new_anchors)
                autoSam.anchors = anchors
            # Record end time
            end_time = datetime.now()

            # Calculate elapsed time
            elapsed_time = end_time - start_time

            # Print elapsed time
            print("iteration: ", i)
            print("Elapsed time:", elapsed_time)
    def generate_final_segmentation():
        print("generate_final_segmentation started")
        # autoSam.anchors = anchors_orig
        pts_annotated_by_category = load_pts(instance_category,out_dir_parent, iter=n_pass-1)
        
        filtered_data = {}
        for category in pts_annotated_by_category:
            data = np.array(list(map(list, pts_annotated_by_category[category])))
            filtered_data[category] = remove_outliers(data, z_thresh=2.5)

        
        print("get seed started")
        
        sam_seeds_by_category = autoSam.get_seeds(filtered_data, thresh_proj=-1)
        # # return sam_seeds_by_category
        print("Finished sam_seeds_by_category: ")
        outpath = os.path.join(parent_dir, output_dir, "seed_"+str(n_pass-1)+".json")
        utils_common.save_json(sam_seeds_by_category, outpath)
        # sam_seeds_by_category = json.load(open(outpath))
        print("get_seed finished")
        imgSeg = ImageSegmentation.ImageSegmentation(predictor, instance_category,sam_seeds_by_category, H, W)
        img_dir = os.path.join(parent_dir, "images")
        out_dir = os.path.join(parent_dir, output_dir, "mask_single_"+str(n_pass-1))
        print("segment_all_images started")
        imgSeg.segment_all_images(img_dir, out_dir)

        print("start create_a_modal_mask")
        filtered_data = pts_annotated_by_category
        outdir = os.path.join(parent_dir, output_dir, "amodal_front_"+str(n_pass-1))
        autoSam.create_amodal_mask(filtered_data, outdir)
        print("finished create_a_modal_mask")

    do_first_pass(n_pass)
    generate_final_segmentation()

#------------------------------------------------------#
#                  SAM
#------------------------------------------------------#

# Load Configuration
with open("/home/gmh72/3DReconstruction/ObjectCarverDev/confs/mask.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize SAM Predictor
print("======================started======================")
sam = sam_model_registry["vit_h"](checkpoint=config["sam_checkpoint"])
sam.to(device=config["device"])
predictor = SamPredictor(sam)

scene_name = config["scene_name"]
parent_dir = os.path.join(config["data_dir"], scene_name)
anchors = config["anchors"][scene_name]
start_segmentation(
    parent_dir=parent_dir,
    predictor=predictor,
    anchors=anchors,
    output_dir=config["output_dir"],
    thresh_ratio=3.5,
    n_pass=config["n_pass"],
    surface_type=config["surface_type"],
    sdf_ckpt=config['sdf_ckpt'],
    sdf_conf=config['sdf_conf']
)


