import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from glob import glob
from icecream import ic
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import json

# This function is borrowed from IDR: https://github.com/lioryariv/idr
def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def convert_image_to_binary(image, target_value):
    
    target_value = np.array([target_value[-1], target_value[1],target_value[0]])  #convert RGB to BGR
    
    target_value_broadcasted = target_value.reshape(1, 1, 3)
    
    binary_image = np.where(np.all(image == target_value_broadcasted, axis=2, keepdims=True), 255, 0).astype(np.uint8)
    
    return binary_image

class Dataset:
    def __init__(self, conf):
        super(Dataset, self).__init__()
        print('Load data: Begin')
        self.device = torch.device('cuda')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.mask_path = "mask"
        self.amodal_path = "amodal"
        
        self.camera_outside_sphere = conf.get_bool('camera_outside_sphere', default=True)
        self.scale_mat_scale = conf.get_float('scale_mat_scale', default=1.1)
        
        self.instance_segmentation = json.load(open(os.path.join(self.data_dir, "instance_segmentation.json")))
        self.objects = self.instance_segmentation.keys()
        camera_dict = json.load(open(os.path.join(self.data_dir, "cam_dict.json")))
        for x in list(camera_dict.keys()):
            
            camera_dict[x]["K"] = np.array(camera_dict[x]["K"]).reshape((4, 4))
            camera_dict[x]["C2W"] = np.array(camera_dict[x]["C2W"]).reshape((4, 4))

        self.camera_dict = camera_dict

        self.images_lis = sorted(glob(os.path.join(self.data_dir, 'images/*.png')) + glob(os.path.join(self.data_dir, 'images/*.jpg')))
        self.n_images = len(self.images_lis)
        self.images_np = np.stack([cv.imread(im_name) for im_name in self.images_lis]) / 256.0
        self.masks = {}
        self.masks_lis = sorted(glob(os.path.join(self.data_dir, self.mask_path +'/*.png'))+ glob(os.path.join(self.data_dir, self.mask_path +'/*.jpg')))
        self.mask_all = {}
        for idx in range(len(self.masks_lis)):
            binary_mask_all = []
            self.masks[idx] = {}
            for obj_key in self.objects:
                binary_mask_all.append(convert_image_to_binary(cv.imread(self.masks_lis[idx]), target_value = self.instance_segmentation[obj_key]))
            binary_mask_all = np.array(binary_mask_all).sum(axis=0)
            self.mask_all[idx] = binary_mask_all
            
            for obj_key in self.objects:
                self.masks[idx][obj_key] = {}
                binary_mask = convert_image_to_binary(cv.imread(self.masks_lis[idx]), target_value = self.instance_segmentation[obj_key])
                mask_dont_care = binary_mask_all - binary_mask
                
                self.masks[idx][obj_key]["mask"] = binary_mask/255
                self.masks[idx][obj_key]["mask_dont_care"] = mask_dont_care/255
                #------------ A Modal Dcare----------------#
                img_name = self.masks_lis[idx].split("/")[-1]
                mask_amodal = cv.imread(os.path.join(self.data_dir, self.amodal_path ,obj_key, img_name))[...,0]/255
                self.masks[idx][obj_key]["mask_dont_care"] = (np.logical_and((mask_dont_care/255)[...,0], mask_amodal))[:,:, np.newaxis]
                # ------------ A Modal Dcare----------------#
               

        self.scale_mats_np = []

        # scale_mat: used for coordinate normalization, we assume the scene to render is inside a unit sphere at origin.
        self.scale_mats_np = [np.eye(4).astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []

        for x in self.images_lis:
            x = os.path.basename(x)
            K = self.camera_dict[x]["K"].astype(np.float32)
            C2W = self.camera_dict[x]["C2W"].astype(np.float32)
            self.intrinsics_all.append(torch.from_numpy(K))
            self.pose_all.append(torch.from_numpy(C2W))
            

        self.images = torch.from_numpy(self.images_np.astype(np.float32)).cpu()  # [n_images, H, W, 3]
        for idx in range(len(self.masks_lis)):
            for obj_key in self.objects:
                self.masks[idx][obj_key]["mask"] = torch.from_numpy(self.masks[idx][obj_key]["mask"].astype(np.float32)).cpu()  
                self.masks[idx][obj_key]["mask_dont_care"] = torch.from_numpy(self.masks[idx][obj_key]["mask_dont_care"].astype(np.float32)).cpu()   
       
        self.intrinsics_all = torch.stack(self.intrinsics_all).to(self.device)   # [n_images, 4, 4]
        self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)  # [n_images, 4, 4]
        self.focal = self.intrinsics_all[0][0, 0]
        self.pose_all = torch.stack(self.pose_all).to(self.device)  # [n_images, 4, 4]
        self.H, self.W = self.images.shape[1], self.images.shape[2]
        self.image_pixels = self.H * self.W

        object_bbox_min = np.array([-1.01, -1.01, -1.01, 1.0])
        object_bbox_max = np.array([ 1.01,  1.01,  1.01, 1.0])
        
        object_scale_mat = np.eye(4).astype(np.float32)
        object_bbox_min = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_min[:, None]
        object_bbox_max = np.linalg.inv(self.scale_mats_np[0]) @ object_scale_mat @ object_bbox_max[:, None]
        self.object_bbox_min = object_bbox_min[:3, 0]
        self.object_bbox_max = object_bbox_max[:3, 0]

        print('Load data: End')

    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1) # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = self.pose_all[img_idx, None, None, :3, 3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def gen_random_rays_at(self, img_idx, batch_size):
        """
        Generate random rays at world space from one camera.
        """
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        color = self.images[img_idx][(pixels_y, pixels_x)]    # batch_size, 3
        mask_list = []
        
        for obj_key in self.objects:
            mask_list_item= np.array([])
           
            img_idx_int = int(img_idx)
            mask_list_item = np.stack((self.masks[img_idx_int][obj_key]["mask"][(pixels_y, pixels_x)][:, :1], self.masks[img_idx_int][obj_key]["mask_dont_care"][(pixels_y, pixels_x)][:,:1]))
            mask_list.append([mask_list_item])
        mask_list = np.vstack(mask_list)
       
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1).float()  # batch_size, 3
        p = torch.matmul(self.intrinsics_all_inv[img_idx, None, :3, :3], p[:, :, None]).squeeze() # batch_size, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)    # batch_size, 3
        rays_v = torch.matmul(self.pose_all[img_idx, None, :3, :3], rays_v[:, :, None]).squeeze()  # batch_size, 3
        rays_o = self.pose_all[img_idx, None, :3, 3].expand(rays_v.shape) # batch_size, 3
       
        mask_list = torch.from_numpy(mask_list)
        return torch.cat([rays_o.cpu(), rays_v.cpu(), color], dim=-1).cuda(), mask_list.cuda()   # batch_size, 10
        
       

    def gen_rays_between(self, idx_0, idx_1, ratio, resolution_level=1):
        """
        Interpolate pose between two cameras.
        """
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)
        ty = torch.linspace(0, self.H - 1, self.H // l)
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        p = torch.stack([pixels_x, pixels_y, torch.ones_like(pixels_y)], dim=-1)  # W, H, 3
        p = torch.matmul(self.intrinsics_all_inv[0, None, None, :3, :3], p[:, :, :, None]).squeeze()  # W, H, 3
        rays_v = p / torch.linalg.norm(p, ord=2, dim=-1, keepdim=True)  # W, H, 3
        trans = self.pose_all[idx_0, :3, 3] * (1.0 - ratio) + self.pose_all[idx_1, :3, 3] * ratio
        pose_0 = self.pose_all[idx_0].detach().cpu().numpy()
        pose_1 = self.pose_all[idx_1].detach().cpu().numpy()
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        rot_0 = pose_0[:3, :3]
        rot_1 = pose_1[:3, :3]
        rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
        key_times = [0, 1]
        slerp = Slerp(key_times, rots)
        rot = slerp(ratio)
        pose = np.diag([1.0, 1.0, 1.0, 1.0])
        pose = pose.astype(np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        rot = torch.from_numpy(pose[:3, :3]).cuda()
        trans = torch.from_numpy(pose[:3, 3]).cuda()
        rays_v = torch.matmul(rot[None, None, :3, :3], rays_v[:, :, :, None]).squeeze()  # W, H, 3
        rays_o = trans[None, None, :3].expand(rays_v.shape)  # W, H, 3
        return rays_o.transpose(0, 1), rays_v.transpose(0, 1)

    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far

    def image_at(self, idx, resolution_level):
        img = cv.imread(self.images_lis[idx])
        return (cv.resize(img, (self.W // resolution_level, self.H // resolution_level))).clip(0, 255)