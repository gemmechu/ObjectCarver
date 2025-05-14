import sys
sys.path.append('/home/gmh72/3DReconstruction/SceneCut/training/full_scene/')
import torch
import os
import numpy as np
import trimesh
# import mcubes
from pyhocon import ConfigFactory
from Neus.models.fields import SDFNetwork
from typing import Literal
import matplotlib.pyplot as plt
import json
import cv2
import glob 

def sphere_trace(sdf, origin, dir, max_steps=100, eps=1e-3):
    num = dir.shape[0]
    p = torch.tensor(origin).repeat(num, 1)
    mask = torch.zeros(num, dtype=torch.bool)
    for i in range(max_steps):
        dist = sdf(p)[:, :1]
        p += dist * dir
        is_itx = (torch.abs(dist) < eps)
        in_range = torch.all(torch.abs(p) < 1, dim=1, keepdim=True)
        mask = is_itx * in_range
        if mask.all():
            break
    return p.detach(), mask
    
def is_visible(p1,C2W, sdf_network, eps=1e-2):
    origin = C2W[:,3]
    ray_direction = p1 - origin[:3]
    ray_direction /= np.linalg.norm(ray_direction, ord=2)
    # ray_direction = ray_direction.unsqueeze(0)
    ray_direction = torch.tensor(ray_direction)
    p2 = sphere_trace(sdf_network, origin, ray_direction)
    p1 = torch.tensor(p1)
    distance =torch.norm(p1 - p2)
    if distance < eps:
        return True
    return False

# def is_visible(sdf_values):
#     minimum = sdf_values[0]
#     for value in sdf_values:
#         if minimum > value:
#             return False
#     return True

def get_sdf_network(ckpt_path, conf_path, sdf_name):
    device = "cpu"
    f = open(conf_path)
    conf_text = f.read()
    f.close()

    # conf = ConfigFactory.parse_string(conf_text)
    # sdf_network = SDFNetwork(**conf['model.sdf_network']).to(device)
    sdf_network = SDFNetwork(d_in=3,
                 d_out=257,
                 d_hidden=256,
                 n_layers=8,
                 skip_in=(4,),
                 multires=6,
                 bias=0.5,
                 scale=1,).to(device)
    
    checkpoint = torch.load((ckpt_path), map_location=device)

    sdf_network.load_state_dict(checkpoint[sdf_name])
    
    return sdf_network
def get_surf_pcl_rejection(
                    net, 
                    npoints, 
                    dim, 
                    batch_size=100000, 
                    thr=0.05, 
                    bound=0.95, 
                    return_rej_x=False,
                    eps=1e-15):
               
    out = []
    out = torch.zeros(npoints, dim).cuda()
    ys = torch.zeros(npoints, 1).cuda()
    cnt = 0
    max_duration = 30
    with torch.no_grad():
        while cnt < npoints:
            x = (torch.rand(batch_size, dim).cuda().float() * (2*bound) - bound).cuda()
            y = net(x)[:,:1]
            m = (torch.abs(y) < thr).view(batch_size)
            m_cnt = m.sum().detach().cpu().item()
            if m_cnt < 1:
                continue
            x_eq = x[m].view(m_cnt, dim)

            sidx = cnt
            eidx = min(npoints, cnt + m_cnt)
            out[sidx:eidx] = x_eq[:npoints-cnt]
            ys[sidx:eidx] = y[m].view(m_cnt, 1)[:npoints-cnt]
            # out.append(x_eq)
            cnt += m_cnt
            
    rej_x = out

    return rej_x,ys

def export_to_ply_trimesh(points, filename):
    # Convert the points tensor to a NumPy array
    # points_np = points.cpu().detach()
    points_np = np.array(points)
    
    # Create a Trimesh object from the points
    mesh = trimesh.points.PointCloud(points_np)

    # Export the mesh to a PLY file
    mesh.export(filename)

def get_sdf_value( net, 
                    points, 
                    dim, 
                    batch_size=100000,
                    thr=0.05, 
                    bound=0.95, 
                    ):
               
    out = []
    npoints = len(points)
    out = torch.zeros(npoints, dim).cuda()
    ys = torch.zeros(npoints, 1).cuda()
    cnt = 0
    
    with torch.no_grad():
        while cnt < npoints:
            x = (torch.rand(batch_size, dim).cuda().float() * (2*bound) - bound).cuda()
            y = net(x)[:,:1]
            m = (torch.abs(y) < thr).view(batch_size)
            m_cnt = m.sum().detach().cpu().item()
            if m_cnt < 1:
                continue
            x_eq = x[m].view(m_cnt, dim)

            sidx = cnt
            eidx = min(npoints, cnt + m_cnt)
            out[sidx:eidx] = x_eq[:npoints-cnt]
            ys[sidx:eidx] = y[m].view(m_cnt, 1)[:npoints-cnt]
            # out.append(x_eq)
            cnt += m_cnt
            
    rej_x = out

    return rej_x,ys


