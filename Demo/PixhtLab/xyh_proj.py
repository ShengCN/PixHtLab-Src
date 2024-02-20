import cv2
import os

import sys
import inspect

import h5py
import matplotlib.pyplot as plt
import numpy as np
from camera import pitch_camera, axis_camera
from tqdm import tqdm

from torchvision import utils
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from meshplot import plot, subplot, interact
import time
import hshadow

def to_numpy(tensor):
    return tensor[0].detach().cpu().numpy().transpose(1,2,0)


def test_intersect(rgb, mask, hmap, rechmap, light):
    b, c, h, w = rgb.shape

    start = time.time()
    intersection = hshadow.ray_intersect(rgb, mask, hmap, rechmap, light)
    end = time.time()
    # print('Reflection rendering: {}s'.format(end-start))
    return intersection
    intersection = to_numpy(intersection)
    return intersection



def compute_normal(xyz):
    zx = cv2.Sobel(xyz, cv2.CV_64F, 1, 0, ksize=3)
    zy = cv2.Sobel(xyz, cv2.CV_64F, 0, 1, ksize=3)
    norm = np.cross(zy, zx)
    return norm


def deg2rad(deg):
    return deg/180.0 * 3.14159265


def direction(theta, phi):
    t, p = deg2rad(theta), deg2rad(phi)
    return np.array([np.sin(p)* np.cos(t), np.cos(p), np.sin(p) * np.sin(t)])


def shading(normal, light_dir):
    relighted = np.clip(np.dot(normal, light_dir), 0.0, 1.0)
    return relighted


def get_camera_mat(cur_camera):
    """ return 3 x 3 camera matrix
    """
    cam_mat = cur_camera.get_ABC_mat()
    return cam_mat


def get_ray_mat(cur_camera):
    """ return H x W x 3 vector field
    """
    h, w = cur_camera.h, cur_camera.w
    u = np.arange(0, w, 1)
    v = np.arange(0, h, 1)
    uu, vv = np.meshgrid(u, v)
    uniform_coord = np.concatenate([uu[..., None], vv[..., None], np.ones_like(uu)[..., None]], axis=2)

    cam_mat = get_camera_mat(cur_camera)
    ray_mat = np.einsum('ct,hwt->hwc', cam_mat, uniform_coord)

    return ray_mat


def project(xyz, cur_camera):
    """ xyz:    B x 3
        return: B x 2
    """
    O = cur_camera.O
    relative = xyz - O

    cam_mat = get_camera_mat(cur_camera)  # 3 x 3
    inv_cam_mat = np.linalg.inv(cam_mat)  # 3 x 3

    # M x UV * w = P - O
    B = len(xyz)
    pp    = np.einsum('ct,Bt->Bc', inv_cam_mat, relative)    # B x 3
    pixel = pp/pp[..., -1:]                                    # B x 3

    return pixel[..., :2]


def xyz2xyh(xyz, cur_camera):
    """ B x 3 -> B x 3
    """
    ori_shape = xyz.shape
    
    xyz = xyz.reshape(-1, 3)    
    foot_xyz = np.copy(xyz)
    
    foot_xyz[..., 1] = 0.0                      # 0.0 is ground
    foot_xy = project(foot_xyz, cur_camera)     # B x 3
    xy      = project(xyz, cur_camera)          # B x 3

    ret     = np.copy(xyz)                      # B x 3
    ret[..., :2] = xy
    ret[..., 2]  = foot_xy[..., 1] - xy[..., 1] # B x 3
    
    ret = ret.reshape(*ori_shape)
    xyz = xyz.reshape(*ori_shape)
    return ret


def xyh2xyz(xyh, cur_camera):
    """ xyh: H x W x 1, pixel height channel
    """
    h, w = xyh.shape[:2]

    u = np.arange(0, w, 1)
    v = np.arange(0, h, 1)
    uu, vv = np.meshgrid(u, v)   # H x W

    h = xyh
    coord = np.concatenate([uu[..., None], vv[..., None], h], axis=-1)

    coord[..., 1] = coord[..., 1] + coord[..., 2] # foot Y coord
    coord[..., 2] = 0.0                           # height 0

    fu, fv, fh = coord[..., 0], coord[..., 1], coord[..., 2] # H x W

    a = cur_camera.right()   # 3
    b = -cur_camera.up()     # 3
    c = cur_camera.C()       # 3

    ww = -cur_camera.height/(a[1] * fu + b[1] * fv + c[1]) # H x W

    ww[np.isinf(ww)] = 0.0
    ww[np.isnan(ww)] = 0.0

    cam_origin = cur_camera.O
    cam_mat    = get_camera_mat(cur_camera) # 3 x 3

    uniform_coord = np.concatenate([uu[..., None], vv[..., None], np.ones_like(uu)[..., None]], axis=-1) # H x W x 3
    xyz = cam_origin + np.einsum('ct,hwt->hwc', cam_mat, uniform_coord) * ww[..., None]                # H x W x 3

    return xyz, ww


def normalize_vec3(vec3):
    """ vec3: ... x 3
    """
    return vec3/(np.linalg.norm(vec3, axis=-1, keepdims=True)+1e-8)


def to_tensor(np_img):
    device = torch.device("cuda:0")
    return torch.tensor(np_img.transpose((2,0,1)))[None, ...].to(device).float()



