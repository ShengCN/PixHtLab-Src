import cv2
import os

import sys
import inspect
import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
from torchvision import utils
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import time
import numpy as np
import math
from abc import ABC
import copy


class camera(ABC):
    def __init__(self, hfov, h, w, height=100.0):
        self.fov = hfov
        self.h   = h
        self.w   = w

        self.ori_height = height
        self.height     = copy.deepcopy(self.ori_height)
        self.O          = np.array([0.0, self.height, 0.0]) # ray origianl


    ######################################################################################
    """ Abstraction
    """
    def align_horizon(self, cur_horizon):
        raise NotImplementedError('Not implemented yet')

    def C(self):
        raise NotImplementedError('Not implemented yet')

    def right(self):
        raise NotImplementedError('Not implemented yet')

    def up(self):
        raise NotImplementedError('Not implemented yet')

    ######################################################################################

    def deg2rad(self, d):
        return d / 180.0 * 3.1415925


    def rad2deg(self, d):
        return d / 3.1415925 * 180.0


    def get_ray(self, xy):
        """ Assume the center is on the top-left corner
        """
        u, v = xy
        mat  = self.get_ABC_mat()
        r    = np.dot(mat, np.array([u, v, 1.0]).T)

        # r = r/np.sqrt(r @ r)
        return r


    def project(self, xyz):
        relative = xyz - self.O

        mat   = self.get_ABC_mat()
        pp    = np.dot(np.linalg.inv(mat), relative)
        pixel = np.array([pp[0]/pp[2], pp[1]/pp[2]])

        return pixel


    def xyh2w(self, xyh):
        u, v, h = xyh

        foot_xyh    = np.copy(xyh)
        foot_xyh[1] = foot_xyh[1] + foot_xyh[2]
        foot_xyh[2] = 0.0
        fu, fv, fh  = foot_xyh

        a   = self.right()
        b   = -self.up()
        c   = self.C()
        mat = self.get_ABC_mat()

        w = -self.height/(a[1] * fu + b[1] * fv + c[1])
        return w


    def xyh2xyz(self, xyh):
        u, v, h = xyh

        foot_xyh    = np.copy(xyh)
        foot_xyh[1] = foot_xyh[1] + foot_xyh[2]
        foot_xyh[2] = 0.0
        fu, fv, fh  = foot_xyh

        a   = self.right()
        b   = -self.up()
        c   = self.C()
        mat = self.get_ABC_mat()

        w = -self.height/(a[1] * fu + b[1] * fv + c[1])
        # print('w: {} a*u + b * v + c: {}, b/c: {}/{}, fv: {}'.format(w, a[1] * fu + b[1] * fv + c[1], b, c, fv))
        xyz = self.O + np.dot(mat, np.array([u, v, 1.0]).T) * w

        # print('w: {}, -{}/{}'.format(w, self.height, a[1] * fu + b[1] * fv + c[1]))

        return xyz


    def xyz2xyh(self, xyz):
        foot_xyz    = np.copy(xyz)
        foot_xyz[1] = 0.0

        foot_xy = self.project(foot_xyz)
        xy      = self.project(xyz)

        ret     = np.copy(xyz)
        ret[:2] = xy
        ret[2]  = foot_xy[1] - xy[1]

        return ret


    def get_ABC_mat(self):
        a = self.right()
        b = -self.up()
        c = self.C()

        mat = np.concatenate([a[:, None], b[:,None], c[:, None]], axis=1)
        return mat


class pitch_camera(camera):
    """ Picth alignment camera
    """
    def __init__(self, hfov, h, w, height=100.0):
        """
           alignment algorithm:
                  1. pitch alignment
                  2. axis alignment
        """
        super().__init__(hfov, h, w, height)

        self.ori_view   = np.array([0.0, 0.0, -1.0])
        self.cur_view   = np.copy(self.ori_view)


    def align_horizon(self, cur_horizon):
        """ Given horizon, compute the camera pitch
        """
        ref_horizon  = self.h / 2
        rel_distance = -(ref_horizon - cur_horizon)

        focal = self.focal()
        pitch = math.atan2(rel_distance, focal)

        # construct a rotation matrix
        c, s = np.cos(pitch), np.sin(pitch)
        rot = np.array([[0, 0, 0], [0, c, -s], [0, s, c]])

        # compute the new view vector
        img_plane_view = self.ori_view * focal
        img_plane_view = rot @ img_plane_view.T

        self.cur_view = img_plane_view / math.sqrt(np.dot(img_plane_view, img_plane_view))

    def C(self):
        return self.view() * self.focal() - 0.5 * self.w * self.right() + 0.5 * self.h * self.up()


    def right(self):
        return np.array([1.0, 0.0, 0.0])


    def up(self):
        return np.cross(self.right(), self.view())


    def focal(self):
        focal = self.w * 0.5 / math.tan(self.deg2rad(self.fov * 0.5))
        return focal


    def view(self):
        return self.cur_view



class axis_camera(camera):
    """ Axis alignment camera
    """
    def __init__(self, hfov, h, w, height=100.0):
        super().__init__(hfov, h, w, height)

        focal          = self.w * 0.5 / math.tan(self.deg2rad(self.fov * 0.5))
        self.up_vec    = np.array([0.0, 1.0, 0.0])
        self.right_vec = np.array([1.0, 0.0, 0.0])

        self.ori_c = np.array([-0.5 * self.w, 0.5 * self.h, -focal])
        self.c_vec = np.copy(self.ori_c)


    def align_horizon(self, cur_horizon):
        """ Given horizon, we move the axis to update the horizon
            i.e. we need to change C
        """
        ref_horizon   = self.h // 2
        delta_horizon = cur_horizon - ref_horizon
        self.c_vec    = self.ori_c + delta_horizon * self.up()
        # self.height   = self.ori_height + delta_horizon
        # self.O        = np.array([0.0, self.height, 0.0])



    def C(self):
        return self.c_vec


    def right(self):
        return self.right_vec


    def up(self):
        return self.up_vec


def test(ppc):
    xyh = np.array([500, 500, 100.0])

    proj_xyz = ppc.xyh2xyz(xyh)
    proj_xyh = ppc.xyz2xyh(proj_xyz)

    print('xyh: {}, proj xyz: {}, proj xyh: {}'.format(xyh, proj_xyz, proj_xyh))

    # import pdb; pdb.set_trace()
    new_horizon_list = [0, 100, 250, 400, 500]
    new_horizon_list = [100, 250, 400, 500]

    # import pdb; pdb.set_trace()
    for cur_horizon in new_horizon_list:
        ppc.align_horizon(cur_horizon)
        test_xyh = np.array([500, cur_horizon, 0])
        test_xyz = ppc.xyh2xyz(test_xyh)

        # print('{} -> {} -> {}'.format(test_xyh, test_xyz, ppc.xyz2xyh(test_xyz)))
        print('{} \t -> {} \t -> {}'.format(test_xyh, test_xyz, ppc.xyz2xyh(test_xyz)))


if __name__ == '__main__':
    p_camera = pitch_camera(90.0, 500, 500)
    a_camera = axis_camera(90.0, 500, 500)

    test(p_camera)
    test(a_camera)


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

    return xyz


def normalize_vec3(vec3):
    """ vec3: ... x 3
    """
    eps = 1e-8
    return vec3/(np.linalg.norm(vec3, axis=-1, keepdims=True)+eps)
