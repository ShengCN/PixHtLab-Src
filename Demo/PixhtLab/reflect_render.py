import cv2
import os

import sys
import inspect

import h5py
import matplotlib.pyplot as plt
import numpy as np
from camera import pitch_camera, axis_camera
from tqdm import tqdm

import torch
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

    return xyz


def normalize_vec3(vec3):
    """ vec3: ... x 3
    """
    return vec3/np.linalg.norm(vec3, axis=-1, keepdims=True)


def to_tensor(np_img):
    device = torch.device("cuda:0")
    return torch.tensor(np_img.transpose((2,0,1)))[None, ...].to(device).float()


def sphere_cartesian(theta, phi):
    # t, p = deg2rad(theta), deg2rad(phi)
    t, p = theta, phi
    coord = np.array([np.sin(p)* np.cos(t), np.cos(p), np.sin(p) * np.sin(t)])
    return coord.T


def uniform_over_sphere(b, n):
    theta = 2 * np.pi * np.random.uniform(0.0, 1.0, (b, n))
    phi = np.arccos(2*np.random.uniform(0.0, 1.0, (b, n))-1.0)

    coord = sphere_cartesian(theta, phi)
    return coord.transpose((1,0,2))


def solid_angle_sampling(cur_dir, ang, n):
    h,w = cur_dir.shape[:2]

    sphere_samples = uniform_over_sphere(h * w, n)

    rad = deg2rad(ang)
    dis = 1.0/np.tan(rad)

    sphere_samples = sphere_samples.reshape(n, h, w, 3)

    samples = sphere_samples + (cur_dir/np.linalg.norm(cur_dir, axis=2, keepdims=True) * dis)
    samples = samples/np.linalg.norm(samples, axis=-1, keepdims=True)
    return samples


def glossy_samples(reflected_dirs, glossy_ness, sample_n):
    """ Inputs:
            reflected_dirs: H x W x 3
            glossy_ness: [0, 1]
            sample_n: [1, N]
        Outputs:
            [sample_n, H, W, 3]
    """
    solid_ang = glossy_ness * 90.0
    samples   = solid_angle_sampling(reflected_dirs, solid_ang, sample_n)

    return samples


def Fresnel(wi, ref_idx):
    cosine = np.cos(wi)
    r0 = (1.0 - ref_idx)/(1.0 + ref_idx)
    r0 = r0 ** 2
    return r0 + (1.0-r0) * np.power((1.0-cosine), 5)


def BRDF(wi, wo, n, brdf_type, params=None):
    if brdf_type == 'oren_nayar':
        sigma = params['sigma']
        fr = oren_nayar_reflect(sigma, wi, wo)
    elif brdf_type == 'diffuse':
        fr = np.ones_like(wo) * 1.0  # 50 x H x W
        fr = fr[:, None, ...]
    elif brdf_type == 'empirical':
        pass
    else:
        raise NotImplementedError('{} not implemented yet'.format(brdf_type))


    ref_idx = params['ref_idx']
    # fresnel = (1.0-Fresnel(wi, ref_idx))
    fresnel = Fresnel(wi, ref_idx)
    return fr, fresnel


def ray_intersect(rgb, mask, hmap, ro, rd, dh):
    b, c, h, w = rgb.shape

    start = time.time()
    intersection = hshadow.ray_scene_intersect(rgb, mask, hmap, ro, rd, dh)
    end = time.time()
    # print('Ray-Scene intersect: {}s'.format(end-start))
    return intersection



def render_reflection(fg_rgba, fg_height, bg_rgba, bg_height, params):
    """ Note, samples_n in params should be n * 50
    """
    horizon    = params['horizon']
    ref_idx    = params['ref_idx']
    samples_n  = params['sample_n']
    glossness  = params['glossness']
    dh         = params['dh']
    batch_size = params['batch_size']
    camera_h   = params['camera_h']

    n          = batch_size
    batch_size = max(samples_n // batch_size, 1)

    device     = torch.device("cuda:0")

    fg_rgb, fg_mask = fg_rgba[..., :3], fg_rgba[..., -1:]
    bg_rgb, bg_mask = bg_rgba[..., :3], bg_rgba[..., -1:]

    h, w = fg_rgba.shape[:2]
    cur_camera = axis_camera(80.0, h, w, camera_h)
    cur_camera.align_horizon(horizon)

    xyz = xyh2xyz(bg_height, cur_camera)
    normal = normalize_vec3(compute_normal(xyz)) * bg_mask
    normal[np.isnan(normal)] = 0.0

    ray_mat        = get_ray_mat(cur_camera)   # H x W x 3
    ray_mat_normal = normalize_vec3(ray_mat)
    rr = ray_mat_normal - 2.0 * (ray_mat_normal * normal).sum(axis=2, keepdims=True) * normal
    wi = np.arccos(-(ray_mat_normal * np.array([0.0, 1.0, 0.0])[None, None, ...]).sum(axis=-1))  # for BRDF, 1 x H x W

    h,w = ray_mat_normal.shape[:2]
    result = np.zeros((h, w, 3))
    alpha = np.zeros((h, w, 1))

    cur_rgb     = torch.tensor(fg_rgb.transpose((2,0,1)))[None, ...].repeat(n, 1, 1, 1).float().to(device)
    cur_mask    = torch.tensor(fg_mask.transpose((2,0,1)))[None, ...].repeat(n, 1, 1, 1).float().to(device)
    cur_hmap    = torch.tensor(fg_height.transpose((2,0,1)))[None, ...].repeat(n, 1, 1, 1).float().to(device)
    cur_rechmap = torch.tensor(bg_height.transpose((2,0,1)))[None, ...].repeat(n, 1, 1, 1).float().to(device)

    cur_rgb_np = np.repeat(bg_rgba[..., :3].transpose((2,0,1))[None, ...], n, axis=0)

    ro_np = xyz2xyh(xyz, cur_camera)
    ro    = torch.tensor(ro_np.transpose((2,0,1)))[None, ...].repeat(n, 1, 1, 1).float().to(device)

    for i in tqdm(range(batch_size), desc='Render'):
        rr_samples = glossy_samples(rr, glossness, n)                           # 5 x H x W x 3
        wo = np.arccos((rr_samples * np.array([[0.0, 1.0, 0.0]])).sum(axis=-1)) # n x H x W

        fr, frenel = BRDF(wi, wo, normal, brdf_type='diffuse', params={'ref_idx': ref_idx})                                     # n x H x W

        # # compute reflected xyh
        scale = 1.0
        reflected_xyz = xyz[None, ...] + rr_samples * scale
        reflected_xyh = xyz2xyh(reflected_xyz, cur_camera)

        rd = reflected_xyh - ro_np
        rd = torch.tensor(rd.transpose((0, 3, 1, 2))).float().contiguous().to(device)

        intersect     = ray_intersect(cur_rgb, cur_mask, cur_hmap, ro, rd, dh).detach().cpu().numpy()
        rgb_channel   = intersect[:, :3]       # n x 3 x H x W
        rgb_channel   = rgb_channel * fr
        alpha_channel = intersect[:, -1:]

        missing_pos = alpha_channel[0, 0] == 0
        rgb_channel[:, :, missing_pos] = cur_rgb_np[:, :, missing_pos]
        alpha_channel = intersect[:, -1:] * fr * frenel

        result += rgb_channel.sum(axis=0).transpose((1,2,0))
        alpha  += alpha_channel.sum(axis=0).transpose((1,2,0))

        torch.cuda.empty_cache()

    rgb_channel = result / (n * batch_size)
    alpha_channel = alpha/(n * batch_size)
    return rgb_channel, alpha_channel



def reflection_layer(fg_rgba, fg_height, bg_rgba, bg_height, bg_reflection_layer, params):
    """ Example:
            params = {
                'sample_n': 1,
                'horizon': h//2,
                'ref_idx': 0.8,
                'glossness': 0.01,
                'dh': 5.0,
                'batch_size': 1
            }

            bg_reflection_layer = np.copy(bg_height)
            bg_reflection_layer[bg_reflection_layer>0.0] = 1.0
            bg_reflection_layer = 1.0 - bg_reflection_layer

            # fg_rgba, fg_height = glass_rgba, glass_height
            fg_rgba, fg_height = flower_rgba, flower_height
            reflection_rgb = reflection_layer(fg_rgba, fg_height, bg_rgba, bg_height, bg_reflection_layer, params)
            show(reflection_rgb)
    """
    reflect_alpha = params['reflect_alpha']

    bg_rgba, bg_height = bg_rgba, bg_height
    h, w                             = bg_rgba.shape[:2]
    reflection_rgb, reflection_alpha = render_reflection(fg_rgba, fg_height, bg_rgba, bg_height, params)
    reflection_alpha                 = reflect_alpha * reflection_alpha * bg_reflection_layer
    reflection_rgb = reflection_rgb * reflection_alpha  + bg_rgba[..., :3] * (1.0-reflection_alpha)

    return reflection_rgb



def render_refraction(fg_rgba, fg_height, bg_rgba, bg_height, params):
    etai_over_etat = params['etai_over_etat']
    horizon = params['horizon']
    dh = params['dh']

    h, w = fg_rgba.shape[:2]
    cur_camera = axis_camera(80.0, h, w)
    cur_camera.align_horizon(horizon)

    fg_mask = fg_rgba[..., -1:]
    fg_mask[fg_mask<0.999] = 0.0
    fg_mask[fg_mask>0.999] = 1.0

    # ray intersection
    xyz    = xyh2xyz(fg_height, cur_camera)
    normal = compute_normal(xyz)
    normal = normalize_vec3(normal)

    # ray refraction
    rays = get_ray_mat(cur_camera)
    rays = normalize_vec3(rays)

    """
    auto cos_theta = fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp =  etai_over_etat * (uv + cos_theta*n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
    """
    cos_theta        = (-rays * normal).sum(axis=-1, keepdims=True)
    cos_theta[cos_theta>=1.0] = 1.0

    ray_out_perp     = etai_over_etat * (rays + cos_theta * normal)
    ray_out_parallel = -np.sqrt(np.abs(1.0-(ray_out_perp*ray_out_perp).sum(axis=-1, keepdims=True))) * normal
    ray_out = ray_out_perp + ray_out_parallel
    ray_out[np.isnan(ray_out)] = 0.0
    ray_out = normalize_vec3(ray_out)
    ray_out = ray_out * fg_mask

    # everything needs to be transformed into xyh domain
    ray_out_xyz = xyz + ray_out
    ray_out_xyh = xyz2xyh(ray_out_xyz, cur_camera)
    xyh         = xyz2xyh(xyz, cur_camera)
    ray_out_xyh = ray_out_xyh - xyh
    ray_out_xyh[np.isnan(ray_out_xyh)] = 0.0

    device    = torch.device('cuda:0')
    cur_rgb   = torch.tensor(bg_rgba[..., :3].transpose((2,0,1)))[None, ...].float().contiguous().to(device)
    cur_mask  = torch.tensor(bg_rgba[..., -1:].transpose((2,0,1)))[None, ...].float().contiguous().to(device)
    cur_hmap  = torch.tensor(bg_height.transpose((2,0,1)))[None, ...].float().contiguous().to(device)

    ro = torch.tensor(xyh.transpose((2, 0, 1))).float()[None, ...].contiguous().to(device)
    rd = torch.tensor(ray_out_xyh.transpose((2, 0, 1))).float()[None, ...].contiguous().to(device)

    intersect = ray_intersect(cur_rgb, cur_mask, cur_hmap, ro, rd, dh)
    refracted = intersect[0, :3].detach().cpu().numpy().transpose((1,2,0))
    return refracted


def refraction_composite(fg_rgba, fg_height, bg_rgba, bg_height, refract_layer, params):
    """ Example:
            params = {
                'etai_over_etat': 1.0/1.5,
                'horizon': 395.0,
                'dh': 30.0
            }
            refract_layer = fg_rgba[..., -1:]
            final_comp = refraction_composite(fg_rgba, fg_height, bg_rgba, bg_height, refract_layer, params)

            show(final_comp)
    """
    fg_mask = fg_rgba[..., -1:]
    fg_rgb  = fg_rgba[..., :3]
    bg_rgb  = bg_rgba[..., :3]
    bg_mask = bg_rgba[..., -1:]

    refracted  = render_refraction(fg_rgba, fg_height, bg_rgba, bg_height, params)
    fg_mask    = fg_mask * refract_layer
    final_comp = fg_mask * refracted + (1.0-fg_mask) * bg_rgb

    return final_comp
