import time
import torch
import hshadow
import plane_visualize
import numpy as np
from torchvision import transforms
from scipy.ndimage import uniform_filter
from ShadowStyle.inference import inference_shadow
import cv2
import matplotlib.pyplot as plt
from utils import *
from GSSN.inference_shadow import SSN_Infernece

device     = torch.device("cuda:0")
to_tensor  = transforms.ToTensor()
model      = inference_shadow.init_models('/home/ysheng/Documents/Research/GSSN/HardShadow/qtGUI/weights/human_baseline_all_21-July-04-52-AM.pt')
# GSSN_model = SSN_Infernece('GSSN/weights/0000000700.pt')
GSSN_model = SSN_Infernece('/home/ysheng/Documents/Research/GSSN/HardShadow/qtGUI/GSSN/weights/only_shadow/0000000200.pt')

def crop_mask(mask):
        hnon, wnon = np.nonzero(mask)
        aabb = (hnon.min(), hnon.max(), wnon.min(), wnon.max())
        return aabb

def norm_output(np_img):
        return np.clip(cv2.normalize(np_img, None, 0.0, 1.0, cv2.NORM_MINMAX),0.0,1.0)

def padding(mask, shadow, mask_aabb, shadow_aabb, final_shape=(512, 512)):
        mh, mhh, mw, mww = mask_aabb
        sh, shh, sw, sww = shadow_aabb
        cropped_mask, cropped_shadow = mask[mh:mhh, mw:mww], shadow[sh:shh, sw:sww]
        global_h, global_w = mask.shape[:2]
        h, w, c, sc = *cropped_mask.shape, shadow.shape[2]
        fract = 0.4
        if h > w:
                newh = int(final_shape[0]*fract)
                neww = int(newh/h*w)
        else:
                neww = int(final_shape[1]*fract)
                newh = int(neww/w*h)

        small_mask = cv2.resize(cropped_mask, (neww, newh), interpolation=cv2.INTER_AREA)
        if len(small_mask.shape) == 2:
                small_mask = small_mask[...,np.newaxis]

        mask_ret, shadow_ret = np.zeros((final_shape[0], final_shape[1], c)),np.ones((final_shape[0], final_shape[1], sc))
        paddingh, paddingw = 10, (final_shape[0]-neww)//2
        mask_lpos = (paddingh, paddingw)
        mask_ret = overlap_replace(mask_ret, small_mask, mask_lpos)

        # padding shadow
        hscale, wscale = newh/h, neww/w
        newsh, newsw = int((shh-sh) * hscale), int((sww-sw) * wscale)
        small_shadow = cv2.resize(cropped_shadow, (newsw, newsh), interpolation=cv2.INTER_AREA)

        if len(small_shadow.shape) == 2:
                small_shadow = small_shadow[...,np.newaxis]


        loffseth, loffsetw = int((sh-mh)*hscale), int((sw-mw)*wscale)
        shadow_lpos = (paddingh + loffseth, paddingw + loffsetw)
        shadow_ret = overlap_replace(shadow_ret, small_shadow, shadow_lpos)

        # return mask_ret, shadow_ret[...,0:1], [mask_aabb, mask_lpos, hscale, wscale, final_shape, mask.shape[0], mask.shape[1]]
        return mask_ret, shadow_ret, [mask_aabb, mask_lpos, hscale, wscale, final_shape, mask.shape[0], mask.shape[1]]


def transform_input(mask, hardshadow):
        """ Note, trans_info marks the AABBs, and scaling factors
        """
        mask_aabb, shadow_aabb = crop_mask(mask[...,0]), crop_mask(hardshadow[...,0])
        # import pdb; pdb.set_trace()
        cmask, cshadow, trans_info = padding(mask, hardshadow, mask_aabb, shadow_aabb)
        return cmask.transpose(2,0,1)[np.newaxis,...], 1.0 - cshadow.transpose(2,0,1)[np.newaxis, ...], trans_info


def transform_output(softshadow, trans_info):
        mask_aabb, mask_lpos, hscale, wscale, final_shape, h, w = trans_info
        # import pdb; pdb.set_trace()
        ret, gsh, gsw = np.zeros((h,w,1)), int(final_shape[0]/hscale), int(final_shape[1]/wscale)
        global_shadow = cv2.resize(softshadow[0,0], (gsw, gsh))

        # global start = global_mask_aabb - (local_mask_start)/scaling
        mh, mw, mask_lh, mask_lw = mask_aabb[0], mask_aabb[2], mask_lpos[0], mask_lpos[1]
        starth, startw = int(mh - mask_lh / hscale), int(mw - mask_lw / wscale)
        ret = norm_output(overlap_replace(ret, global_shadow[...,np.newaxis], (starth, startw)))
        if len(ret.shape) == 2:
                ret = ret[..., np.newaxis]

        return 1.0-ret.repeat(3,axis=2)

def style_hardshadow(mask, hardshadow, softness):
        mask_net, hardshadow_net, trans_info = transform_input(mask, hardshadow)
        netsoftshadow = inference_shadow.net_render_np(model, mask_net, hardshadow_net, softness, 0.0)
        softshadow = transform_output(netsoftshadow, trans_info)

        return softshadow, (norm_output(mask_net[0,0]), norm_output(hardshadow_net[0,0]), norm_output(netsoftshadow[0,0]))

def gssn_shadow(mask, pixel_height, shadow_channels, softness):
        # mask_net, hardshadow_net, trans_info = transform_input(mask, shadow_channels)

        mask_aabb, shadow_aabb                 = crop_mask(mask[...,0]), crop_mask(shadow_channels[...,0])
        ph_channel, hardshadow_net, trans_info = padding(pixel_height, shadow_channels, mask_aabb, shadow_aabb)

        ph_channel     = ph_channel/512.0
        hardshadow_net = 1.0-hardshadow_net
        input_np       = np.concatenate([ph_channel, hardshadow_net], axis=2)

        # import pdb; pdb.set_trace()

        netsoftshadow = np.clip(GSSN_model.render_ss(input_np, softness), 0.0, 1.0)
        netsoftshadow = netsoftshadow.transpose((2,0,1))[None, ...]
        softshadow    = transform_output(netsoftshadow, trans_info)

        return softshadow


def proj_ground(p, light_pos):
        tmpp = p.copy()

        t = (0-tmpp[2])/(light_pos[:, 2:3]-tmpp[2]+1e-6)
        tmpp = (1.0-t) * tmpp[:2] + t * light_pos[:, :2]
        return tmpp

def proj_bb(mask, hmap, light_pos, mouse_pos):
        tmp_lights = light_pos.copy()
        if len(light_pos.shape) == 1:
                tmp_lights = tmp_lights[..., np.newaxis]

        # bb -> four points
        highest = hmap.max()
        highest_h, highest_w = list(np.unravel_index(np.argmax(hmap), hmap.shape))
        hbb, wbb = np.nonzero(mask)
        h, hh, w, ww = hbb.min(), hbb.max(), wbb.min(), wbb.max()
        bb0, bb1, bb2, bb3 = np.array([w, h, hmap.max()]), np.array([ww, h, hmap.max()]), np.array([w, hh, 0]), np.array([ww, hh, 0])

        # compute projection for the four points
        tmp_lights = tmp_lights.transpose(1,0)
        bb0, bb1, bb2, bb3 = proj_ground(bb0, tmp_lights), proj_ground(bb1, tmp_lights), proj_ground(bb2, tmp_lights), proj_ground(bb3, tmp_lights)

        batch = len(tmp_lights)
        new_bb = np.zeros((batch, 4))
        for i in range(batch):
                new_bb[i, 0] = min([bb0[i, 1], bb1[i,1], bb2[i, 1], bb3[i, 1], mouse_pos[1], h]) # h
                new_bb[i, 1] = max([bb0[i, 1], bb1[i,1], bb2[i, 1], bb3[i, 1], mouse_pos[1], hh])
                new_bb[i, 2] = min([bb0[i, 0], bb1[i,0], bb2[i, 0], bb3[i, 0], mouse_pos[0], w]) # w
                new_bb[i, 3] = max([bb0[i, 0], bb1[i,0], bb2[i, 0], bb3[i, 0], mouse_pos[0], ww])

        return new_bb

def to_torch_device(np_img):
        if len(np_img.shape) == 3:
                return to_tensor(np_img).float().unsqueeze(dim=0).contiguous().to(device)
        else:
                return torch.from_numpy(np_img).float().contiguous().to(device)

def hshadow_render(rgb, mask, hmap, rechmap, light_pos, mouse_pos):
        """ Heightmap Shadow Rendering
                rgb:            H x W x e
                mask:           H x W x 1
                hmap:           H x W x 1
                rechmap:        H x W x 1
                light_pos:  (3,B)
                return:
                        shadow masking
        """

        hbb, wbb = np.nonzero(mask[...,0])
        # speed optimization
        bb = proj_bb(mask[...,0], hmap[...,0], light_pos, mouse_pos)

        # import pdb; pdb.set_trace()
        if len(light_pos.shape) == 1:
                light_pos_d = torch.from_numpy(light_pos).to(device).unsqueeze(dim=0).float()
                rgb_d, mask_d, hmap_d, rechmap_d = to_torch_device(rgb), to_torch_device(mask), to_torch_device(hmap), to_torch_device(rechmap)
                bb_d = torch.from_numpy(bb).float().to(device)
                batch = 1
        else:
                light_pos_d = torch.from_numpy(np.ascontiguousarray(light_pos.transpose(1,0))).float().to(device)
                batch = len(light_pos_d)
                h,w = rgb.shape[:2]
                rgb_d = to_torch_device(np.repeat(rgb[np.newaxis,...].transpose(0,3,1,2), batch, axis=0))
                mask_d = to_torch_device(np.repeat(mask[np.newaxis,...].transpose(0,3,1,2), batch, axis=0))
                hmap_d = to_torch_device(np.repeat(hmap[np.newaxis,...].transpose(0,3,1,2), batch, axis=0))
                rechmap_d = to_torch_device(np.repeat(rechmap[np.newaxis,...].transpose(0,3,1,2), batch, axis=0))
                bb_d = torch.from_numpy(np.ascontiguousarray(bb)).float().to(device)

        shadow = hshadow.forward(rgb_d, mask_d, bb_d, hmap_d, rechmap_d, light_pos_d)
        # mask_top_pos = list(np.unravel_index(np.argmax(hmap), hmap.shape))
        # x,y = mask_top_pos[1], mask_top_pos[0]
        # mh = hmap[y,x,0]
        # light_top_d = light_pos_d - torch.tensor([[x,y,mh]]).to(light_pos_d)
        # weights = torch.abs(light_top_d[:,2]/torch.sqrt((light_top_d[:,0] **2 + light_top_d[:,1] **2)))
        # print('weights: ', weights)
        # weights = (weights)/weights.sum()

        # print(weights.shape, shadow[0].shape)
        # flipped = (weights[...,None, None,None] * (1.0-shadow[0])).sum(dim=0, keepdim=True)
        # shadow = shadow[0].sum(dim=0, keepdim=True)/len(shadow[0])
        # return (1.0-flipped)[0].detach().cpu().numpy().transpose(1,2,0)

        shadow = shadow[0].sum(dim=0, keepdim=True)/len(shadow[0])
        return shadow[0].detach().cpu().numpy().transpose(1,2,0)

def refine_shadow(shadow, intensity=0.6, filter=5):
        shadow[...,0] = uniform_filter(shadow[...,0], size=filter)
        shadow[...,1] = uniform_filter(shadow[...,1], size=filter)
        shadow[...,2] = uniform_filter(shadow[...,2], size=filter)
        return 1.0 - (1.0-shadow) * intensity

def render_ao(rgb, mask, hmap):
        rechmap = np.zeros_like(hmap)
        hbb, wbb = np.nonzero(mask[...,0])
        # light_pos = np.array([hbb.min(), (wbb.min() + wbb.max()) * 0.8, -100000])
        light_pos = np.array([-1300.10811363, -46999.86253089, 46486.73121776])
        mouse_pos = light_pos

        shadow = hshadow_render(rgb, mask, hmap, rechmap, light_pos, mouse_pos)
        softshadow = style_hardshadow(mask, shadow[..., :1], 0.45)[0]
        softshadow = refine_shadow(softshadow)
        return softshadow

def ao_composite(rgb, mask, hmap, rechmap, light_pos, mouse_pos):
        # shadow = hshadow_render(rgb, mask, hmap, rechmap, light_pos, mouse_pos)
        # softshadow = style_hardshadow(mask, shadow, 0.45)[0]
        # softshadow = refine_shadow(softshadow)

        softshadow = render_ao(rgb, mask, hmap)
        mask_ = np.repeat(mask, 3, axis=2)
        return (1.0-mask_) * softshadow * rgb + mask_ * rgb, softshadow.copy()


def render_shadow(rgb, mask, hmap, rechmap, light_pos, mouse_pos, softness, shadow_intensity=0.6):
        shadow = hshadow_render(rgb, mask, hmap, rechmap, light_pos, mouse_pos)

        if softness is not None:
                shadow, dbgs = style_hardshadow(mask, shadow[..., :1], softness)
        else:
                dbgs = None

        shadow = refine_shadow(shadow, intensity=shadow_intensity)
        return shadow, dbgs


def hshadow_composite(rgb, mask, hmap, rechmap, light_pos, mouse_pos, softness, shadow_intensity=0.6):
        """ Shadow Rendering and Composition
                rgb:            H x W x 3
                mask:           H x W x 1
                hmap:           H x W x 1
                rechmap:        H x W x 1
                light_pos:  [x,y,h]
                return:
                        Compositied image
        """
        shadow, dbgs = render_shadow(rgb, mask, hmap, rechmap, light_pos, mouse_pos, softness, shadow_intensity)
        mask_ = np.repeat(mask, 3, axis=2)
        return (1.0-mask_) * shadow * rgb + mask_ * rgb, shadow.copy(), dbgs

def vis_horizon(fov, horizon, h, w):
        # fov, horizon = 120, 400
        camera = torch.tensor([[fov, horizon]])
        planes = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

        camera = camera.float().to(device)
        planes = planes.float().to(device)

        ground_vis = plane_visualize.forward(planes, camera, h, w)[0]
        return 1.0-ground_vis[0].detach().cpu().numpy().transpose(1,2,0)
