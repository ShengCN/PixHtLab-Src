from visdom import Visdom
import numpy as np
import torch

# viz = Visdom(port=8002)
# viz2 = Visdom(port=8003)

def setup_visdom(port=8002):
    return Visdom(port=port)

def visdom_plot_loss(win_name, loss, cur_viz):
    loss_np = np.array(loss)
    x = np.arange(1, 1 + len(loss))
    cur_viz.line(win=win_name,
                 X=x,
                 Y=loss_np,
                 opts=dict(showlegend=True, legend=[win_name]))

def guassian_light(light_tensor):
    light_tensor = light_tensor.detach().cpu()
    channel = light_tensor.size()[0]
    tensor_ret = torch.zeros(light_tensor.size())
    for i in range(channel):
        light_np = light_tensor[0].numpy() * 100.0
        light_np = gaussian_filter(light_np, sigma=2)
        tensor_ret[i] = torch.from_numpy(light_np)
        tensor_ret[i] = torch.clamp(tensor_ret[i], 0.0, 1.0)
        
    return tensor_ret

def normalize_img(imgs):
    b,c,h,w = imgs.shape
    gt_batch = b//2
    for i in range(gt_batch):
        factor = torch.max(imgs[i])
        imgs[i] = imgs[i]/factor
        imgs[gt_batch + i] = imgs[gt_batch + i]/factor
        # imgs[i] = imgs[i]/3.0
        
    imgs = torch.clamp(imgs, 0.0,1.0)
    return imgs

def visdom_show_batch(imgs, cur_viz, win_name=None, nrow=2, normalize=True):
    if normalize:
        imgs = normalize_img(imgs)
    
    if win_name is None:
        cur_viz.images(imgs, win="batch visualize",nrow=nrow)
    else:
        cur_viz.images(imgs, win=win_name, opts=dict(title=win_name),nrow=nrow)
    
def visdom_log(log_info, viz, win_name='logger'):
    viz.text(log_info, win=win_name)