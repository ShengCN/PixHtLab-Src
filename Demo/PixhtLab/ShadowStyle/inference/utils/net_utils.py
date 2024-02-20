#import matplotlib.pyplot as plt
import os
from torchvision import transforms, utils
import torch
#import matplotlib.pyplot as plt
import numpy as np
from utils.utils_file import get_cur_time_stamp, create_folder

def compute_differentiable_params(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def convert_Relight_latent_light(latent_feature):
    """ Convert n x 6 x 16 x 16 -> n x 3 x 16 x 32 """
    # torch image: C X H X W
    batch_size, C, H, W = latent_feature.size()
    latent_feature = torch.reshape(latent_feature, (batch_size, 3, 16, 32))  # make sure it is right
    # print(latent_feature.size())
    return latent_feature

def show_batch(sample_batch, out_file=None):
    grid = utils.make_grid(sample_batch)
    plt.figure(figsize=(30,20))
    plt.imshow(grid.detach().cpu().numpy().transpose((1,2,0)))

    if not out_file is None:
        print('try save ', out_file)
        plt.savefig(out_file)

    plt.show()

def show_light_batch(light_batch):
    light_batch = convert_Relight_latent_light(light_batch)
    show_batch(light_batch)
    
def save_loss(figure_fname, train_loss, valid_loss):
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.legend(['train_loss', 'valid_loss'])
    plt.savefig(figure_fname)

def save_model(output_folder, model, optimizer, epoch, best_loss, fname, hist_train_loss, hist_valid_loss, hist_lr, params):
    """ Save current best model into some folder """
    create_folder(output_folder)

    # cur_time_stamp = get_cur_time_stamp()
    # output_fname = os.path.join(output_folder, exp_name + '_' + cur_time_stamp + ".pt")
    output_fname = os.path.join(output_folder, fname)
    tmp_model = model
    if params.multi_gpu and hasattr(tmp_model, 'module'):
        tmp_model = model.module

    torch.save({
        'epoch': epoch,
        'best_loss': best_loss,
        'model_state_dict': tmp_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'hist_train_loss': hist_train_loss,
        'hist_valid_loss': hist_valid_loss,
        'hist_lr':hist_lr,
        'params':str(params)
        }, output_fname)
    return output_fname
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
