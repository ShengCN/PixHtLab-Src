import argparse

class params():
    """ Singleton class for doing experiments """
    
    class __params():
        def __init__(self):
            self.norm = 'group_norm'
            self.prelu = False
            self.weight_decay = 5e-4
            self.small_ds = False
            self.multi_gpu = False
            self.log = False
            self.input_channel = 1
            self.vis_port = 8002
            self.cpu = False
            self.pred_touch = False
            self.tbaseline = False
            self.touch_loss = False
            self.input_channel = 1

        def set_params(self, options):
            self.options = options
            self.norm = options.norm
            self.prelu = options.prelu
            self.weight_decay = options.weight_decay
            self.small_ds = options.small_ds
            self.multi_gpu = options.multi_gpu
            self.log = options.log
            self.input_channel = options.input_channel
            self.vis_port = options.vis_port
            self.cpu = options.cpu
            self.ds_folder = options.ds_folder
            self.pred_touch = options.pred_touch
            self.tbaseline = options.tbaseline
            self.touch_loss = options.touch_loss
            
        def __str__(self):
            return 'norm: {}  prelu: {} weight decay: {} small ds: {}'.format(self.norm, self.prelu, self.weight_decay, self.small_ds)

    # private static variable
    param_instance = None
    
    def __init__(self):
        if not params.param_instance:
            params.param_instance = params.__params()
    
    def get_params(self):
        return params.param_instance
    
    def set_params(self, options):
        params.param_instance.set_params(options)

def parse_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=16)
    parser.add_argument('--batch_size', type=int, default=28, help='input batch size during training')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.003, help='learning rate, default=0.005')
    parser.add_argument('--beta1', type=float, default=0.9, help='momentum for SGD, default=0.9')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--relearn', action='store_true', help='forget previous best validation loss')
    parser.add_argument('--weight_file',type=str,  help='weight file')
    parser.add_argument('--multi_gpu', action='store_true', help='use multiple GPU training')
    parser.add_argument('--timers', type=int, default=1, help='number of epochs to train for')
    parser.add_argument('--use_schedule', action='store_true',help='use automatic schedule')
    parser.add_argument('--patience', type=int, default=2, help='use automatic schedule')
    parser.add_argument('--exp_name', type=str, default='l1 loss',help='experiment name')    
    parser.add_argument('--norm', type=str, default='group_norm', help='use group norm')
    parser.add_argument('--ds_folder', type=str, default='./dataset/general_dataset', help='Dataset folder')
    parser.add_argument('--hd_dir', type=str, default='/mnt/yifan/data/Adobe/HD_styleshadow/', help='Dataset folder')
    parser.add_argument('--prelu', action='store_true', help='use prelu')
    parser.add_argument('--small_ds', action='store_true', help='small dataset')
    parser.add_argument('--log', action='store_true', help='log information')
    parser.add_argument('--vis_port', default=8002,type=int, help='visdom port')
    parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay for model weight')
    parser.add_argument('--save', action='store_true', help='save batch results?')
    parser.add_argument('--cpu', action='store_true', help='Force training on CPU')
    parser.add_argument('--pred_touch', action='store_true', help='Use touching surface')
    parser.add_argument('--input_channel', type=int, default=1, help='how many input channels')
    
    # based on baseline method, for fine tuning
    parser.add_argument('--from_baseline', action='store_true', help='training from baseline')
    parser.add_argument('--tbaseline', action='store_true', help='T-baseline, input two channels')
    parser.add_argument('--touch_loss', action='store_true', help='Use touching loss')
    

    arguments = parser.parse_args()
    parameter = params()
    parameter.set_params(arguments)
    
    return arguments
