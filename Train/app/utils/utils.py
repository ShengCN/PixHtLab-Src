import os
from os.path import join
import yaml
import logging
import argparse
import datetime


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='csv')
    params = parser.parse_args()

    with open(params.config, 'r') as stream:
        try:
            configs=yaml.safe_load(stream)
            return configs
        except yaml.YAMLError as exc:
            logging.error(exc)
            return {}


def logging_init(exp_name, folder='logs', style='verbose'):
    fmt_styles = {
        'verbose': "[%(levelname)s] %(message)s - %(filename)s %(funcName)s %(asctime)s ",
        'light': "[%(levelname)s] %(message)s",
    }

    os.makedirs(folder, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
        format=fmt_styles.get(style),
        handlers=[logging.FileHandler(join(folder, "{}.log".format(exp_name))), logging.StreamHandler()])


def get_cur_time_stamp():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    # size_all_mb = (param_size + buffer_size) / 1024**2
    # print('model size: {:.3f}MB'.format(size_all_mb))
    return param_size + buffer_size
