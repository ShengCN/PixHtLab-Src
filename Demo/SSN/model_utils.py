import os
import yaml
import logging

import torch


def parse_configs(config: str):
    """ Parse the config file and return a dictionary of configs
    :param config: path to the config file
    :returns:
    """
    if not os.path.exists(config):
        logging.error('Cannot find the config file: {}'.format(config))
        exit()

    with open(config, 'r') as stream:
        try:
            configs=yaml.safe_load(stream)
            return configs

        except yaml.YAMLError as exc:
            logging.error(exc)
            return {}


def load_model(config: str, weight: str, model_def, device):
    """ Load the model from the config file and the weight file
    :param config: path to the config file
    :param weight: path to the weight file
    :param model_def: model class definition
    :param device: pytorch device
    :returns:
    """
    assert os.path.exists(weight), 'Cannot find the weight file: {}'.format(weight)
    assert os.path.exists(config), 'Cannot find the config file: {}'.format(config)


    opt = parse_configs(config)
    model = model_def(opt)
    cp = torch.load(weight, map_location=device)

    models = model.get_models()
    for k, m in models.items():
        m.load_state_dict(cp[k])
        m.to(device)

    model.set_models(models)
    return model
