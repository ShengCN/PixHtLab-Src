# SRC: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/__init__.py
import logging
import importlib
from models.abs_model import abs_model


def find_model_using_name(model_name):
    """Import the module "models/[model_name].py".
    In the file, the class called DatasetNameModel() will
    be instantiated. It has to be a subclass of BaseModel,
    and it is case-insensitive.
    """
    model_filename = "models." + model_name
    modellib = importlib.import_module(model_filename)
    model = None

    target_model_name = model_name
    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() \
           and issubclass(cls, abs_model):
            model = cls

    if model is None:
        err = "In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_filename, target_model_name)
        logging.error(err)
        exit(0)

    return model


def create_model(opt):
    """Create a model given the option.
    This funct
    This is the main interface between this package and 'train.py'/'test.py'
    Example:
        >>> from models import create_model
        >>> model = create_model(opt)
    """
    model = find_model_using_name(opt['model']['name'])
    instance = model(opt)
    logging.info("model [%s] was created" % type(instance).__name__)
    return instance
