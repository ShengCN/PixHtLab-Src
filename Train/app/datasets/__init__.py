# SRC: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/__init__.py
import logging
import importlib
from models.abs_model import abs_model
from torch.utils.data import DataLoader


def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".
    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    # dataset_filename = "datasets." + dataset_name + "_dataset"
    dataset_filename = "datasets." + dataset_name

    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    # target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    target_dataset_name = dataset_name

    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset = cls

    if dataset is None:
        log_err = "In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name)
        logging.error(log_err)
        raise NotImplementedError(log_err)

    return dataset


def create_dataset(opt, is_eval=False) -> dict:
    """Create a dataset given the option.
    """
    if is_eval:
        batch_size = opt['hyper_params']['eval_batch']
        is_shuffle = False
    else:
        batch_size = opt['hyper_params']['batch_size']
        is_shuffle = True

    num_workers   = opt['hyper_params']['workers']
    ds_opt        = opt['dataset']
    dataset_name  = ds_opt['name']
    dataset_class = find_dataset_using_name(dataset_name)

    train_ds = dataset_class(ds_opt, is_training = True)
    eval_ds  = dataset_class(ds_opt, is_training = False)

    ret = {'train': DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True),
           'eval': DataLoader(eval_ds, batch_size=batch_size, num_workers=num_workers, shuffle=is_shuffle, drop_last=False)}

    if 'test_dataset' in opt.keys(): # Use standalone test dataset
        test_ds_opt   = opt['test_dataset']
        test_ds_name  = test_ds_opt['name']
        test_ds_class = find_dataset_using_name(test_ds_name)
        test_ds       = test_ds_class(test_ds_opt, is_training = False)
        ret['test']   = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=is_shuffle, drop_last=False)


    return ret
