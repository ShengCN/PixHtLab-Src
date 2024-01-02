import objaverse
import multiprocessing
import random

processes = 8

random.seed(0)
uids = objaverse.load_uids()
random_object_uids = random.sample(uids, 100)
objects = objaverse.load_objects(
    uids=random_object_uids,
    download_processes=processes
)


objects = objaverse.load_objects(
    uids=random_object_uids,
    download_processes=processes
)

import pdb; pdb.set_trace()
