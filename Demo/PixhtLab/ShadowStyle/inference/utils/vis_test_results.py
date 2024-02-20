import numpy as np
import json
import pdb
import os
from os.path import join
import html
from tqdm import tqdm
import argparse
import pandas as pd
import cv2 as cv

base_softness = "0.1"
base_exp_name = "fov_results_real_89"
root_dir = "/mnt/share/yifan/code/soft_shadow-master/vis_res/"
base_dir = root_dir + base_exp_name + '/' + base_softness

exp_name = ["fov_results_real_hd", "fov_results_real_hd_0713"]

softness = [0.1, 0.2]

case_name = ["case1", "case2", "case7"]
