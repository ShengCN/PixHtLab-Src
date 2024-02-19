from shutil import copyfile
from os import listdir
from os.path import isfile, join
import os
import datetime

def get_all_folders(folder):
    if check_file_exists(folder) == False:
        print("Cannot find the folder ", folder)
        return []
    subfolders = [f for f in os.listdir(folder) if not isfile(join(folder,f))]
    return subfolders

def get_all_files(folder):
    if check_file_exists(folder) == False:
        print("Cannot find the folder ", folder)
        return []
 
    ori_files = [f for f in listdir(folder) if isfile(join(folder, f))]
    return ori_files

def create_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)

def create_folders(folder_list):
    for f in folder_list:
        create_folder(f)

def replace_file_ext(fname, new_ext):
    ext_pos = fname.find(".")
    if ext_pos != -1:
        return fname[0:ext_pos] + "."+ new_ext
    else:
        print("Please check " + fname)

def check_file_exists(fname, verbose=True):
    try:
        if not os.path.exists(fname) or get_file_size(fname) == 0:
            if verbose:
                print("file {} does not exists! ".format(fname))
            return False
    except:
        print("File {} has some issue! ".format(fname))
        return False
    return True

def delete_file(fname):
    if check_file_exists(fname):
        os.remove(fname)

def get_file_size(fname):
    return os.path.getsize(fname)

def get_folder_size(folder):
    return sum(os.path.getsize(folder + f) for f in listdir(folder) if isfile(join(folder, f)))

def get_cur_time_stamp():
    return datetime.datetime.now().strftime("%d-%B-%I-%M-%p")