import os
import shutil
import json
from tqdm import tqdm

def check_or_mkdirs(check_path):
    '''
    check whether path is existing.
    '''
    if not os.path.exists(check_path):
        os.makedirs(check_path)

def read_label_file(label_file):
    '''
    args:
        label_file: label.txt file
    return:
        label_dict: {name: class_number, ...}
        inverted_label_dict: {class_number: name, ...}
    '''
    with open(label_file) as f:
        label_dict = json.load(f)
    inverted_label_dict = dict(zip(label_dict.values(), label_dict.keys()))
    return label_dict, inverted_label_dict

# copy files
def copy(path_list, save_path):
    """
    copy files(directories) to save_path
    args:
        path_list: files(directories) path list
        save_path: 
    """
    for path in tqdm(path_list):
        if os.path.isfile(path):
            try:
                shutil.copy(path, save_path)
                # shutil.move(path, save_path)
            except:
                continue
        elif os.path.isdir(path):
            files = os.listdir(path)
            _, path_name = os.path.split(path)
            save_file_path = os.path.join(save_path, path_name)
            check_or_mkdirs(save_file_path)
            for file_name in files:
                file_path = os.path.join(path, file_name)
                try:
                    shutil.copy(file_path, save_file_path)
                    # shutil.move(file_path, save_file_path)
                except:
                    continue
        else:
            continue