import sys
import os
sys.path.append(os.getcwd())

import glob

from utils.utils_func import check_or_mkdirs, copy

def main():
    xml_paths = glob.glob('/var/cdy_data/obj_det/0945W_new/*xml')
    save_path = '/var/cdy_data/obj_det/0945W_1204'
    image_paths = []
    for xml_path in xml_paths:
        # print(xml_path)
        image_path = xml_path.replace('.xml', '.bmp')
        # print(image_path)
        image_paths.append(image_path)

    check_or_mkdirs(save_path)
    copy(xml_paths, save_path)
    copy(image_paths, save_path)

if __name__ == '__main__':
    main()