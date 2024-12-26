# coding=gbk
import os
import numpy as np
import sys
import cv2
import glob

print(os.getcwd())
sys.path.append(os.getcwd())

from cut.config.config import get_config
from cut.cut_die import cut_die_single
from cut.read_imgs import read_img

def cut_die_single_img(path, recipe):
    die_para = {}
    binary_para = {}
    model_para = {}
    image_info, binary_info, model_info = get_config(recipe)
    die_para["refer_light"] = image_info.get("refer_light")
    die_para["die_height"] = image_info.get("die_height")
    die_para["die_width"] = image_info.get("die_width")
    die_para["margin_y"] = image_info.get("margin_y")
    die_para["margin_x"] = image_info.get("margin_x")
    die_para["L1_offset"] = image_info.get("L1_offset")
    binary_para["threshold"] = binary_info.get("threshold")
    binary_para["binary_type"] = binary_info.get("binary_type")
    binary_para["open_close_type"] = binary_info.get("open_close_type")
    binary_para["struct_element_tuple"] = binary_info.get("struct_element_tuple")
    model_para["resize_y"] = model_info.get("resize_y")
    model_para["resize_x"] = model_info.get("resize_x")

    die_names, die_mats = cut_die_single(
        path,
        light_list=image_info["light_list"],
        die_para=die_para,
        binary_para=binary_para,
        model_para=model_para,
        recipe=recipe,
    )

    return die_names, die_mats

if __name__ == '__main__':
    recipe='3535'
    root_path = '/var/cdy_data/zhaochi/wafer/3535/NA068240417A29'
    image_info, binary_info, model_info = get_config(recipe)
    light_list=image_info["light_list"]
    save_path = os.path.join(root_path+'_die')
    print(f'save_path: {save_path}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for path in glob.glob(os.path.join(root_path, '*L1*bmp')):
        print(path)
        light_image = read_img(path, light_list)
        die_names, die_mats = cut_die_single_img(
            path=path,
            recipe=recipe
            )
        for i, die_name in enumerate(die_names):
            cls_path = os.path.join(
                save_path, str(die_name) + ".bmp"
                )
            x1, y1, x2, y2 = map(int, die_name.split("#")[-1].split("_")[:4])
            die_light = die_name.split("#")[0].split("_")[-1]
            # print(die_name)
            # print(x1, y1, x2, y2, die_light)

            try:
                # print(cls_path)
                cv2.imwrite(cls_path, light_image[die_light][y1: y2, x1: x2])
            except:
                print("缺失光源图片")
                continue
                exit()