import os
import sys

sys.path.append(os.getcwd())

import cv2
import time

from cut.read_imgs import read_img
from cut.refer_die import find_dies
from cut.cut_light import cut_die_light
from cut.config.config import get_config


def cut_die_single(
    refer_path=None,
    light_list=None,
    die_para=None,
    binary_para=None,
    model_para=None,
    recipe=None,
):
    print(f"image path: {refer_path}")
    find_die_start = time.time()
    # 读取图片
    light_image = read_img(refer_path, light_list)

    # 查找标准光源的die坐标
    dies, die_twin, dies_rotate = find_dies(
        light_image=light_image,
        die_para=die_para,
        binary_para=binary_para,
        recipe=recipe,
    )
    all_dies = dies + die_twin + dies_rotate
    find_die_end = time.time()
    print(f"find dies time : {find_die_end - find_die_start}")

    # 切全部光源的图片
    cut_die_start = time.time()
    die_name_single, die_mat_single = [], []
    if len(all_dies) > 0:
        die_name_single, die_mat_single = cut_die_light(
            light_image=light_image,
            light_list=light_list,
            refer_dies=all_dies,
            die_para=die_para,
            model_para=model_para,
            recipe=recipe,
        )
    cut_die_end = time.time()
    print(f"final cut die count: {len(die_name_single)}")
    print(f"cut  dies time : {cut_die_end - cut_die_start}")

    # 绘制矩形框
    # save_image(refer_img_path_s, all_dies, False)
    # save_image(refer_img_path_s, die_name_single, True)

    return die_name_single, die_mat_single

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


if __name__ == "__main__":
    # 切die测试
    path = "/var/cdy_data/aoyang/wafer/HNA52K19240C05/R02C10L1.bmp"
    save_path = os.path.split(path)[0] + "_test"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    die_names, die_mats = cut_die_single_img(path=path, recipe="BOB1J37H-AA")
    img_disp = cv2.imread(path, 1)
    for i, die_mat in enumerate(die_mats):
        die_name = die_names[i]
        cls_path = os.path.join(save_path, str(die_name) + ".bmp")

        # try:
        #     print(cls_path)
        #     cv2.imwrite(cls_path, die_mat)
        # except:
        #     print("缺失光源图片")
        #     continue

        coordinate_flags = die_name.split("#")[1]
        x_min, y_min, x_max, y_max = map(int, coordinate_flags.split("_")[:4])
        cv2.rectangle(img_disp, (x_min, y_min), (x_max, y_max), (0, 0, 255), 5)

    cv2.imwrite(os.path.join(save_path, os.path.split(path)[1]), img_disp)
