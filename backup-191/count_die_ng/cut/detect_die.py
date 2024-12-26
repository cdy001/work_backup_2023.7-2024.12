import os
import cv2
import numpy as np
import time
import re

from cut.config import get_config_new
from cut.retrive_contours import retrive_contours_binary


def _contours2dies(contours, img_height, img_width, die_height, die_width):
    dies = []
    dies_multi_core = []
    die_Area = die_height * die_width
    for _, contorPoint in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contorPoint)
        contour_area = cv2.contourArea(contorPoint)
        rect_area = w * h
        if x < 2 or y < 2 or x + w > img_width - 2 or y + h > img_height - 2:
            continue
        if contour_area < 0.6 * die_Area or contour_area > 20 * die_Area:
            continue
        # 多胞
        if contour_area > 1.5 * die_Area:  # 多胞判定
            dies_multi_core.append([x, y, x + w, y + h, contour_area, rect_area])
        elif 0.6 * die_Area < contour_area < 1.5 * die_Area:
            dies.append([x, y, x + w, y + h, contour_area, rect_area])
        else:
            continue

    # 将轮廓面积较小的过滤掉当作空洞以复判色差
    contour_areas = np.array([die[4] for die in dies])
    contour_area_mean = np.mean(contour_areas)
    indexes = np.where(contour_areas < 0.9 * contour_area_mean)[0]
    dies = [die for i, die in enumerate(dies) if i not in indexes]
    
    return dies, dies_multi_core

def dieDetect(img_path):
    time_start = time.time()
    # root_path, img_name = os.path.split(img_path)
    img_name = os.path.basename(img_path)
    # 使用多个分隔符分割字符串
    recipe, correct_die_number = re.split("[_.]", img_name)[-3:-1]
    # print(recipe, die_number)
    # 获取配置
    image_info, binary_info = get_config_new(recipe)
    lower_thr = binary_info["lower_threshold"]
    upper_thr = binary_info["upper_threshold"]
    struct_element_tuple = binary_info["struct_element_tuple"]
    die_height = image_info["die_height"]
    die_width = image_info["die_width"]
    time_read_img = time.time()
    img = cv2.imread(img_path, flags=0)
    img_height, img_width = img.shape[:2]
    time_preprocess_end = time.time()
    print(f"time read image: {time_preprocess_end - time_read_img}")
    # print(f"time preprocess: {time_preprocess_end - time_start}")
    contours = retrive_contours_binary(
        img=img,
        lower_thr=lower_thr,
        upper_thr=upper_thr,
        struct_element_tuple=struct_element_tuple
    )
    time_1 = time.time()
    print(f"time retrive contours: {time_1 - time_preprocess_end}")
    dies, dies_mutil_core = _contours2dies(contours, img_height, img_width, die_height, die_width)
    print(f"time contours to die: {time.time() - time_1}")
    
    return dies, dies_mutil_core, correct_die_number, [die_height, die_width]
