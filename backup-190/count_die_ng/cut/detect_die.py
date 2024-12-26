import os
import cv2
import numpy as np
import time
import re

from cut.config import get_config_new
from cut.retrive_contours import retrive_contours_binary
from utils.z_score import z_score_outliers

def _contours2dies(contours, img_height, img_width, die_height, die_width):
    dies = []
    dies_center = []
    die_Area = die_height * die_width
    for _, contorPoint in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contorPoint)
        contour_area = cv2.contourArea(contorPoint)
        rect_area = w * h
        if x < 2 or y < 2 or x + w > img_width - 2 or y + h > img_height - 2:
            continue
        if contour_area < 0.6 * die_Area or contour_area > 20 * die_Area:
            continue
        dies.append([x, y, x + w, y + h, contour_area, rect_area])
        dies_center.append([x + w // 2, y + h //2])
            
    # 对die过滤，通过位置异常去掉误识别的die
    dies_center = np.array(dies_center)
    index_1 = z_score_outliers(dies_center[:, 0], 3)
    index_2 = z_score_outliers(dies_center[:, 1], 3)
    indexs = list(set(index_1) | set(index_2))  # 取index_1和index_2的并集
    dies = [die for i, die in enumerate(dies) if i not in indexs]
    
    return dies

def dieDetect(img_path):
    root_path, img_name = os.path.split(img_path)
    # 使用多个分隔符分割字符串
    recipe, correct_die_number = re.split("[_.]", img_name)[-3:-1]
    # print(recipe, die_number)
    # 获取配置
    image_info, binary_info = get_config_new(recipe)
    lower_thr = binary_info["threshold"]
    struct_element_tuple = binary_info["struct_element_tuple"]
    die_height = image_info["die_height"]
    die_width = image_info["die_width"]
    img = cv2.imread(img_path, flags=0)
    img_height, img_width = img.shape[:2]
    contours = retrive_contours_binary(
        img=img,
        lower_thr=lower_thr,
        upper_thr=255,
        struct_element_tuple=struct_element_tuple
    )
    time_1 = time.time()
    dies = _contours2dies(contours, img_height, img_width, die_height, die_width)
    print(f"time contours to die: {time.time() - time_1}")
    
    return dies, correct_die_number