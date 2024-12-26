import os
import sys
sys.path.append(os.getcwd())
import cv2
import time
import numpy as np
from collections import defaultdict

from cut.config import get_config_new
from cut.retrive_contours import retrive_contours_edge


def hole_or_die(die_height, die_width, img):
    # 直方图均衡化
    equalized = cv2.equalizeHist(img)
    # 调整阈值，可以尝试不同的上下限值，或者计算自适应阈值
    v = np.median(equalized)
    lower = int(max(0, 0.7 * v))
    upper = int(min(255, 1.3 * v))
    contours, edges = retrive_contours_edge(
        img=img,
        Canny_thrs=(lower, upper),
        # close_struct=(5, 5)
        )

    die_rects = []
    die_Area = die_height * die_width
    for _, contorPoint in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contorPoint)
        contour_area = cv2.contourArea(contorPoint)
        if contour_area < 0.6 * die_Area:
            continue
        die_rects.append([x, y, x + w, y + h])
    return die_rects

def find_color_diff(img_path, cut_die_path, sort_die_path, recipe, save_result_img=False):    
    cut_die_bbox = defaultdict()
    with open(cut_die_path) as f:
        cut_die_lines = f.readlines()
    for line in cut_die_lines:
        line = line.strip()
        real_bbox = list(map(int, line.split("_")[:4]))
        sort_bbox = "_".join(line.split("_")[-4:])
        if sort_bbox not in cut_die_bbox:
            cut_die_bbox[sort_bbox] = real_bbox

    time_start = time.time()
    color_diff_number = 0
    with open(sort_die_path) as f:
        sort_die_lines = f.readlines()
    sort_die_lines = sorted(sort_die_lines, key=lambda x: list(map(int, x.split("_")[:2])))
    holes = []
    true_dies = defaultdict()
    for line in sort_die_lines:
        line = line.strip()
        col, row, xmin, ymin, xmax, ymax, is_true = map(int, line.split("_"))
        # if not is_true:
        #     holes.append([row, col, xmin, ymin, xmax, ymax])
        # else:
        #     true_dies[f"{row}-{col}"] = [xmin, ymin, xmax, ymax]
        if not is_true:
            holes.append([row, col, xmin, ymin, xmax, ymax])
        else:
            true_dies[f"{row}-{col}"] = cut_die_bbox[f"{xmin}_{ymin}_{xmax}_{ymax}"]

    
    # 获取配置
    image_info, binary_info = get_config_new(recipe)
    die_height = image_info["die_height"]
    die_width = image_info["die_width"]

    img = cv2.imread(img_path, flags=0)    
    print(img.shape)
    
    die_color_diff = []
    for hole in holes:
        row, col = hole[:2]

        # left
        left = [f"{row}-{col-1}", f"{row-1}-{col-1}", f"{row+1}-{col-1}"]
        # right
        right = [f"{row}-{col+1}", f"{row-1}-{col+1}", f"{row+1}-{col+1}"]
        # top
        top = [f"{row-1}-{col}", f"{row-1}-{col-1}", f"{row-1}-{col+1}"]
        # bottom
        bottom = [f"{row+1}-{col}", f"{row+1}-{col-1}", f"{row+1}-{col+1}"]
        x1_left = [true_dies[row_col][2] for row_col in left if row_col in true_dies]
        x2_right = [true_dies[row_col][0] for row_col in right if row_col in true_dies]
        y1_top = [true_dies[row_col][3] for row_col in top if row_col in true_dies]
        y2_bottom = [true_dies[row_col][1] for row_col in bottom if row_col in true_dies]
        x1 = max(x1_left) + 2 if len(x1_left) > 0 else hole[2] - die_width // 3
        x2 = min(x2_right) - 2 if len(x2_right) > 0 else hole[4] + die_width // 3
        y1 = max(y1_top) + 2 if len(y1_top) > 0 else hole[3] - die_height // 3
        y2 = min(y2_bottom) - 2 if len(y2_bottom) > 0 else hole[5] + die_height // 3
        if y2 - y1 < die_height or x2 - x1 < die_width:
            print("error hole")
            continue
        img_hole = img[y1:y2, x1:x2]
        die_rects = hole_or_die(die_height, die_width, img_hole)
        if len(die_rects) > 0:
            color_diff_number += 1
        for die_rect in die_rects:
            x1_sub, y1_sub, x2_sub, y2_sub = die_rect
            x1_true = x1_sub + x1
            y1_true = y1_sub + y1
            x2_true = x2_sub + x1
            y2_true = y2_sub + y1
            die_rect = [x1_true, y1_true, x2_true, y2_true]
            die_color_diff.append(die_rect)
        # die_color_diff.append([x1, y1, x2, y2])
    
    if save_result_img:
        save_path = os.path.dirname(img_path)
        img_disp = cv2.imread(img_path, 1)
        for i, die_rect in enumerate(die_color_diff):
            x_min, y_min, x_max, y_max = die_rect
            cv2.rectangle(img_disp, (x_min,y_min), (x_max,y_max), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(save_path, "color_diff.jpg"), img_disp)

    print(f"count of cut die finally: {len(die_color_diff)}")
    print(f"time consume: {time.time() - time_start}s")    

    return color_diff_number, die_color_diff
    
def main():
    img_path = '/var/cdy_data/aoyang/counter_img/display/color_diff/HAE2240625045735CA146/original.jpg'    
    sort_die_path = '/var/cdy_data/aoyang/counter_img/display/color_diff/HAE2240625045735CA146/sort_die.txt'
    cut_die_path = "/var/cdy_data/aoyang/counter_img/display/color_diff/HAE2240625045735CA146/cut_die.txt"
    # recipe = 'BPB0L32K'
    recipe = 'BPB0Q35G'

    color_diff_number, die_color_diff = find_color_diff(img_path, cut_die_path, sort_die_path, recipe, save_result_img=True)
    print(color_diff_number)
    print(len(die_color_diff))

if __name__ == '__main__':
    main()