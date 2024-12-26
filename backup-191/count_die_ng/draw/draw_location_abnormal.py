import os
import sys
sys.path.append(os.getcwd())
import cv2
import time
import numpy as np
from collections import defaultdict

def find_location_abnormal(img_path, cut_die_path, sort_die_path):    
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
    location_abnormal = []
    with open(sort_die_path) as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        col, row, xmin, ymin, xmax, ymax, symbol = map(int, line.split("_"))
        if symbol == 2:
            location_abnormal.append(cut_die_bbox[f"{xmin}_{ymin}_{xmax}_{ymax}"])
    
    
    if len(location_abnormal) > 0:
        save_path = os.path.dirname(img_path)
        img_disp = cv2.imread(img_path, 1)
        for i, die_rect in enumerate(location_abnormal):
            x_min, y_min, x_max, y_max = die_rect
            cv2.rectangle(img_disp, (x_min,y_min), (x_max,y_max), (0, 0, 255), 1)
        cv2.imwrite(os.path.join(save_path, "location_abnormal.jpg"), img_disp)

    print(f"count of cut die finally: {len(location_abnormal)}")
    print(f"time consume: {time.time() - time_start}s")    

    return location_abnormal
    
def main():
    img_path = r'/var/cdy_data/aoyang/counter_img/display/hole/HAH5240904223143BA030/original.jpg'    
    sort_die_path = r'/var/cdy_data/aoyang/counter_img/display/hole/HAH5240904223143BA030/sort_die.xml'
    cut_die_path = r"/var/cdy_data/aoyang/counter_img/display/hole/HAH5240904223143BA030/cut_die.txt"

    location_abnormal = find_location_abnormal(img_path, cut_die_path, sort_die_path)
    print(len(location_abnormal))

if __name__ == '__main__':
    main()