import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import cv2
import glob
import re

def main():
    cut_die_files = "/var/cdy_data/aoyang/counter_img/test-0924/*/defect.txt"
    i = 0
    for cut_die_file in glob.glob(cut_die_files):
        i += 1
        print(cut_die_file, f"{i}/{len(glob.glob(cut_die_files))}")
        save_path = os.path.dirname(cut_die_file)        
        draw_file_die(cut_die_file, save_path)

def draw_file_die(txt_file, save_path):
    root_name = os.path.dirname(txt_file)
    img_path = os.path.join(root_name, "original.jpg")
    
    with open(txt_file) as f:
        lines = f.readlines()
    dies_rect = []
    for line in lines:
        line = line.strip()
        # x1, y1, x2, y2, *_ = map(int, line.split("_"))
        x1, y1, x2, y2 = map(int, re.split("[_,]", line)[:4])
        dies_rect.append([x1, y1, x2, y2])
    # 创建白板
    img = cv2.imread(img_path, 1)
    for die_rect in dies_rect:
        x1, y1, x2, y2 = die_rect
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
    cv2.imwrite(os.path.join(save_path, "defect.jpg"), img)

if __name__ == "__main__":
    main()