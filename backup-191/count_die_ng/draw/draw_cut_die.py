import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import cv2
import glob

def main():
    cut_die_files = "/var/cdy_data/aoyang/counter_img/NG/*/cut_die.txt"
    for cut_die_file in glob.glob(cut_die_files):
        save_path = os.path.dirname(cut_die_file)        
        draw_file_die(cut_die_file, save_path)

def draw_file_die(cut_die_file, save_path):
    img_height, img_width = 8000, 8000
    
    with open(cut_die_file) as f:
        lines = f.readlines()
    dies_rect = []
    for line in lines:
        x1, y1, x2, y2, *_ = map(int, line.split("_"))
        dies_rect.append([x1, y1, x2, y2])
    # 创建白板
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    for die_rect in dies_rect:
        x1, y1, x2, y2 = die_rect
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 128, 0), thickness=-1)
    cv2.imwrite(os.path.join(save_path, "cut_die.jpg"), img)

if __name__ == "__main__":
    main()