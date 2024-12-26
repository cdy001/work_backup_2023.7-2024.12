import os
import sys
# 添加项目根目录到系统环境变量搜索路径
sys.path.append(os.getcwd())
import glob
import cv2
import time

from cut.detect_die import dieDetect
from ng_die_recognize.angle_skew import ngSkew


def main():
    # 获取当前脚本文件的绝对路径
    file_path = os.path.abspath(__file__)
    # 获取文件所在的目录
    current_directory = os.path.dirname(file_path)
    img_paths = 'data/NG/*.JPG'
    for img_path in glob.glob(img_paths):
        print(img_path)

        # 晶粒识别
        time_start = time.time()
        dies, correct_die_number = dieDetect(img_path=img_path)
        time_detect_end = time.time()

        print(f"time detect dies: {time_detect_end - time_start}")
        
        # 晶粒歪斜判断
        indexs = ngSkew(dies)

        ng_die = 0
        img_disp = cv2.imread(img_path, 1)
        if indexs.size:
            for i, die in enumerate(dies):
                x_min, y_min, x_max, y_max, contour_area, rect_area = dies[i]
                if i in indexs:
                    cv2.rectangle(img_disp, (x_min,y_min), (x_max,y_max), (0, 0, 255), 2)
                    ng_die += 1
                else:
                    cv2.rectangle(img_disp, (x_min,y_min), (x_max,y_max), (0, 255, 0), 2)

        print(f"count of cut die finally: {len(dies)}")
        print(f"correct count of die: {correct_die_number}")
        print(f"ng_die number: {ng_die}")
        print(f"time draw contours: {time.time() - time_detect_end}s")

        # save
        save_path = os.path.join(current_directory, "result")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path_img = os.path.join(save_path, os.path.split(img_path)[-1])
        cv2.imwrite(save_path_img, img_disp)



if __name__ == '__main__':
    main()