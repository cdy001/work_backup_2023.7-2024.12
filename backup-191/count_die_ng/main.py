import os
import sys
# 添加项目根目录到系统环境变量搜索路径
sys.path.append(os.getcwd())
import glob
import cv2
import time
import shutil

from cut.detect_die import dieDetect
from ng_die_recognize.ng_skew import angle_abnormal
from ng_die_recognize.ng_standing import OnnxYolo, inferencePerPatch


def main():
    img_paths = '/var/cdy_data/aoyang/counter_img/NG/HAFF240627013620BA043_BPB0X43G_3028.JPG'
    model_path = "ng_die_recognize/onnx_model/yolov10-0925.onnx"
    device = 0   
    model = OnnxYolo(model_path, device)
    img_number = 0
    defect_num = 0
    for img_path in glob.glob(img_paths):
        # create save_path
        save_root_path, img_name = os.path.split(img_path)
        save_path = os.path.join(save_root_path, img_name.split("_")[0])
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        img_number += 1
        print(img_path, f"{img_number}/{len(glob.glob(img_paths))}")

        # detect dies
        time_start = time.time()
        dies, dies_multi_core, correct_die_number, [die_height, die_width] = dieDetect(img_path=img_path)
        time_detect_end = time.time()

        print(f"time detect dies: {time_detect_end - time_start}")
        
        # judge ng_skew
        # indexes_whRatio = wh_ratio_abnormal(dies, die_width/die_height)
        # indexes_skew = angle_abnormal(dies, die_width/die_height)
        # indexes = np.union1d(indexes_skew, indexes_whRatio)  # 取并集
        indexes = angle_abnormal(dies, die_width/die_height, threshold_angle=10)

        # # judge ng_standing
        dies_standing = inferencePerPatch(model, img_path)
        # dies_standing = []

        defect_num += (len(indexes) + len(dies_standing) + len(dies_multi_core))

        time_judge_end = time.time()
        print(f"time judge OK-NG: {time_judge_end - time_detect_end}")
        print(f"count of cut die finally: {len(dies) + len(dies_multi_core)}, dies: {len(dies)}, dies_multi_core: {len(dies_multi_core)}")
        print(f"correct count of die: {correct_die_number}")     

        # copy original image
        shutil.copy(img_path, os.path.join(save_path, "original.jpg"))
        
        # draw rectangle of ng_dies
        draw_ng_rect(img_path, save_path, dies, dies_multi_core, indexes, dies_standing)

        # write cut_die.txt
        cut_die_file = os.path.join(save_path, "cut_die.txt")
        write_cut_dies(cut_die_file, dies, die_height, die_width)

        # write defect.txt
        defect_file = os.path.join(save_path, "defect.txt")
        write_defect_dies(dies, dies_multi_core, indexes, defect_file)
    
    print(f"defect_num: {defect_num}")


def write_cut_dies(file_path, dies, die_height, die_width):
    for die in dies:
        if die_width * die_height > 30 * 15:
            die.extend([die[0], die[1], die[0] + die_width, die[1] + die_height])
        else:
            die.extend([die[0], die[1], die[0] + die_width + 2, die[1] + die_height + 2])
    with open(file=file_path, mode="w") as f:
        for die in dies:
            f.write("_".join(map(str, die[:4])) + f"_{die_height}_{die_width}" + "_" + "_".join(map(str, die[-4:])) + "\n")

def write_defect_dies(dies, dies_multi_core, indexes, defect_file):
    with open(defect_file, mode="w") as f:
        for _, die in enumerate(dies_multi_core):
            f.write("_".join(map(str, die[:4])) + ",2" + "\n")
        for i, die in enumerate(dies):   
            if i in indexes:
                f.write("_".join(map(str, die[:4])) + ",3" + "\n")

def draw_ng_rect(img_path, save_path, dies, dies_multi_core, indexes, dies_standing):
    img_disp = cv2.imread(img_path, 1)
    for i, die in enumerate(dies_multi_core):
        x_min, y_min, x_max, y_max, contour_area, rect_area = die
        cv2.rectangle(img_disp, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), (0, 0, 255), 2)
    for i, die in enumerate(dies):
        x_min, y_min, x_max, y_max, contour_area, rect_area = die
        if i in indexes:
            cv2.rectangle(img_disp, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), (0, 0, 255), 2)
        else:
            cv2.rectangle(img_disp, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), (0, 255, 0), 1)
    for i, die in enumerate(dies_standing):
        x_min, y_min, x_max, y_max = map(int, die[:4])
        cv2.rectangle(img_disp, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), (0, 0, 255), 2)
    save_path_img = os.path.join(save_path, "result_preview.jpg")
    cv2.imwrite(save_path_img, img_disp)



if __name__ == '__main__':
    main()