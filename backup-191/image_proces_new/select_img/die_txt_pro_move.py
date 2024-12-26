# coding=utf-8
import os
import shutil
import sys
import glob

print(os.getcwd())
sys.path.append(os.getcwd())

from csv_file import save_read_csv

# 复制文件
def copy(path_list, save_path):
    len_path_list = len(path_list)
    for i, path in enumerate(path_list):
        print(f"{i+1}/{len_path_list}")
        shutil.copy(path, save_path)
        # shutil.move(path, save_path)

# main
def pre_copy_img():
    # die路径
    result_folders = os.path.join("/media/data/code_and_dataset/wafer/1024-0945W/result", "*die")
    # 结果保存路径
    save_path = os.path.join("/media/data/code_and_dataset/wafer/1024-0945W/result", "low_pro_Good")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 创建die路径的列表，用于保存低概率的die路径
    die_paths = []
    # 逐wafer
    for folder in glob.glob(result_folders):
        # 图像路径列表
        img_list = glob.glob(os.path.join(folder, '*/*bmp'))
        # 图像名字和路径对应的字典，用于后续查询图像路径
        img_path_dict = dict()
        for img_path in img_list:
            img_folder, img_name = os.path.split(img_path)
            img_name = img_name.split('.bmp')[0]
            img_path_dict[img_name] = img_folder
        # txt路径
        txt_folder = folder.replace('_die', '')
        txt_path = os.path.join(txt_folder, "predict_out.txt")
        # 读取csv文件
        data = save_read_csv.read_csv(txt_path)
        for die_name, label, pro1 in zip(data["name"], data["label"], data["pro_1"]):
            if label == 0 and pro1 < 0.8:
                if die_name in img_path_dict:
                    die_path = os.path.join(img_path_dict[die_name], die_name+'.bmp')
                    die_paths.append(die_path)
    copy(die_paths, save_path)
    print(f"number of images is {len(die_paths)}")


if __name__ == "__main__":
    pre_copy_img()