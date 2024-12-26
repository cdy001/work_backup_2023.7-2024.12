# coding=utf-8
import os
import shutil
import sys
import glob
import ast

print(os.getcwd())
sys.path.append(os.getcwd())

from csv_file import save_read_csv

# 复制文件
def copy(path_list, save_path):
    for path in path_list:
        shutil.copy(path, save_path)
        # shutil.move(path, save_path)

# main
def pre_copy_img():
    # die路径
    result_folders = os.path.join("/media/data/code_and_dataset/wafer/1023-0945W/result", "*die")
    # 结果保存路径
    save_path_root = os.path.join("/media/data/code_and_dataset/wafer/1023-0945W/result", "defect")
    # save_path_root = os.path.join("/media/data/code_and_dataset/wafer/1020-0945Y/result", "high_pro_defect")
    if not os.path.exists(save_path_root):
        os.makedirs(save_path_root)
    # 创建标签值与标签名对应的字典
    labels = dict()
    # 读取标签文件
    with open('models_label/0945.txt') as f:
        label_dict = f.read()
    # 转换字典字符串为字典
    label_dict = ast.literal_eval(label_dict)
    for key, val in label_dict.items():
        labels[val] = key
    # 创建die路径的列表，用于保存缺陷die路径
    die_paths = [[] for _ in range(len(labels))]
    # 逐wafer寻找与txt文件对应的die路径
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
            if die_name in img_path_dict:
                if label != 0:
                # if label == 0 and pro1 >= 0.99999:
                    die_path = os.path.join(img_path_dict[die_name], die_name+'.bmp')
                    die_paths[label].append(die_path)

    num_all_images = 0
    # 逐标签保存
    for label, die_path_label in enumerate(die_paths):
        label_name = labels[label]
        print(f"number of {label_name}: {len(die_paths[label])}")
        num_all_images += len(die_paths[label])
        save_path = os.path.join(save_path_root, label_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        copy(die_path_label, save_path)
    print(f"number of all images is {num_all_images}")


if __name__ == "__main__":
    pre_copy_img()