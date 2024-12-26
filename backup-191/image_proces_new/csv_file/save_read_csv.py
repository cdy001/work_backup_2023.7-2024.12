import glob
import os
import cv2
import pandas as pd
import json
import random
import numpy as np
from predict_image import read_images
from cut.config.config import get_config, get_config_new


# 将列表保存为csv文件
def save_csv(total_names, total_pre, top_values, top_labels, dest_csv_path):
    df = pd.DataFrame({
        "name": total_names,
        "label": total_pre,
        "pro_1": np.array(top_values)[:, 0],
        "pro_2": np.array(top_values)[:, 1],
        "label_1": np.array(top_labels)[:, 0],
        "label_2": np.array(top_labels)[:, 1]
        })
    print(df["label"].value_counts())

    df.to_csv(dest_csv_path, index=False)


# 读取csv文件
def read_csv(csv_path):
    file_hz = os.path.splitext(csv_path)[1]

    if "csv" in file_hz:
        data = pd.read_csv(
            csv_path,
            header=None,
            names=["name", "label", "pro_1", "pro_2", "label_1", "label_2"])
        data["name"] = data["name"].str.replace("'", "")
        data["name"] = data["name"].str.replace("b", "")
    elif "txt" in file_hz:
        # data = pd.read_csv(csv_path, header=None, sep=",", names=["name", "label"])
        data = pd.read_csv(
            csv_path,
            header=None,
            sep=",",
            names=["name", "label", "pro_1", "pro_2", "label_1", "label_2"])
    else:
        data = None

    print(data.head(10))

    return data


# 创建标签文件
def create_file(label_path, save_path):
    pass
    with open(label_path, "r") as f:
        label_dict = json.load(f)

    # # 创建低概率文件夹
    # low_pro_good_path = os.path.join(save_path, "100")
    # if not os.path.exists(low_pro_good_path):
    #         os.makedirs(low_pro_good_path)
    # for label in label_dict:
    #     # cls_path = os.path.join(save_path, str(label_dict[label]))
    #     if label == 'Good':
    #         continue
    #     cls_path = os.path.join(save_path, label)
    #     if not os.path.exists(cls_path):
    #         os.makedirs(cls_path)

    for label in label_dict:
        if label != 'Luminous_ITO':
            continue
        cls_path = os.path.join(save_path, label)
        if not os.path.exists(cls_path):
            os.makedirs(cls_path)


# 修改标签文件名字
def rename_file(save_path, label_path):
    with open(label_path, "r") as f:
        label_dict = json.load(f)
    in_label_dict = {}
    for key in label_dict.keys():
        value = label_dict[key]
        in_label_dict[value] = key

    for label_name in os.listdir(save_path):
        src_path = os.path.join(save_path, label_name)
        new_path = os.path.join(save_path, in_label_dict[label_name])

        os.rename(src_path, new_path)


# 保存die
def save_die(csv_path, img_path, save_path, label_path, img_type, recipe):
    with open(label_path, "r") as f:
        label_dict = json.load(f)

    label = list(label_dict.values())

    label_name_path = dict()
    for label_name, class_number in label_dict.items():
        label_name_path[class_number] = label_name
    print("path_dict:", label_name_path)

    image_info, _, _ = get_config_new(recipe)

    create_file(label_path, save_path)

    # 读取csv文件
    data = read_csv(csv_path)

    # 光源
    light_list = image_info["light_list"]

    paths = glob.glob(os.path.join(img_path, "*" + light_list[0] + f"*{img_type}"))
    # random.shuffle(paths)
    for path in paths:
        print(path)

        # 查找相同rc的全部die
        if "bmp" == img_type:
            rc = os.path.split(path)[1].split("L")[0]
            rc_tf = data["name"].str.startswith(rc + "_")
            rc_data = data[rc_tf]
        elif "raw" == img_type:
            rc = os.path.split(os.path.splitext(path)[0])[-1].split("_")[-1]
            rc_tf = data["name"].str.startswith(rc + "_")
            rc_data = data[rc_tf]

        # 读取全部图片
        light_img = read_images.read_img(path, light_list)

        # 循环图片名与标签
        for die_name, die_class, pro_1 in zip(rc_data["name"], rc_data["label"], rc_data["pro_1"]):
            if int(die_class) in label:
                # if str(die_class) != "6":
                #         continue
                # if "0" == str(die_class):
                #     if random.randint(0, 20) > 10:
                #         continue
                # if "1" == str(die_class):
                #     if random.randint(0, 20) > 10:
                #         continue
                # if 2 == int(
                #     os.path.splitext(die_name)[0].split("#")[-1].split("_")[-4]
                # ):
                #     continue

                x1, y1, x2, y2, *_ = (
                    os.path.splitext(die_name)[0].split("#")[-1].split("_")
                )
                cls_path = os.path.join(
                    save_path, label_name_path[int(die_class)]
                    )
                if not os.path.exists(cls_path) and \
                    label_name_path[int(die_class)] != "Good":
                    os.makedirs(cls_path)
                cls_path = os.path.join(
                    cls_path, str(die_name) + ".bmp"
                )
                die_light = os.path.splitext(die_name)[0].split("#")[0].split("_")[-1]
       
                # 低概率Good
                if "0" == str(die_class):
                    if float(pro_1) >= 0.8:
                        continue
                    else:
                        low_pro_good_path = os.path.join(save_path, "100")
                        if not os.path.exists(low_pro_good_path):
                            os.makedirs(low_pro_good_path)
                        cls_path = os.path.join(
                            save_path, "100", str(die_name) + ".bmp"
                            )
                
                # 双胞、多胞
                if int(die_name.split("_")[-4]) in [2, 100] and label_name_path[int(die_class)] != "Mark":
                    cls_path = os.path.join(
                        save_path, "Cutting_double_twin"
                        )
                    if not os.path.exists(cls_path):
                        os.makedirs(cls_path)
                    cls_path = os.path.join(
                        cls_path, str(die_name) + ".bmp"
                    )

                try:
                    cv2.imwrite(
                        cls_path,
                        light_img[die_light][int(y1) : int(y2), int(x1) : int(x2)],
                    )
                except:
                    print("缺失光源图片")
                    continue
                    exit()
