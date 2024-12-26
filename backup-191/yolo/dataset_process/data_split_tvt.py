import os
import sys
sys.path.append(os.getcwd())
import glob
import random
import xml.etree.ElementTree as ET

from utils.utils_func import copy
from dataset_process.data_split import dataset_path_create
from dataset_process.voc2yolo import voc2yolo

root_dir = '/var/cdy_data/obj_det/0945W_1204'  # image and label.xml
save_path_root = '/var/cdy_data/obj_det/0945W-1204'  # path to save
label_txt_file = '/data/cdy/yolo/models_and_labels/0945.txt'  # defect categories
val_rate = 0.1

def xml_2_label(xml_file):
    tree = ET.parse(xml_file)
    objects = tree.findall('object')
    if objects:
        name = objects[0].find('name').text
    else:
        name = 'none'
    return name

def main():
    train_image_path, train_label_path, val_image_path, val_label_path = dataset_path_create(save_path_root)
    xml_path = os.path.join(root_dir, "*.xml")
    xml_files = glob.glob(xml_path)
    # classify files for every categories
    label_dict = dict()
    for xml_file in xml_files:
        label = xml_2_label(xml_file)
        if label not in label_dict:
            label_dict[label] = [xml_file]
        elif label == 'none':
            continue
        else:
            label_dict[label].append(xml_file)
    # split every category to train and val
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []
    for paths in label_dict.values():
        num = len(paths)
        val_number = int(val_rate * num)
        random.shuffle(paths)
        for i, path in enumerate(paths):
            if i < val_number:
                val_labels.append(path)
                val_images.append(path.replace('.xml', '.bmp'))
            else:
                train_labels.append(path)
                train_images.append(path.replace('.xml', '.bmp'))
    # copy image and label to train_dir
    print(f"copy train files:")
    copy(train_images, train_image_path)
    copy(train_labels, train_label_path)
    # copy image and label to val_dir
    print(f"copy val files:")
    copy(val_images, val_image_path)
    copy(val_labels, val_label_path)

    # voc label to yolo label
    # train
    voc2yolo(
        dirpath=train_label_path,
        newdir=train_label_path,
        label_txt_file=label_txt_file
    )
    # validation
    voc2yolo(
        dirpath=val_label_path,
        newdir=val_label_path,
        label_txt_file=label_txt_file
    )

if __name__ == '__main__':
    main()