import sys
import os
sys.path.append(os.getcwd())
import glob
import random

from utils.utils_func import check_or_mkdirs, copy
from voc2yolo import voc2yolo

data_root_path = '/var/cdy_data/obj_det/BGB1Q42E-D'
save_path = '/var/cdy_data/obj_det/BGB1Q42E-D_train_val_v2'
label_txt_file = 'models_and_labels/aoang.txt'
val_rate = 0.5

def dataset_path_create(save_path):
    train_image_path = os.path.join(save_path, 'train', 'images')
    train_label_path = os.path.join(save_path, 'train', 'labels')
    val_image_path = os.path.join(save_path, 'val', 'images')
    val_label_path = os.path.join(save_path, 'val', 'labels')
    check_or_mkdirs(train_image_path)
    check_or_mkdirs(val_image_path)
    check_or_mkdirs(train_label_path)
    check_or_mkdirs(val_label_path)
    return train_image_path, train_label_path, val_image_path, val_label_path

def main():
    train_image_path, train_label_path, val_image_path, val_label_path = dataset_path_create(save_path)
    train_images = []
    train_labels = []
    val_images = []
    val_labels = []

    defect_file_paths = os.listdir(data_root_path)
    defect_file_paths.sort()
    for defect_file_path in defect_file_paths:
        xml_paths = glob.glob(os.path.join(data_root_path, defect_file_path, "*.xml"))
        val_number = int(val_rate * len(xml_paths))
        random.shuffle(xml_paths)
        for i, xml_path in enumerate(xml_paths):
            if i < val_number:
                val_images.append(xml_path.replace(".xml", ".bmp"))
                val_labels.append(xml_path)
            else:
                train_images.append(xml_path.replace(".xml", ".bmp"))
                train_labels.append(xml_path)

        print(f"{defect_file_path}: {len(xml_paths)}")
    
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