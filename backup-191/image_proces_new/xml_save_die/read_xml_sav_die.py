# coding=utf-8
import glob
import os
import sys
import cv2 as cv
import numpy as np

print(os.getcwd())
sys.path.append(os.getcwd())

from cut.config.config import get_config
from predict_image import predict
from xml_save_die import read_xml, die_vs_die


# read xml save die
def read_xml_save_die(xml_image_path, save_path, img_type, recipe):
    image_info, _, _ = get_config(recipe)

    # 查找单个文件�???所有xml文件
    xml_paths = glob.glob(os.path.join(xml_image_path, '*xml'))
    for xml_path in xml_paths:
        if "bmp" == img_type:
            img_path, _ = os.path.splitext(xml_path)
            base_path, img_name = os.path.split(xml_path)
            path = img_path[:-2] + image_info['refer_light'] + f'.{img_type}'
            # path = os.path.join(base_path, image_info['refer_light'] + "_" + img_path.split("_")[-1] + f'.{img_type}')
            image_path = img_path + f'.{img_type}'
            image_type = img_path[-2:]
            # image_type = img_name.split("_")[0]
            img = cv.imread(image_path, 0)
        
        elif "raw" == img_type:
            img_path, img_name = os.path.split(xml_path)
            img_name_1, img_name_2 = img_name.split("_")
            img_name_2 = os.path.splitext(img_name_2)[0] + f".{img_type}"
            path = os.path.join(img_path, image_info['refer_light'] + "_" + img_name_2)

            image_path = os.path.join(img_path, img_name_1 + "_" + img_name_2)
            img = np.fromfile(image_path, dtype="uint8")
            if img.size == 5120 * 5120:
                img = img.reshape(5120, 5120, 1)

            image_type = img_name_1


        # 读取die
        die_name_l, _ = predict.cut_die_single_img(path, recipe)

        label_cut_die = 0
        defect_list, xy_list = read_xml.read_xml(xml_path)
        print(f'number of label in xml: {len(defect_list)}')
        for defect, xy in zip(defect_list, xy_list):
            dst_path = os.path.join(save_path, defect)
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            xmin1, ymin1, xmax1, ymax1 = xy

            thresh = 0.8
            label_cut_die = die_vs_die.compare_die(xmin1, ymin1, xmax1, ymax1, die_name_l, image_type, img, dst_path,
                                                   thresh,
                                                   label_cut_die)

        print('number of label of cut :', label_cut_die)


def read_xml_save_die_main():
    # S-08HGAUD-C
    # S-24ABHUD-E
    # S-32BBMUD-K
    # S-34VBMUD-G
    # S-35EBMUD-Q
    recipe = 'BWB1B29A'
    img_type = "bmp"
    folder_path = os.path.join('/var/cdy_data/aoyang/wafer/1B29B_Mark')
    for file_path in glob.glob(folder_path):
        print(f'xml_file_root_path: {file_path}')

        xml_image_path = file_path
        save_path = file_path + '_die'
        read_xml_save_die(xml_image_path, save_path, img_type, recipe)


if __name__ == '__main__':
    read_xml_save_die_main()
