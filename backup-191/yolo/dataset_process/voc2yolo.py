# coding=gbk
import os
import xml.etree.ElementTree as ET
from decimal import Decimal
import json
 
def voc2yolo(dirpath, newdir, label_txt_file):
    # dirpath = 'dataset/labels'  # xml文件目录
    # newdir = 'dataset/yolo_labels'  # txt文件存放目录

    # label_txt_file = 'dataset/0945.txt'
    with open(label_txt_file) as f:
        label_dict = json.load(f)
    
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    
    for fp in os.listdir(dirpath):
        print(os.path.join(dirpath, fp))
    
        root = ET.parse(os.path.join(dirpath, fp)).getroot()
    
        xmin, ymin, xmax, ymax = 0, 0, 0, 0
        sz = root.find('size')
        width = float(sz[0].text)
        height = float(sz[1].text)
        filename = root.find('filename').text
        # print(fp)
        with open(os.path.join(newdir, fp.split('.')[0] + '.txt'), 'a+') as f:
            for child in root.findall('object'):  # 找到所有标注目标
                sub = child.find('bndbox')  # 标注框
                sub_label = child.find('name')
                xmin = int(sub[0].text)
                ymin = int(sub[1].text)
                xmax = int(sub[2].text)
                ymax = int(sub[3].text)
                try:  # 归一化
                    x_center = Decimal(str(round(float((xmin + xmax) / (2 * width)),6))).quantize(Decimal('0.000000'))
                    y_center = Decimal(str(round(float((ymin + ymax) / (2 * height)),6))).quantize(Decimal('0.000000'))
                    w = Decimal(str(round(float((xmax - xmin) / width),6))).quantize(Decimal('0.000000'))
                    h = Decimal(str(round(float((ymax - ymin) / height),6))).quantize(Decimal('0.000000'))
                    # print(str(x_center)+' '+ str(y_center)+' '+str(w)+' '+str(h))
                    label = label_dict[sub_label.text]
                    f.write(' '.join([str(label), str(x_center), str(y_center), str(w), str(h) + '\n']))
                except ZeroDivisionError:
                    print(f"{filename}的width有问题")