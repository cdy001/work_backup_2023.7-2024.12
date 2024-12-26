import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

root_dir = r'/media/data/code_and_dataset/wafer/0945U/1030/K02F1361V_18Z'
save_dir = os.path.join(root_dir, 'bmp_files')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
img_paths = os.listdir(root_dir)
for img_name in img_paths:
    if img_name.endswith('.raw'):
        img_path = os.path.join(root_dir, img_name)
        print(img_path)
        img = np.fromfile(img_path, dtype="uint8")
        if img.size == 5120 * 5120:
            img = img.reshape(5120, 5120, 1)
        cv2.imwrite(os.path.join(save_dir, img_name.replace('.raw', '.bmp')), img)
        # print(img.shape)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show()