import glob
import os
import random


# 随机删除部分图片
def random_remove_img():
    img_paths = glob.glob(r'/var/cdy_data/aoyang/data/BGB1X53B/0304/Good/*bmp')
    random.shuffle(img_paths)
    rm_img_paths = img_paths[:500]
    for rm_img_path in rm_img_paths:
        os.remove(rm_img_path)

if __name__ == '__main__':
    random_remove_img()

