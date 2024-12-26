import glob
import os.path
import random
import shutil


def copy_img(src_path, dest_path):
    src_path_g = os.path.join(src_path, '*')
    src_paths = glob.glob(src_path_g)
    random.shuffle(src_paths)
    src_paths = src_paths[:2000]
    for img_path in src_paths:
        shutil.copy(img_path, dest_path)


if __name__ == '__main__':
    src_paths = r'/var/weizhi/jucan/wafer/*_die/0'
    for src_path in glob.glob(src_paths):
        # dest_path = r'/var/weizhi/FMLR06BU/wafer/TMC2305B231510_20230602_140236_die/2_Good'
        dest_path = src_path + '_new'
        os.makedirs(dest_path, exist_ok=True)
        copy_img(src_path, dest_path)