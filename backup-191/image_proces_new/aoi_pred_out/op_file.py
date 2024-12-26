import os, sys

print(os.getcwd())
sys.path.append(os.getcwd())
import glob
import shutil
from aoi_pred_out import test_predict_from_aoi


def rename_txt(out_path):
    txt_paths = glob.glob(os.path.join(out_path, '*'))
    for txt_path in txt_paths:
        base_path, hz = os.path.splitext(txt_path)
        if hz in '.TXT' or hz in '.txt':
            new_path = base_path + '_aoi.txt'
            os.rename(txt_path, new_path)


def move_txt(jt_out_path, sf_out_path):
    for j_out_ptah in glob.glob(os.path.join(jt_out_path, '*')):
        _, txt_name = os.path.split(j_out_ptah)
        file_name = txt_name[:-8]
        s_out_path = os.path.join(sf_out_path, file_name)
        if os.path.exists(os.path.join(sf_out_path, file_name)):
            shutil.copy(j_out_ptah, s_out_path)


def rename_move_txt():
    result_txt = '/data/wz/data/data/34VB/10.13/result1'
    # jt_out_path, sf_out_path文件必须
    jt_out_path = os.path.join(result_txt, 'jt_out')
    sf_out_path = os.path.join(result_txt, 'sf_out')

    rename_txt(jt_out_path)

    move_txt(jt_out_path, sf_out_path)

    test_predict_from_aoi.aoi_predict(sf_out_path)

    base_path, _ = os.path.split(result_txt)
    dest_path = os.path.join(base_path, 'result2')
    result_path = os.path.join(sf_out_path, 'result2')
    shutil.copytree(result_path, dest_path)


if __name__ == '__main__':
    rename_move_txt()
