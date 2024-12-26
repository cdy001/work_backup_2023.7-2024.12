import os
import sys
sys.path.append(os.getcwd())
import glob
import shutil
from collections import defaultdict


if __name__ == "__main__":
    img_paths = "/var/cdy_data/aoyang/counter_img/test-0924/*.JPG"
    display_wafer_root_path = "/var/cdy_data/aoyang/counter_img/display"
    
    wafer_dict = defaultdict()
    for img_path in glob.glob(img_paths):
        img_name = os.path.basename(img_path)
        wafer_id = img_name.split("_")[0]
        if wafer_id not in wafer_dict:
            wafer_dict[wafer_id] = [img_path, img_name]
    
    display_wafer_id = []
    sub_dictorys = os.listdir(display_wafer_root_path)
    sub_dictorys = [name for name in sub_dictorys if name != "result"]
    for sub_dictory in sub_dictorys:
        wafer_ids = os.listdir(os.path.join(display_wafer_root_path, sub_dictory))
        display_wafer_id.extend(wafer_ids)
    print(len(display_wafer_id), display_wafer_id)
    for wafer_id in display_wafer_id:
        shutil.copy(wafer_dict[wafer_id][0], os.path.join(display_wafer_root_path, wafer_dict[wafer_id][1]))