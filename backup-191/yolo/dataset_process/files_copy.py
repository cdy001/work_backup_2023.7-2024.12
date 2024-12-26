import shutil
import os
import glob
from tqdm import tqdm

# copy files to save_path
def copy(path_list, save_path):
    '''
    args:
    return:
    '''
    for path in tqdm(path_list):
        if os.path.isfile(path):
            try:
                shutil.copy(path, save_path)
            except:
                continue

if __name__ == '__main__':
    root_path = "/var/cdy_data/0945W"
    dataset_save_path = "dataset"
    dataset_images = os.path.join(dataset_save_path, "images")
    dataset_labels = os.path.join(dataset_save_path, "yolo_labels")
    if not os.path.exists(dataset_images):
        os.makedirs(dataset_images)
    if not os.path.exists(dataset_labels):
        os.makedirs(dataset_labels)
    defect_file_paths = os.listdir(root_path)
    images = []
    labels = []
    for defect_file_path in defect_file_paths:
        xml_paths = glob.glob(os.path.join(root_path, defect_file_path, "*.xml"))
        print(f"{defect_file_path}: {len(xml_paths)}")
        for xml_path in xml_paths:
            images.append(xml_path.replace(".xml", ".bmp"))
            labels.append(xml_path)
    # copy(images, dataset_images)
    # copy(labels, dataset_labels)