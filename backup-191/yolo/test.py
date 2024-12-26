from ultralytics import YOLO
import os
import sys
sys.path.append(os.getcwd())
from PIL import Image
from torch.utils.data import DataLoader
import shutil
from tqdm import tqdm

from utils.dataset import ImageDataset
from utils.utils_func import read_label_file, check_or_mkdirs

model_config = 'configs/models/yolov10s-aoyang.yaml'
model_path = 'runs/detect/train-custom-pretrain-fintune/weights/best.pt'
img_root_path = '/var/cdy_data/aoyang/data/BGB1Q42E-D/0729/Luminous_ITO'
# image_size = (160, 640)  # (h, w)
image_size = (640, 640)  # 1Q42E
device = 1  # 0 or 1 or 'cpu'

save_path_root = '/var/cdy_data/obj_det_results/0809/1Q42E-Luminous_ITO'
label_file = 'models_and_labels/aoang.txt'

def process_results(results):
    '''
    args:
        results: detection model predict results
    return:
        paths: path of every result
        ims: label image of every result
        labels: label of every result
    '''
    labels = []
    ims = []
    paths = []
    for result in results:
        im = Image.fromarray(result.plot())
        path = result.path
        boxes = result.boxes
        if not boxes:  # no detect results
            label = 0
        else:
            label_list = list(map(int, boxes.cls))
            if 1 in label_list:
                label = 1
                conf = boxes.conf[label_list.index(1)]
            else:
                label = int(boxes.data[0][-1])
        ims.append(im)
        paths.append(path)
        labels.append(label)
    return paths, ims, labels

def main():
    # Load a model
    # model = YOLO(model=model_path)
    model = YOLO(model=model_config).load(weights=model_path)
    # read label file
    _, label_path_dict = read_label_file(label_file)
    
    # process data for every batch
    dataset = ImageDataset(directory=img_root_path)
    data_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)

    # predict
    print("predicting...")
    img_paths = []
    img_ims = []
    img_labels = []
    for batch in tqdm(data_loader):
        results = model.predict(
            source=batch,
            imgsz=image_size,
            device=device,
            # device='cpu',
            # conf=0.1,  # confidence threshold
            )
        paths, ims, labels = process_results(results)
        img_paths.extend(paths)
        img_ims.extend(ims)
        img_labels.extend(labels)
    
    # save results
    print("results saving...")
    for i, path in tqdm(enumerate(img_paths)):
        img_name = os.path.split(path)[-1]
        label = img_labels[i]
        detect_img = img_ims[i]
        save_path = os.path.join(save_path_root, label_path_dict[label])
        check_or_mkdirs(save_path)
        if label == 0:
            shutil.copy(path, save_path)
        else:
            detect_img.save(os.path.join(save_path, img_name))
        # shutil.copy(path, save_path)
    print("Successful!")

        

if __name__ == '__main__':
    main()