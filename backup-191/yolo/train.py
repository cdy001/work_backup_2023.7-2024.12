import os
import sys
sys.path.append(os.getcwd())

import torch

from ultralytics import YOLO, settings, YOLOv10
from ultralytics.utils.torch_utils import intersect_dicts


def fine_tune(model, data, pretrained=True, device=[0]):
    # load a model
    model = YOLO(model=model)
    # train the model
    model.train(
        data=data,  # dataset
        batch=16,  # batch_size
        device=device,  # device 0 or [0, 1] or 'cpu'
        epochs=300,
        imgsz=640,  # img_size
        amp=False,  # Automatic Mixed Precision (AMP) training
        optimizer='AdamW',
        lr0=1e-3,
        lrf=1e-4,
        cos_lr=True,
        multi_scale=True,  # Whether to use multiscale during training
        # freeze=11,  # freeze first n layers
        pretrained=pretrained,
    )

def main():
    # Update a settins
    # settings.update({
    #     # 'datasets_dir': '/var/cdy_data/0945W_train_val',
    #     'weights_dir': 'weights',
    #     'save_dir': 'tmp'
    #     })
    print(settings)
    # Load a model
    # model = YOLO(model='configs/models/yolov8s.yaml')
    # model.load(weights='models_and_labels/0945W.pt')
    model_1 = YOLOv10(model='configs/models/yolov10s-aoyang.yaml')
    # model.load(weights='runs/detect/train-official-pretrain/weights/best.pt')
    # model = YOLO(model='configs/models/yolov10x-aoyang.yaml').load(weights='pretrained/yolov10x.pt')
    
    sd = torch.load("runs/detect/train-official-pretrain/weights/best.pt", map_location="cpu")["model"].state_dict()
    # 只保留匹配的参数
    sd = {k: v for k, v in sd.items() if k in model_1.model.state_dict() and v.size() == model_1.model.state_dict()[k].size()}
    # csd = intersect_dicts(sd, model.model.state_dict())
    model_1.model.load_state_dict(sd , strict=False)

    model = YOLOv10(model='configs/models/yolov10s-aoyang.yaml').load(weights=model_1)
    # print(f"Transferred {len(csd)}/{len(model.state_dict())} items from pretrained weights")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(name)
    # for key, val in model.state_dict().items():
    #     print(key, val)

    # # sd = torch.load("self_supervised_pretrain/models/yolov10s_epoch250.pt", map_location="cpu")
    
    # print(f"Transferred {len(csd)}/{len(model.model.state_dict())} items from pretrained weights")

    # # train the model
    # model.train(
    #     data='configs/datasets/aoyang.yaml',  # dataset
    #     batch=12,  # batch_size
    #     device=[1],  # device 0 or [0, 1] or 'cpu'
    #     epochs=300,
    #     imgsz=640,  # img_size
    #     amp=False,  # Automatic Mixed Precision (AMP) training
    #     optimizer='AdamW',
    #     lr0=1e-3,
    #     lrf=1e-4,
    #     multi_scale=True,  # Whether to use multiscale during training
    #     cos_lr=True,
    #     # freeze=11,  # freeze first n layers
    #     pretrained=True,
    #     name='train-custom-pretrain',
    # )

    

if __name__ == '__main__':
    main()

    # fine_tune(
    #     model='configs/models/yolov10s-aoyang.yaml',
    #     data='configs/datasets/aoyang.yaml',
    #     pretrained='pretrained/yolov10s.pt',
    #     device=[0]
    # )