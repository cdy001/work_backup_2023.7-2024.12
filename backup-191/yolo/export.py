import os
import sys
sys.path.append(os.getcwd())
from ultralytics import YOLO, settings

def main():
    print(settings)
    # model = YOLO(model="configs/models/yolov10s-aoyang.yaml").load(weights="runs/detect/train-official-pretrain/weights/best.pt")
    model = YOLO(model="runs/detect/train-official-pretrain/weights/best.pt")
    print(model.device)

    # export the model
    model.export(
        # format="onnx",
        format="engine",
        dynamic=True,
        batch=8,
        workspace=1,
        half=True,  # fp16
    )

if __name__ == '__main__':
    main()