import os
import sys
sys.path.append(os.getcwd())
from ultralytics import YOLOv10, settings

def main():
    print(settings)
    model = YOLOv10(model='runs/detect/train-no-pretrain/weights/best.pt')

    # train the model
    model.val(
        data="configs/datasets/aoyang.yaml",
        name="no-pretrain",
        device=[1]
        )

if __name__ == '__main__':
    main()