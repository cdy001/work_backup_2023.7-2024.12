# import os
# import sys
# sys.path.append(os.getcwd())
from ultralytics import YOLO, settings

def main():
    # Update a settins
    print(settings)

    # Load a model
    model = YOLO(model='configs/models/yolov10s-aoyang.yaml').load(weights='pretrained/yolov10s.pt')

    for k, v in model.named_parameters():
        print(k)


if __name__ == '__main__':
    main()