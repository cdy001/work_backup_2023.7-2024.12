from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

model_path = 'models_and_labels/1Q42E-0729.pt'
img_path = 'test_imgs/1Q42E/R00C03_6_L1_1721609030.bmp'
# Load a model
model = YOLO(model="yolov10s-aoyang.engine", task="detect")
# model = YOLO(model='configs/models/yolov10s-aoyang.yaml').load(weights=model_path)
# read img
img = cv2.imread(img_path, 0)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
# img = np.array(img)
results = model.predict(
    source=[img],
    # imgsz=(160, 640),  # image_size (h, w)
    imgsz=(640, 640),  # 1Q42E
    device=[0, ],
    )
im = Image.fromarray(results[0].plot(line_width=1, font_size=0.05, labels=True, conf=True))
im.save('result.bmp')
print(results[0].boxes)