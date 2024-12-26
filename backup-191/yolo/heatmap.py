import cv2
from ultralytics.solutions import Heatmap


heatmap = Heatmap(model="runs/detect/train-official-pretrain/weights/best.pt", colormap=cv2.COLORMAP_JET)
frame = cv2.imread("test_imgs/1Q42E/R00C03_6_L1_1721609030.bmp")
processed_frame = heatmap.generate_heatmap(frame)
cv2.imwrite('heatmap_result.bmp', processed_frame)