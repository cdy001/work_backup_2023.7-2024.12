import os
import sys
sys.path.append(os.getcwd())
import glob
import time

from predict_dies.data_process import read_images, image_batch
from tf2trt.test_inference.trt_predict import trtPredict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    
    trt_model_path = "models_label/1Q42E-D_0726_b24.trt"
    root_path = "/var/cdy_data/aoyang/data/BGB1Q42E-D"
    img_paths = glob.glob(os.path.join(root_path, "*/*/*.bmp"))
    batch_paths = image_batch(img_paths, 24)
    total_images = []
    total_predictions = []
    for batch_path in batch_paths:
        time_start = time.time()
        img_batch = read_images(batch_path, img_shape=(682, 682))
        time_end = time.time()
        print(f"{img_batch.shape}, time cost: {time_end - time_start:.4f}")
        start_time = time.time()
        label, pros = trtPredict(
            model_path=trt_model_path,
            input_batch=img_batch
            )
        end_time = time.time()
        print(f"{label}, time cost: {end_time - start_time: .4f}")
        total_images.extend(batch_path)
        total_predictions.extend(label)
    
    with open("predict_dies/result.txt", mode="w") as f:
        for image_path, label in zip(total_images, total_predictions):
            try:
                f.write(f"{image_path},{label}\n")
            except Exception as e:
                print(image_path)
                print(e)
                continue