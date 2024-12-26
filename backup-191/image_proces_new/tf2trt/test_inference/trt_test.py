import os
import sys
sys.path.append(os.getcwd())
import time

from tf2trt.test_inference.data_process import data_preprocess
from tf2trt.test_inference.trt_predict import trtPredict
from tf2trt.test_inference.tf_predict import tfPredict

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main():
    trt_model_path = "models_label/0945Y_1105_40_epoch_b536.trt"
    tf_model_path = "models_label/0945Y_1105_40_epoch.h5"
    img_path = "tf2trt/test_img/0945Y/0001_16_IMAGE3#4232_2207_4973_2381_0_0_0_0.bmp"
    # img_shape = (505, 510)  # 1B28C
    # img_shape = (500, 500)  # 1C30D
    # img_shape = (714, 714)  # (height, width) 1O42A
    # img_shape = (682, 682)  # 1Q42E
    # img_shape = (148, 95)
    # img_shape = (457, 195)  # L28E
    img_shape = (134, 588)  # 0945Y
    # img_shape = (100, 100)
    batch_size = 24
    input_batch = data_preprocess(
        img_path=img_path,
        img_shape=img_shape,
        batch_size=batch_size
    )
    time_all = 0
    for i in range(100):
        start_time = time.time()
        # label, pros = trtPredict(
        #     model_path=trt_model_path,
        #     input_batch=input_batch
        # )
        label, pros = tfPredict(
            model_path=tf_model_path,
            input_batch=input_batch
        )    
        end_time = time.time()
        time_all += end_time - start_time
        # print("推理时间：%.4fs"%(end_time - start_time))
        print(label)
        # print(pros)
    print("mean taking time:{}s".format(round(time_all/100, 4)))

if __name__ == '__main__':
    main()