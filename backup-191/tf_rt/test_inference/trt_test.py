import os
import sys
sys.path.append(os.getcwd())
import time

from test_inference.data_process import data_preprocess
from test_inference.trt_predict import trtPredict
from test_inference.tf_predict import tfPredict

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def main():
    trt_model_path = "model/1Q42E-D_0702_b24.trt"
    tf_model_path = "model/1Q42E-D_0702.h5"
    img_path = "test_img/1Q42E/R02C00_0_L1#3743_3994_4437_4682_0_0_0_0.bmp"
    # img_shape = (714, 714)  # (height, width)
    img_shape = (682, 682)  # 1Q42E
    batch_size = 24
    input_batch = data_preprocess(
        img_path=img_path,
        img_shape=img_shape,
        batch_size=batch_size
    )
    time_all = 0
    for i in range(100):
        start_time = time.time()
        label = trtPredict(
            model_path=trt_model_path,
            input_batch=input_batch
        )
        # label = tfPredict(
        #     model_path=tf_model_path,
        #     input_batch=input_batch
        # )    
        end_time = time.time()
        time_all += end_time - start_time
        # print("推理时间：%.4fs"%(end_time - start_time))
        print(label)
    print("mean taking time:{}s".format(round(time_all/100, 4)))

if __name__ == '__main__':
    main()