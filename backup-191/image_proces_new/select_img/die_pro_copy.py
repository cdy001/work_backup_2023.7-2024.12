import os
import shutil
import sys
import numpy as np
import tensorflow as tf
import time

print(os.getcwd())
sys.path.append(os.getcwd())

from cut.config.config import get_config
from predict_image import predict


# 加载图片
def load_test(file_path, img_height, img_width, batch_size):
    def process_path(path_label):
        img = tf.io.read_file(path_label)
        img = tf.io.decode_bmp(img, channels=0)
        img = tf.image.resize(img, [img_height, img_width])

        path = path_label

        return img, path

    list_pa = tf.data.Dataset.list_files(file_path, shuffle=False)
    image_label = list_pa.map(
        process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_batch = image_label.batch(batch_size=batch_size).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE
    )

    return test_batch


# 标签0的概率最大， 且概率低于某一阈值
# 08HG:0.7 32BB 35EB-Q:0.8
def judge_pro(pros, paths):
    path_list = []
    # i=0
    for pro, path in zip(pros, paths):
        label = np.argmax(pro)
        if pro[0] < 0.8 and label == 0:
            path = path.decode()
            print(path)
            path_list.append(path)

    return path_list


# 预测
def test_model(model, test_dataset):
    pro_list = []
    path_list = []
    i = 0
    for x_batch, y_batch in test_dataset:
        i = i + 1
        print(i)
        pred_pro = model(x_batch, training=False)

        pro_list.extend(pred_pro.numpy())
        path_list.extend(y_batch.numpy())

    return pro_list, path_list


# 复制文件
def copy(path_list, save_path):
    for path in path_list:
        shutil.copy(path, save_path)
        # shutil.move(path, save_path)


# main
def pre_copy_img():
    # die路径
    file_path = r"/media/data/code_and_dataset/wafer/*die/*/*bmp"
    recipe = "0945W"
    _, _, model_info = get_config(recipe)
    img_height = model_info["resize_y"]
    img_width = model_info["resize_x"]
    batch_size = 128

    model_path = r"models_label/290_epoch.h5"
    gpu = "0"

    save_path = r"/media/data/code_and_dataset/wafer/result"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 加载模型
    model = predict.load_model(model_path, gpu)

    test_batch = load_test(file_path, img_height, img_width, batch_size)

    # 测试
    pro_list, path_list = test_model(model, test_batch)

    # 处理概率
    spe_path_list = judge_pro(pro_list, path_list)
    
    print(f"number of image: {len(spe_path_list)}")

    # 保存图片
    copy(spe_path_list, save_path)


if __name__ == "__main__":
    pre_copy_img()
