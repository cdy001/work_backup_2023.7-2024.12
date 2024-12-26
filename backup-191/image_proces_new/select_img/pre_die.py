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


def judge_pro(pros, paths):
    path_list = []
    label_list = []
    for pro, path in zip(pros, paths):
        label = np.argmax(pro)
        path_list.append(path)
        label_list.append(label)

    return path_list, label_list


# 预测
def test_model(model, test_dataset):
    time_inference_start = time.time()
    pro_list = []
    path_list = []
    i = 0
    for x_batch, y_batch in test_dataset:
        i = i + 1
        print(i)
        pred_pro = model(x_batch, training=False)

        pro_list.extend(pred_pro.numpy())
        path_list.extend(y_batch.numpy())
    time_inference_end = time.time()
    print("inference time: {:.4f}s".format(time_inference_end - time_inference_start))

    return pro_list, path_list


# 复制文件
def copy(path_list, label_list, save_path):
    for path, label in zip(path_list, label_list):
        path = path.decode('utf-8')
        save_path_lable = os.path.join(save_path, str(label))
        if not os.path.exists(save_path_lable):
            os.mkdir(save_path_lable)
        shutil.copy(path, save_path_lable)


# main
def pre_copy_img():
    # die路径
    recipe = "0945U"
    gpu = "0"
    model_path = "models_label/0945U_1107_40_epoch.h5"
    file_path = "/var/cdy_data/jucan/data/0945U/1107-0945U/Good/*bmp"

    _, _, model_info = get_config(recipe)
    img_height = model_info["resize_y"]
    img_width = model_info["resize_x"]
    batch_size = model_info["batch_size"]

    save_path = os.path.split(file_path)[0] + "_die"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 加载模型
    model = predict.load_model(model_path, gpu)

    test_batch = load_test(file_path, img_height, img_width, batch_size)

    # 测试
    pro_list, path_list = test_model(model, test_batch)

    # 处理概率
    path_list, label_list = judge_pro(pro_list, path_list)
    
    # 保存图片
    copy(path_list, label_list, save_path)


if __name__ == "__main__":
    pre_copy_img()