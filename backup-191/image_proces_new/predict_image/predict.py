import os
import numpy as np
import tensorflow as tf
import sys
import time

print(os.getcwd())
sys.path.append(os.getcwd())

from cut.config.config import get_config, get_config_new
from cut.cut_die import cut_die_single
from predict_image.tf_runtime import TfRuntime
from predict_image.trt_runtime import TtrRuntime

def cut_die_single_img(path, recipe):
    die_para = {}
    binary_para = {}
    model_para = {}
    # image_info, binary_info, model_info = get_config(recipe)
    image_info, binary_info, model_info = get_config_new(recipe)
    die_para["refer_light"] = image_info.get("refer_light")
    die_para["die_height"] = image_info.get("die_height")
    die_para["die_width"] = image_info.get("die_width")
    die_para["margin_y"] = image_info.get("margin_y")
    die_para["margin_x"] = image_info.get("margin_x")
    die_para["L1_offset"] = image_info.get("L1_offset")
    binary_para["threshold"] = binary_info.get("threshold")
    binary_para["binary_type"] = binary_info.get("binary_type")
    binary_para["open_close_type"] = binary_info.get("open_close_type")
    binary_para["struct_element_tuple"] = binary_info.get("struct_element_tuple")
    model_para["resize_y"] = model_info.get("resize_y")
    model_para["resize_x"] = model_info.get("resize_x")

    die_names, die_mats = cut_die_single(
        path,
        light_list=image_info["light_list"],
        die_para=die_para,
        binary_para=binary_para,
        model_para=model_para,
        recipe=recipe,
    )

    return die_names, die_mats

def load_model(model_path, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    model = tf.keras.models.load_model(model_path)

    return model

def test_predict(model, die_mat, die_name, total_name, total_pre, batch_size):
    images_names = tf.data.Dataset.from_tensor_slices(
        (np.array(die_mat), np.array(die_name))
    )
    images_names = images_names.batch(batch_size)

    for x_batch, n_batch in images_names:
        pred_pro = model(x_batch, training=False)
        pred_label = tf.math.argmax(pred_pro, axis=1)

        total_pre.extend(pred_label.numpy())
        total_name.extend(n_batch.numpy())
        # x1, y1, x2, y2, *_ = os.path.splitext(n_batch.numpy()[0].decode())[0].split('#')[-1].split('_')
        # if 2091-60<int(x1)<2091+60 and 100-40<int(y1)<100+40:
        #     s=0

def test_patch_predict(model_path, gpu, total_paths, total_name, total_pre, recipe):
    # _, _, model_info = get_config(recipe)
    _, _, model_info = get_config_new(recipe)
    batch_size = model_info.get("batch_size")
    model = load_model(model_path, gpu)

    for path_p in total_paths:
        # 迭代图像路径
        die_name = []
        die_mat = []
        for path in path_p:
            print(path)
            try:
                die_names, die_mats = cut_die_single_img(path, recipe)
            except:
                continue
            die_name.extend(die_names)
            die_mat.extend(die_mats)

        test_predict(model, die_mat, die_name, total_name, total_pre, batch_size)


class WaferPredict:
    def __init__(self, model_path, gpu, total_paths, total_name, total_pre, top_values, top_labels, recipe):
        self.model_path = model_path
        self.gpu = gpu
        self.total_paths = total_paths
        self.total_name = total_name
        self.total_pre = total_pre
        self.top_values = top_values
        self.top_labels = top_labels
        self.recipe = recipe
        self.batch_size = get_config_new(recipe)[-1].get("batch_size")

    def _cut_die_single_img(self, path):
        die_para = {}
        binary_para = {}
        model_para = {}
        image_info, binary_info, model_info = get_config_new(self.recipe)
        die_para["refer_light"] = image_info.get("refer_light")
        die_para["die_height"] = image_info.get("die_height")
        die_para["die_width"] = image_info.get("die_width")
        die_para["margin_y"] = image_info.get("margin_y")
        die_para["margin_x"] = image_info.get("margin_x")
        die_para["L1_offset"] = image_info.get("L1_offset")
        binary_para["threshold"] = binary_info.get("threshold")
        binary_para["binary_type"] = binary_info.get("binary_type")
        binary_para["open_close_type"] = binary_info.get("open_close_type")
        binary_para["struct_element_tuple"] = binary_info.get("struct_element_tuple")
        model_para["resize_y"] = model_info.get("resize_y")
        model_para["resize_x"] = model_info.get("resize_x")

        die_names, die_mats = cut_die_single(
            path,
            light_list=image_info["light_list"],
            die_para=die_para,
            binary_para=binary_para,
            model_para=model_para,
            recipe=self.recipe,
        )

        return die_names, die_mats

    def _cut_one_patch(self, path_p):
        time_start = time.time()
        # 迭代图像路径
        die_name_all = []
        die_mat_all = []
        for path in path_p:
            print(path)
            try:
                die_names, die_mats = self._cut_die_single_img(path)
            except :
                import traceback
                print(traceback.format_exc())
                continue
            die_name_all.extend(die_names)
            die_mat_all.extend(die_mats)
        time_end = time.time()
        time_cost = time_end - time_start
        
        return die_name_all, die_mat_all, time_cost

    def _predict_batch(self, model, die_mat, die_name):
        time_inference = 0
        self.total_name.extend(die_name)
        for i in range(0, len(die_mat), self.batch_size):
            if i + self.batch_size < len(die_mat):
                batch = die_mat[i:i+self.batch_size]
            else:
                batch = die_mat[i:]
            time_inference_start = time.time()
            data = model.predict(batch)
            time_inference_end = time.time()
            time_inference += (time_inference_end - time_inference_start)
            print(f"predict {len(batch)} dies, cost {round(time_inference_end - time_inference_start, 4)}s")
            self.total_pre.extend(data.get("indices")[:, 0])
            self.top_values.extend(data.get("values"))
            self.top_labels.extend(data.get("indices"))
        
        return time_inference
    
    def test_patch_predict(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)
        gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if self.model_path.endswith(".h5"):
            model = TfRuntime(self.model_path)
        elif self.model_path.endswith('.trt'):
            model = TtrRuntime(self.model_path)
        else:
            print(f"{self.model_path} is not a valid model")
            exit()
        time_inference = 0
        time_cut = 0
        for path_p in self.total_paths:
            die_name, die_mat, time_cut_patch = self._cut_one_patch(path_p)
            time_inference_patch = self._predict_batch(model, die_mat, die_name)
            time_cut += time_cut_patch
            time_inference += time_inference_patch

        return time_cut, time_inference