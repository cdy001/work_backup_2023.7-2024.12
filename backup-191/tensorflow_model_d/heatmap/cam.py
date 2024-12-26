import os
import sys
import glob
sys.path.append(os.getcwd())
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from heatmap.cam_trt_runtime import TtrRuntime


class ModelWithConvOut(keras.Model):
    def __init__(self, model_path=None, conv_layer_name='top_conv'):
        super(ModelWithConvOut, self).__init__()
        self.original_model = tf.keras.models.load_model(model_path, compile=False)
        self.conv_layer_name = conv_layer_name
        self.model = tf.keras.Model(
            inputs=self.original_model.input,
            outputs=[
                self.original_model.get_layer(self.conv_layer_name).output,
                self.original_model.output
            ]
        )
        self.input_spec = self.original_model.input_spec
        
    def __call__(self, inputs):

        return self.model(inputs)


def get_gradcam(model, img_array, class_idx=None):
    class_idxes = []
    class_outputs = []        
    heatmaps = []
    with tf.GradientTape() as tape:
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        tape.watch(img_array)
        conv_outputs, predictions = model(img_array)
        for i, prediction in enumerate(predictions):
            class_idx_i = np.argmax(prediction) if not class_idx else class_idx
            class_idxes.append(class_idx_i)
            class_outputs.append(prediction[class_idx_i])            
    grads = tape.gradient(class_outputs, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # 对所有空间位置平均池化
    # heatmaps
    for i, conv_output in enumerate(conv_outputs):
        conv_output = conv_output @ pooled_grads[i][..., tf.newaxis]  # 点乘，channel维度梯度加权
        heatmap = tf.reduce_sum(conv_output, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # ReLU激活
        heatmap = heatmap / (np.max(heatmap) - np.min(heatmap))  # 归一化
        heatmaps.append(heatmap)

    return heatmaps, class_idxes, class_outputs


def cam(model, img_arrays, weights_file, class_idx=None):
    class_idxes = []
    class_outputs = []        
    heatmaps = []
    conv_outputs, predictions = model(img_arrays)
    json_result = read_json_file(weights_file)
    fc_weights = np.array(json_result)
    for i, prediction in enumerate(predictions):
        class_idx_i = np.argmax(prediction) if not class_idx else class_idx
        class_idxes.append(class_idx_i)
        class_outputs.append(prediction[class_idx_i])
        
        # 对卷积层输出进行加权
        weights_for_class = fc_weights[:, class_idx_i]
        weighted_sum = np.dot(conv_outputs[i], weights_for_class)
        heatmap = np.maximum(weighted_sum, 0)  # ReLU激活
        heatmap = heatmap / np.max(heatmap)  # 归一化

        heatmaps.append(heatmap)
    
    return heatmaps, class_idxes, class_outputs


def heatmaps_sum_all(heatmaps, img_path, alpha=0.4):
    heatmaps_stack = np.stack(heatmaps, axis=0)
    heatmaps_sum = np.sum(heatmaps_stack, axis=0)
    heatmaps_sum /= (np.max(heatmaps_sum) - np.min(heatmaps_sum))  # 归一化
    # Rescale heatmap to a range 0-255
    heatmaps_sum = np.uint8(255 * heatmaps_sum)
    # Use jet colormap to colorize heatmap
    jet = plt.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmaps_sum]
    # Create an image with RGB colorized heatmap
    img = cv2.imread(img_path, flags=0)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_bgr.shape[1], img_bgr.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
    # Superimpose重叠 the heatmap on original image        
    superimposed_img = jet_heatmap * alpha + img_bgr
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    # Save the superimposed image
    root_path = os.path.dirname(img_paths[0])
    cam_save_path = os.path.join(root_path, "cam_result")
    check_or_makedir(cam_save_path)
    cam_path = os.path.join(cam_save_path, "all_result.bmp")
    superimposed_img.save(cam_path)

    return


def main(model, img_paths, batch_size=1, alpha=0.4, save=True, weights_file=None):
    heatmaps_all, class_idxes_all, class_outputs_all = [], [], []
    for j in range(0, len(img_paths), batch_size):
        img_arrays = []
        # 加载并预处理输入图像
        shape = model.input_spec[0].shape[1:3]  # (h, w)
        for img_path in img_paths[j:j+batch_size]:
            img = cv2.imread(img_path, flags=0)
            img_array = cv2.resize(img, (shape[1], shape[0]))
            if len(img_array.shape) == 2:
                img_array = np.expand_dims(img_array, axis=-1)
            img_arrays.append(img_array)
        img_arrays = np.array(img_arrays).astype(np.float32)

        # Grad-CAM        
        heatmaps, class_idxes, class_outputs = get_gradcam(model, img_arrays)
        # CAM
        # heatmaps, class_idxes, class_outputs = cam(model, img_arrays, weights_file)

        if save:
            root_path = os.path.dirname(img_paths[0])
            cam_save_path = os.path.join(root_path, "cam_result")
            check_or_makedir(cam_save_path)
            for i, heatmap in enumerate(heatmaps):
                img_path = img_paths[j+i]
                # Rescale heatmap to a range 0-255
                heatmap = np.uint8(255 * heatmap)
                # Use jet colormap to colorize heatmap
                jet = plt.get_cmap("jet")
                # Use RGB values of the colormap
                jet_colors = jet(np.arange(256))[:, :3]
                jet_heatmap = jet_colors[heatmap]
                # Create an image with RGB colorized heatmap
                img = cv2.imread(img_path, flags=0)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
                jet_heatmap = jet_heatmap.resize((img_bgr.shape[1], img_bgr.shape[0]))
                jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)
                # Superimpose重叠 the heatmap on original image        
                superimposed_img = jet_heatmap * alpha + img_bgr
                superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
                # Save the superimposed image
                name = os.path.basename(img_path)
                cam_path = os.path.join(cam_save_path, name)
                superimposed_img.save(cam_path)

                print(f"img_path:{img_path}, class_idx:{class_idxes[i]}, class_output:{class_outputs[i]}")
        
        heatmaps_all.extend(heatmaps)
        class_idxes_all.extend(class_idxes)
        class_outputs_all.extend(class_outputs)

    return heatmaps_all, class_idxes_all, class_outputs_all


def saveConvModel(original_model_path, save_path, last_conv_layer_name='top_conv'):
    original_model = load_model(original_model_path)
    check_or_makedir(save_path)
    grad_model = tf.keras.Model(
        inputs=original_model.input,
        outputs=[
            original_model.get_layer(last_conv_layer_name).output,
            original_model.output
            ]
            )
    original_model_name = os.path.basename(original_model_path)
    model_name = "cam_" + original_model_name
    grad_model.save(os.path.join(save_path, model_name), overwrite=True)


def saveWeights_CAM(original_model_path, save_path):
    original_model = load_model(original_model_path)
    check_or_makedir(save_path)
    fc_weights = original_model.layers[-1].weights[0]
    fc_weights = fc_weights.numpy().tolist()
    original_model_name = os.path.basename(original_model_path)
    weight_file_name = "cam_" + original_model_name.replace(".h5", ".json")
    with open(os.path.join(save_path, weight_file_name), "w") as f:
        json.dump(fc_weights, f)


def init_gpu_env(gpu_index):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def load_model(model_path):  
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def check_or_makedir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.system(f"chmod -R 777 {save_path}")


def read_json_file(weights_file):
    with open(weights_file, "r") as f:
        result = json.load(f)
    return result


if __name__ == "__main__":
    model_path = "heatmap/models/1X53B_0701.h5"
    weights_file = "heatmap/cam_models/cam_1X53B_0701.json"
    img_root_path = '/var/cdy_data/aoyang/data/BGB1X53B/Mesa_chipping'
    img_paths = glob.glob(os.path.join(img_root_path, "*.bmp"))

    # 1. 初始化GPU
    init_gpu_env(2)
    # 2. 加载模型
    model = ModelWithConvOut(model_path)
    # model = load_model(model_path="heatmap/cam_models/new_1X53B_0701.h5")
    # model = TtrRuntime("heatmap/cam_models/new_1X53B_0701_b52.trt")
    # 3. 生成热力图
    heatmaps, class_idxs, class_outputs = main(
        model=model,
        img_paths=img_paths,
        batch_size=4,
        weights_file=weights_file
        )
    
    heatmaps_sum_all(heatmaps, "/var/cdy_data/aoyang/data/BGB1X53B/0613/Good/R00C03_0_L1#4167_3371_5027_4300_0_0_0_0.bmp")

    # save grad_model
    # saveConvModel(model_path, "heatmap/grad_models")
    # saveWeights_CAM(model_path, "heatmap/grad_models")
        