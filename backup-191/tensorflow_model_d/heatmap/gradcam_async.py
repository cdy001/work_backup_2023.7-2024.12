import asyncio
import os
import sys
import glob
sys.path.append(os.getcwd())
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from concurrent.futures import ThreadPoolExecutor

from heatmap.utils.decorator import time_cost, time_cost_async


def init_gpu_env(gpu_index):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def check_or_makedir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.system(f"chmod -R 777 {save_path}")


class ModelWithConvOut(tf.keras.Model):
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


# 后处理每个conv_map
async def process_conv_map(conv_output, pooled_grad):
    conv_output = conv_output @ pooled_grad[..., tf.newaxis]  # 点乘，channel维度梯度加权
    conv_map = tf.reduce_sum(conv_output, axis=-1)
    conv_map = np.maximum(conv_map, 0)  # ReLU激活
    conv_map = conv_map / np.max(conv_map)  # 归一化
    # conv_map = await threshold_map_max(conv_map, percentile=70)  # 应用阈值
    return conv_map


# 用线程池来执行模型推理，异步化推理过程
async def model_inference(model, img_array, class_idx=None):
    class_idxes = []
    class_outputs = [] 
    # 计算梯度并应用于特征图
    with tf.GradientTape() as tape:
        # img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        tape.watch(img_array)
        conv_outputs, predictions = model(img_array)
        # 处理预测结果和类输出
        for i, prediction in enumerate(predictions):
            class_idx_i = np.argmax(prediction) if class_idx is None else class_idx
            class_idxes.append(class_idx_i)
            class_outputs.append(prediction[class_idx_i])
        grads = tape.gradient(class_outputs, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # 对所有空间位置平均池化
    
    return conv_outputs, class_idxes, class_outputs, pooled_grads



# Grad-CAM计算
@time_cost_async
async def get_gradcam(model, img_array, class_idx=None):
    conv_maps = []

    # # 使用ThreadPoolExecutor来异步执行模型推理部分
    # loop = asyncio.get_event_loop()
    # with ThreadPoolExecutor() as executor:
    #     # 使用run_in_executor异步执行模型推理
    #     conv_outputs, class_idxes, class_outputs, pooled_grads = await loop.run_in_executor(executor, model_inference, model, img_array, class_idx)

    conv_outputs, class_idxes, class_outputs, pooled_grads = await model_inference(model, img_array, class_idx)
    
    # 异步计算每个conv_map
    tasks = []
    for i, conv_output in enumerate(conv_outputs):
        # 异步计算每个conv_map
        task = asyncio.create_task(process_conv_map(conv_output, pooled_grads[i]))
        tasks.append(task)

    # 等待所有任务完成
    conv_maps = await asyncio.gather(*tasks)
    
    return conv_maps, class_idxes, class_outputs


# @time_cost_async
# async def get_gradcam(model, img_array, class_idx=None):
#     class_idxes = []
#     class_outputs = []        
#     conv_maps = []
#     with tf.GradientTape() as tape:
#         img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
#         tape.watch(img_array)
#         conv_outputs, predictions = model(img_array)
#         for i, prediction in enumerate(predictions):
#             class_idx_i = np.argmax(prediction) if class_idx == None else class_idx
#             class_idxes.append(class_idx_i)
#             class_outputs.append(prediction[class_idx_i])            
#     grads = tape.gradient(class_outputs, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # 对所有空间位置平均池化
#     # convmaps
#     for i, conv_output in enumerate(conv_outputs):
#         conv_output = conv_output @ pooled_grads[i][..., tf.newaxis]  # 点乘，channel维度梯度加权
#         conv_map = tf.reduce_sum(conv_output, axis=-1)
#         conv_map = np.maximum(conv_map, 0)  # ReLU激活
#         conv_map = conv_map / (np.max(conv_map) - np.min(conv_map))  # 归一化
#         conv_map = await threshold_map_max(conv_map, percentile=70)  # 不关注低于阈值的区域
#         conv_maps.append(conv_map)

#     return conv_maps, class_idxes, class_outputs


async def convmap2colormap(conv_map):
    # Rescale convmap to a range 0-255
    conv_map = np.uint8(255 * conv_map)
    # Use jet colormap to colorize convmap
    jet = plt.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    color_map = jet_colors[conv_map]

    return color_map


async def gen_superimposed_img(conv_map, img_path, alpha):
    img = cv2.imread(img_path, flags=0)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img_bgr.shape[:2]

    # convmap -> resize -> colormap
    resized_conv_map = cv2.resize(conv_map, (w, h), interpolation=cv2.INTER_LINEAR)
    color_map = await convmap2colormap(resized_conv_map)
    color_map = tf.keras.preprocessing.image.array_to_img(color_map)
    color_map = tf.keras.preprocessing.image.img_to_array(color_map)

    # Superimpose重叠 the colormap on original image        
    superimposed_img = color_map * alpha + img_bgr * (1 - alpha)
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img


async def heatmaps_sum_all(conv_maps, show_img_path, alpha=0.4, save_path=None):
    conv_map_all = np.array([None])
    for conv_map in conv_maps:
        conv_map_all = np.maximum(conv_map_all, conv_map) if conv_map_all.any() != None else conv_map
    conv_map_all /= (np.max(conv_map_all) - np.min(conv_map_all))  # 归一化
    superimposed_img = await gen_superimposed_img(conv_map_all, show_img_path, alpha)
    # Save the superimposed image
    cam_save_path = os.path.join(os.path.dirname(img_paths[0]), "cam_result") if save_path == None else save_path
    check_or_makedir(cam_save_path)
    cam_path = os.path.join(cam_save_path, "all_result.bmp")
    superimposed_img.save(cam_path)

    return


async def heatmaps_frequency_sum_all(conv_maps, show_img_path, alpha=0.4, save_path=None):
    conv_maps_stack = np.stack(conv_maps, axis=0)
    conv_maps_sum = np.sum(conv_maps_stack, axis=0)  # 全部相加，等价于根据样本频率加权
    conv_maps_sum = np.maximum(conv_maps_sum, 0)  # ReLU激活
    conv_maps_sum /= (np.max(conv_maps_sum) - np.min(conv_maps_sum))  # 归一化
    superimposed_img = await gen_superimposed_img(conv_maps_sum, show_img_path, alpha)
    # Save the superimposed image
    cam_save_path = os.path.join(os.path.dirname(img_paths[0]), "cam_result") if save_path == None else save_path
    check_or_makedir(cam_save_path)
    cam_path = os.path.join(cam_save_path, "all_result_frequency.bmp")
    superimposed_img.save(cam_path)

    return


async def threshold_map_max(conv_map, percentile=30):
    # 计算map图的最大值
    max_value = np.max(conv_map)
    # 计算阈值：最大值的 percentile%
    threshold = max_value * (percentile / 100.0)
    # 将低于阈值的值置为 0
    conv_map[conv_map < threshold] = 0
    # 归一化
    conv_map /= (np.max(conv_map) - np.min(conv_map))
    return conv_map


def load_data(img_path, shape):
    img_height, img_width = shape
    print(img_path)
    img = tf.io.read_file(img_path)
    img = tf.io.decode_bmp(img, channels=0)
    img_array = tf.image.resize(img, [img_height, img_width])
    
    return img_array


@time_cost_async
async def main(model, img_paths, save_path=None, batch_size=1, alpha=0.4, save=True, class_idx=None):
    class_idxes_all, class_outputs_all = [], []
    conv_maps_all = []
    shape = model.input_spec[0].shape[1:3]  # (h, w)
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)    
    dataset = dataset.map(lambda img_path: load_data(img_path, shape), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    for k, img_arrays in enumerate(dataset):

        # Grad-CAM
        conv_maps, class_idxes, class_outputs = await get_gradcam(model, img_arrays, class_idx)

        conv_maps_all.extend(conv_maps)
        class_idxes_all.extend(class_idxes)
        class_outputs_all.extend(class_outputs)

        if save:
            cam_save_path = os.path.join(os.path.dirname(img_paths[0]), "cam_result") if save_path == None else save_path
            check_or_makedir(cam_save_path)
            for i, conv_map in enumerate(conv_maps):                
                # img_path = img_paths[j+i]
                img_path = img_paths[k*batch_size+i]
                superimposed_img = await gen_superimposed_img(conv_map, img_path, alpha)
                # Save the superimposed image
                name = os.path.basename(img_path)
                cam_path = os.path.join(cam_save_path, name)
                superimposed_img.save(cam_path)
                print(f"img_path:{img_path}, class_idx:{class_idxes[i]}, class_output:{class_outputs[i]}")

    return conv_maps_all, class_idxes_all, class_outputs_all


if __name__ == "__main__":
    model_path = "heatmap/models/2X77G-0823.h5"
    device_id = 2
    alpha = 0.4

    img_root_path = '/var/cdy_data/aoyang/data/heatmap/BGB2X77G/Luminous_scratch'
    img_paths = glob.glob(os.path.join(img_root_path, "*.bmp"))

    show_img_path = "/var/cdy_data/aoyang/data/heatmap/BGB2X77G/R15C21_8_L1#498_730_1823_2060_0_0_0_0.bmp"

    # save_path = "/var/cdy_data/aoyang/data/heatmap/BGB1X53B_cam_result_all"
    save_path = None

    # 1. 初始化GPU
    init_gpu_env(device_id)
    # 2. 加载模型
    # model = ModelWithConvOut(model_path)
    model = ModelWithConvOut(model_path=model_path, conv_layer_name="block4d_add")
    # 3. 生成热力图并绘制
    conv_maps, class_idxs, class_outputs = asyncio.run(
        main(
            model=model,
            img_paths=img_paths,
            batch_size=4,
            alpha=alpha,
            save_path=save_path,
            # class_idx=19
        )
    )
    
    asyncio.run(
        heatmaps_sum_all(
            conv_maps,
            show_img_path=show_img_path,
            alpha=alpha,
            save_path=save_path,
        )
    )

    asyncio.run(
        heatmaps_frequency_sum_all(
            conv_maps,
            show_img_path=show_img_path,
            alpha=alpha,
            save_path=save_path,
        )
    )
        