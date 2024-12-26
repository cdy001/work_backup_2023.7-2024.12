import numpy as np

import tensorflow as tf

from utils.postprocess import threshold_map_max
from utils.decorator import time_cost


@time_cost
def get_gradcam(model, img_array, class_idx=None):
    class_idxes = []
    class_outputs = []        
    conv_maps = []
    with tf.GradientTape() as tape:
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        tape.watch(img_array)
        conv_outputs, predictions = model(img_array)
        for i, prediction in enumerate(predictions):
            class_idx_i = np.argmax(prediction) if class_idx == None else class_idx
            class_idxes.append(class_idx_i)
            class_outputs.append(prediction[class_idx_i])            
        grads = tape.gradient(class_outputs, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # 对所有空间位置平均池化
    # convmaps
    for i, conv_output in enumerate(conv_outputs):
        conv_output = conv_output @ pooled_grads[i][..., tf.newaxis]  # 点乘，channel维度梯度加权
        conv_map = tf.reduce_sum(conv_output, axis=-1)
        conv_map = np.maximum(conv_map, 0)  # ReLU激活
        conv_map = conv_map / np.max(conv_map)  # 归一化
        conv_map = threshold_map_max(conv_map, percentile=70)  # 不关注低于阈值的区域
        conv_maps.append(conv_map)
    conv_maps = np.array(conv_maps)

    return conv_maps, class_idxes, class_outputs


@time_cost
def get_layercam(model, img_array, class_idx=None):
    conv_maps_all = []
    class_idxes = []
    class_outputs = []        
    
    with tf.GradientTape() as tape:
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
        tape.watch(img_array)
        conv_outputs_all, predictions = model(img_array)
        for i, prediction in enumerate(predictions):
            class_idx_i = np.argmax(prediction) if class_idx == None else class_idx
            class_idxes.append(class_idx_i)
            class_outputs.append(prediction[class_idx_i])            
        grads_all = tape.gradient(class_outputs, conv_outputs_all)
        max_shape = grads_all[-1].shape[1:3]
    for index, grads in enumerate(grads_all):
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))  # 对所有空间位置平均池化
        conv_outputs = conv_outputs_all[index]
        # convmaps
        conv_maps = []
        for i, conv_output in enumerate(conv_outputs):
            conv_output = tf.image.resize(conv_output, max_shape)
            conv_output = conv_output @ pooled_grads[i][..., tf.newaxis]  # 点乘，channel维度梯度加权
            conv_map = tf.reduce_sum(conv_output, axis=-1)
            conv_map = np.maximum(conv_map, 0)  # ReLU激活
            conv_map = conv_map / np.max(conv_map)  # 归一化
            conv_map = threshold_map_max(conv_map, percentile=70)  # 不关注低于阈值的区域
            conv_maps.append(conv_map)
        conv_maps_all.append(conv_maps)
    conv_maps_all = np.array(conv_maps_all)
    conv_maps_all = np.amax(conv_maps_all, axis=0)

    return conv_maps_all, class_idxes, class_outputs