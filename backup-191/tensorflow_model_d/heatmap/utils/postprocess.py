import os
import cv2
import numpy as np

from matplotlib import pyplot as plt
import tensorflow as tf


def check_or_makedir(save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.system(f"chmod -R 777 {save_path}")


def create_save_path(img_paths, save_path):
    # cam_save_path = os.path.join(os.path.dirname(img_paths[0]), "cam_result") if save_path == None else save_path
    class_dirname = os.path.basename(os.path.dirname(img_paths[0]))
    root_dir = os.path.dirname(os.path.dirname(img_paths[0]))
    cam_save_path = os.path.join(root_dir, "result", class_dirname) if save_path == None else save_path
    check_or_makedir(cam_save_path)
    return cam_save_path


def threshold_map_max(conv_map, percentile=30):
    # 计算map图的最大值
    max_value = np.max(conv_map)
    # 计算阈值：最大值的 percentile%
    threshold = max_value * (percentile / 100.0)
    # 将低于阈值的值置为 0
    conv_map[conv_map < threshold] = 0
    # conv_map /= (np.max(conv_map) - np.min(conv_map))  # 归一化
    return conv_map


def convmap2colormap(conv_map):
    # Rescale convmap to a range 0-255
    conv_map = np.uint8(255 * conv_map)
    # Use jet colormap to colorize convmap
    jet = plt.get_cmap("jet")
    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    color_map = jet_colors[conv_map]

    return color_map


def gen_superimposed_img(conv_map, img_path, alpha):
    img = cv2.imread(img_path, flags=0)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img_bgr.shape[:2]
    conv_map = cv2.resize(conv_map, (w, h), interpolation=cv2.INTER_LINEAR)
    # h, w = conv_map.shape[:2]
    # img_bgr = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    
    # convmap -> resize -> colormap
    color_map = convmap2colormap(conv_map)
    color_map = tf.keras.preprocessing.image.array_to_img(color_map)
    color_map = tf.keras.preprocessing.image.img_to_array(color_map)

    # Superimpose重叠 the colormap on original image        
    superimposed_img = color_map * alpha + img_bgr * (1 - alpha)
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img