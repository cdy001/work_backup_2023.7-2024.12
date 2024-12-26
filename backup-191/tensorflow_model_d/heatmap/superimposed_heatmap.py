import os
import glob
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from abc import abstractmethod


def heatmaps_sum_all(conv_maps, show_img_path, img_paths, alpha=0.4, save_path=None):
    conv_map_all = np.array([None])
    for i, conv_map in enumerate(conv_maps):
        conv_map_all = np.maximum(conv_map_all, conv_map) if conv_map_all.any() != None else conv_map
    if conv_map_all.all() == None:
        return
    conv_map_all /= np.max(conv_map_all)  # 归一化
    superimposed_img = gen_superimposed_img(conv_map_all, show_img_path, alpha)
    # Save the superimposed image
    cam_save_path = create_save_path(img_paths, save_path)
    cam_path = os.path.join(cam_save_path, "all_result.bmp")
    superimposed_img.save(cam_path)


def heatmaps_frequency_sum_all(conv_maps, show_img_path, img_paths, alpha=0.4, save_path=None):
    if len(conv_maps) <= 0:
        return
    conv_maps_stack = np.stack(conv_maps, axis=0)
    conv_maps_sum = np.sum(conv_maps_stack, axis=0)  # 全部相加，等价于根据样本频率加权
    conv_maps_sum = np.maximum(conv_maps_sum, 0)  # ReLU激活
    conv_maps_sum /= np.max(conv_maps_sum) # 归一化
    superimposed_img = gen_superimposed_img(conv_maps_sum, show_img_path, alpha)
    # Save the superimposed image
    cam_save_path = create_save_path(img_paths, save_path)
    cam_path = os.path.join(cam_save_path, "all_result_frequency.bmp")
    superimposed_img.save(cam_path)
    

def inference(model_path, img_paths, save_path=None, batch_size=1, alpha=0.4, save=True, class_idx=None, method="Grad-CAM"):
    if method not in ["Grad-CAM", "Layer-CAM"]:
        raise ValueError("Invalid method. Must be 'Grad-CAM' or 'Layer-CAM'.")
    if method == "Grad-CAM":  # Grad-CAM
        model = ModelWithConvOut(model_path)  # Grad-CAM model
        get_cam = get_gradcam
    else:  # Layer-CAM
        model = ModelWithConvOuts(model_path)  # Layer-CAM model
        get_cam = get_layercam
    conv_maps_all, class_idxes_all, class_outputs_all = [], [], []
    shape = model.input_spec[0].shape[1:3]  # (h, w)
    dataset = gen_dataset(img_paths, batch_size, shape)
    
    for k, img_arrays in enumerate(dataset):
        
        conv_maps, class_idxes, class_outputs = get_cam(model, img_arrays, class_idx)

        conv_maps_all.extend(conv_maps)
        class_idxes_all.extend(class_idxes)
        class_outputs_all.extend(class_outputs)

        if save:
            cam_save_path = create_save_path(img_paths, save_path)
            for i, conv_map in enumerate(conv_maps):
                if class_idxes[i] == 0:
                    continue
                img_path = img_paths[k*batch_size+i]
                superimposed_img = gen_superimposed_img(conv_map, img_path, alpha)
                # Save the superimposed image
                name = os.path.basename(img_path)
                cam_path = os.path.join(cam_save_path, name)
                superimposed_img.save(cam_path)
                print(f"img_path:{img_path}, class_idx:{class_idxes[i]}, class_output:{class_outputs[i]}")
        
    print(f"conv_maps before filter: {len(conv_maps_all)}")
    conv_maps_all = [conv_map for conv_map, class_idx in zip(conv_maps_all, class_idxes_all) if class_idx != 0]
    print(f"conv_maps after filter: {len(conv_maps_all)}")

    return conv_maps_all, class_idxes_all, class_outputs_all


def remove_invalid_strings_inplace(strings):
    i = 0
    while i < len(strings):
        try:
            # 尝试将每个字符串编码为 UTF-8
            strings[i].encode('utf-8')  # 如果编码成功
            i += 1  # 如果没有错误，则继续下一个
        except UnicodeEncodeError:
            print(f"Unicode error in string at index {i}: {strings[i]}")
            # 编码失败，删除当前元素
            del strings[i]


def init_gpu_env(gpu_index):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def _load_data(img_path, shape):
    img_height, img_width = shape
    print(img_path)
    img = tf.io.read_file(img_path)
    img = tf.io.decode_bmp(img, channels=0)
    img_array = tf.image.resize(img, [img_height, img_width])
    
    return img_array


def gen_dataset(img_paths, batch_size, shape):
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    dataset = dataset.map(lambda img_path: _load_data(img_path, shape), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset


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


class BaseModel(tf.keras.Model):
    def __init__(self, model_path=None):
        super(BaseModel, self).__init__()
        self.original_model = tf.keras.models.load_model(model_path, compile=False)
        self.input_spec = self.original_model.input_spec

    @abstractmethod
    def _new_model(self, *args, **kwargs):        
        raise NotImplementedError

    def __call__(self, inputs):
        model = self._new_model()

        return model(inputs)


class ModelWithConvOut(BaseModel):
    '''
    This model is for Grad-CAM.
    '''
    def __init__(self, model_path=None, conv_layer_name="top_conv"):
        super().__init__(model_path)
        self.conv_layer_name = conv_layer_name
    def _new_model(self):
        model = tf.keras.Model(
            inputs=self.original_model.input,
            outputs=[
                self.original_model.get_layer(self.conv_layer_name).output,
                self.original_model.output
            ]
        )
        return model


class ModelWithConvOuts(BaseModel):
    '''
    This model is for Layer-CAM.
    '''
    def _new_model(self):
        model = tf.keras.Model(
            inputs=self.original_model.input,
            outputs=[
                [
                    self.original_model.get_layer("top_conv").output,
                    self.original_model.get_layer("block6i_add").output,
                    self.original_model.get_layer("block5f_add").output,
                    self.original_model.get_layer("block4d_add").output,
                    self.original_model.get_layer("block3c_add").output,
                    self.original_model.get_layer("block2c_add").output,
                    self.original_model.get_layer("block1b_add").output,
                ],
                self.original_model.output
            ]
        )
        return model


if __name__ == "__main__":
    method = "Grad-CAM"
    device_id = 2
    alpha = 0.4
    
    model_path = "/var/cdy_data/aoyang/data/heatmap/BGB1Q42E/350_epoch.h5"
    
    show_img_path = "/var/cdy_data/aoyang/data/heatmap/BGB1Q42E/828/R19C08_22_L1_1720683295.bmp"
    # save_path = None
    save_path = "/var/cdy_data/aoyang/data/heatmap/BGB1Q42E/828/test_tmp"

    # 初始化GPU
    init_gpu_env(device_id)

    # 生成热力图并绘制
    img_root_path = "/var/cdy_data/aoyang/data/heatmap/BGB1Q42E/828/Electrode_dirty"
    img_paths = glob.glob(os.path.join(img_root_path, "*.bmp"))
    remove_invalid_strings_inplace(img_paths)
    conv_maps, class_idxs, class_outputs = inference(
        model_path=model_path,
        img_paths=img_paths,
        batch_size=4,
        save_path=save_path,
        method=method,
        # class_idx=33
    )
    heatmaps_sum_all(
        conv_maps,
        show_img_path=show_img_path,
        img_paths=img_paths,
        save_path=save_path,
    )

    heatmaps_frequency_sum_all(
        conv_maps,
        show_img_path=show_img_path,
        img_paths=img_paths,
        save_path=save_path,
    )
