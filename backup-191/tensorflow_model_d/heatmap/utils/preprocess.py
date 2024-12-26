import os
import tensorflow as tf


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