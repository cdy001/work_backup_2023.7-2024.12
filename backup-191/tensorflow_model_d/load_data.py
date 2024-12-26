import tensorflow as tf
import tensorflow_addons as tfa
import os

margin_x = int(5)
margin_y = int(5)


# 读取图片 + 图像尺寸变化
def process_img(path, img_height, img_width):
    img = tf.io.read_file(path)
    img = tf.io.decode_bmp(img, channels=0)
    img = tf.image.resize(img, [img_height, img_width])

    return img


# 提取label
def process_label(path):
    parts = tf.strings.split(path, os.path.sep)
    label = parts[-2]
    label = tf.strings.to_number(label, out_type=tf.int32)

    return label


# 数据扩充
def augmentation_fun(img, img_height, img_width):
    if tf.random.uniform(()) > 0.5:
        img = tf.image.flip_left_right(img)

    if tf.random.uniform(()) > 0.9:
        r = tf.random.uniform((), minval=-0.15, maxval=0.15)
        img = tfa.image.rotate(img, r)

    if tf.random.uniform(()) > 0.9:
        cc = tf.random.uniform((), minval=0.8, maxval=1.0)
        img = tf.image.central_crop(img, cc)
        img = tf.image.resize(img, [img_height, img_width])

    if tf.random.uniform(()) > 0.9:
        c = tf.random.uniform((), minval=0.8, maxval=1.0)
        img = tf.image.random_crop(img, [int(img_height * c), int(img_width * c), 1])

    if tf.random.uniform(()) > 0.9:
        b = tf.random.uniform((), minval=-0.15, maxval=0.15)
        img = tf.image.adjust_brightness(img, b)

    if tf.random.uniform(()) > 0.9:
        c = tf.random.uniform((), minval=0.7, maxval=2)
        img = tf.image.adjust_contrast(img, c)

    if tf.random.uniform(()) > 0.7:
        x_t = tf.random.uniform(shape=[], minval=-margin_x, maxval=margin_x, dtype=tf.int32)
        y_t = tf.random.uniform(shape=[], minval=-margin_y, maxval=margin_y, dtype=tf.int32)
        img = tfa.image.transform(img, [1.0, 0.0, x_t, 0.0, 1.0, y_t, 0.0, 0.0])

    img = tf.image.resize(img, [img_height, img_width])

    return img


# 加载数据
def load_dataset(file_path, img_height, img_width, batch_size, model):
    def process_path(path_label):
        img = process_img(path_label, img_height, img_width)

        if model == 'train':
            img = augmentation_fun(img, img_height, img_width)

        label = process_label(path_label)

        return img, label

    if model == 'train' or model == 'validation':
        is_drop = True
    else:
        is_drop = False

    list_pa = tf.data.Dataset.list_files(file_path)
    image_label = list_pa.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    data_batch = image_label.shuffle(buffer_size=1000).batch(batch_size=batch_size, drop_remainder=is_drop).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    return data_batch


# 分别加载训练，验证，测试数据集
def train_validation_test_dataset(data_path, img_shape, global_batch_size, batch_size):
    img_height, img_width = img_shape[0], img_shape[1]

    train_path = os.path.join(data_path, 'train', '*', '*bmp')
    validation_path = os.path.join(data_path, 'validation', '*', '*bmp')
    test_path = os.path.join(data_path, 'test', '*', '*bmp')

    train_batch = load_dataset(train_path, img_height, img_width, global_batch_size, 'train')
    validation_batch = load_dataset(validation_path, img_height, img_width, global_batch_size, 'validation')
    test_batch = load_dataset(test_path, img_height, img_width, batch_size, 'test')

    return train_batch, validation_batch, test_batch
