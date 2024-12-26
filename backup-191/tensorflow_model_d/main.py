# coding=utf-8
import tensorflow as tf
import assign_device
import load_data
import models
import train_test


def train_validation_test(tvt_path, gpu_list, img_shape, batch_size, num_class, weight_path, epochs, learn_rate):
    # 设置gpu的可见性 + gpu内存增长
    assign_device.assign_gpus(gpu_list)

    # create a MirroredStrategy instance
    specified_gpu = []
    for i in gpu_list:
        specified_gpu.append(f"GPU:{i}")
    print(f'MirroredStrategy with devices: {specified_gpu}')
    strategy = tf.distribute.MirroredStrategy(devices=specified_gpu)

    # 数据加载 + distribute the dataset based on the strategy
    num_replicas = strategy.num_replicas_in_sync
    global_batch_size = num_replicas * batch_size
    train_batch, validation_batch, test_batch = load_data.train_validation_test_dataset(tvt_path, img_shape,
                                                                                        global_batch_size, batch_size)
    # print(train_batch)

    train_batch = strategy.experimental_distribute_dataset(train_batch)
    validation_batch = strategy.experimental_distribute_dataset(
        validation_batch)

    # 创建strategy 模型
    with strategy.scope():
        # model = tf.keras.models.load_model("./tmp/100_epoch.h5")
        model = models.build_model(img_shape, weight_path, num_class)
        # model.summary()

    train_test.training_model(strategy=strategy, model=model, train_dataset=train_batch, val_dataset=validation_batch,
                              learn_rate=learn_rate, global_batch_size=global_batch_size, epochs=epochs)

    model = tf.keras.models.load_model("./tmp/best_model.h5")
    train_test.testing_model(model, test_batch)


if __name__ == '__main__':
    # 数据集路径
    tvt_path_s = '/var/cdy_data/aoyang/data/BGB1Q42E-D_tvt'

    # gpus
    gpu_list_s = [0, 1]

    # 模型参数
    weight_path_s = 'tmp/40_epoch.h5'
    num_class_s = 42    # BGB1Q42E-D BPB0W42C
    # num_class_s = 32    # 3535

    # 超参数配置
    # img_shape_s = [128, 608, 1]     # 0945A, [H, W, C]
    # img_shape_s = [134, 588, 1]     # 0945Y 0945W
    # img_shape_s = [136, 592, 1]     # 0945U
    # img_shape_s = [606, 606, 1]     # BOB1J37H
    # img_shape_s = [716, 716, 1]     # BWB1P43B
    # img_shape_s = [680, 680, 1]     # BOB1P42M
    # img_shape_s = [928, 864, 1]     # BGB1X53B
    img_shape_s = [682, 682, 1]     # BGB1Q42E-D
    # img_shape_s = [694, 400, 1]     # BPB0W42C
    # img_shape_s = [714, 714, 1]     # BWB1O42A
    # img_shape_s = [520, 490, 1]     # BWB1B29A
    # img_shape_s = [500, 500, 1]     # BNB1C30D
    # img_shape_s = [505, 510, 1]    # BGB1B28C
    batch_size_s = 8
    # batch_size_s = 24
    # img_shape_s = [96, 62, 1]     # 0305
    # batch_size_s = 512
    # img_shape_s = [616, 616, 1]     # 3535
    # batch_size_s = 16

    learn_rate_s = 0.00001
    epochs_s = 301
    print(
        f"tvt_path: {tvt_path_s}\ngpu_list: {gpu_list_s}\nimg_shape: {img_shape_s}\nbatch_size: {batch_size_s}\n"
        f"num_class: {num_class_s}\nweight_path: {weight_path_s}\nepochs:{epochs_s}\nlearn_rate:{learn_rate_s}")

    # 训练和测试
    train_validation_test(tvt_path=tvt_path_s, gpu_list=gpu_list_s, img_shape=img_shape_s, batch_size=batch_size_s,
                          num_class=num_class_s, weight_path=weight_path_s, epochs=epochs_s, learn_rate=learn_rate_s)
