import os
import json
import shutil
import argparse

from tvt_data import DatasetSplit
from main import train_validation_test


def args_config():
    parser = argparse.ArgumentParser(description="train tensorflow model")
    parser.add_argument(
        "--cfg", type=str, required=True, help="path of config for training"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        required=False,
        default="0",
        help="multi gpus training, i.e. '0,1' for use gpu0 and gpu1",
    )

    args = parser.parse_args()

    return args


def main():
    input_args = args_config()
    json_file = input_args.cfg
    gpu_list = list(map(int, input_args.gpus.split(",")))
    assert json_file.endswith(".json")
    with open(json_file, mode="r") as f:
        args = json.load(f)

    data_path = args["data_path"]
    label_path = args["label_path"]
    weight_path = args["pretrained"]
    img_shape = [args["image_height"], args["image_width"], 1]
    batch_size = args["batch_size"]
    learning_rate = args["lr"]
    epochs = args["epochs"]

    with open(label_path, mode="r") as f:
        labels = json.load(f)
        num_class = len(labels)

    ## 1. 创建data_tvt文件夹用于训练
    data_tvt = data_path + "_tvt"
    # 先删除旧的data_tvt文件夹
    if os.path.exists(data_tvt):
        shutil.rmtree(data_tvt)
    dataset = DatasetSplit(
        data_path=data_path, dest_path=data_tvt, label_path=label_path
    )
    dataset.tvt()

    ## 2. 使用最新data_tvt文件夹的数据训练
    train_validation_test(
        tvt_path=data_tvt,
        gpu_list=gpu_list,
        img_shape=img_shape,
        batch_size=batch_size,
        num_class=num_class,
        weight_path=weight_path,
        epochs=epochs,
        learn_rate=learning_rate,
    )


if __name__ == "__main__":
    main()
