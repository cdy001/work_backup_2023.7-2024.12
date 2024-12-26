# coding=utf-8
import glob
import os
import sys
import time
import argparse

print(os.getcwd())
sys.path.append(os.getcwd())

from cut.config.config import get_config_new
from csv_file import save_read_csv
from predict_image import partition_path
from predict_image.predict import WaferPredict

def pre_images_save_csv(model_path, gpu, refer_paths, dest_csv_path, recipe):
    total_paths = partition_path.partition_path_list(refer_paths)

    # 迭代图像块
    total_name = []
    total_pre = []
    top_values = []
    top_labels = []

    wafer_predict = WaferPredict(
        model_path,
        gpu,
        total_paths,
        total_name,
        total_pre,
        top_values,
        top_labels,
        recipe)
    time_cut, time_inference = wafer_predict.test_patch_predict()
    
    save_read_csv.save_csv(total_name, total_pre, top_values, top_labels, dest_csv_path)

    return time_cut, time_inference

def predict_save_csv_die(recipe, image_folders, label_path, model_path, gpu=0, img_type="bmp"):
    time_start = time.time()
    recipe = recipe
    image_folders = image_folders
    label_path = label_path
    model_path = model_path
    gpu = gpu
    img_type = img_type

    time_cut_all = 0
    time_inference_all = 0
    image_info, _, _ = get_config_new(recipe)
    for folder_path in glob.glob(image_folders):
        refer_paths = glob.glob(os.path.join(folder_path, "*" + image_info["refer_light"] + f"*{img_type}"))

        base_path, file_name = os.path.split(folder_path)
        save_path_csv = base_path
        if not os.path.exists(save_path_csv):
            os.makedirs(save_path_csv)
        dest_csv_path = os.path.join(save_path_csv, file_name + ".csv")

        # predict images + save die by csv
        time_cut, time_inference = pre_images_save_csv(model_path, gpu, refer_paths, dest_csv_path, recipe)
        time_cut_all += time_cut
        time_inference_all += time_inference

        # read csv to save die
        csv_path = dest_csv_path
        img_path = folder_path

        save_path = folder_path + "_die"
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_read_csv.save_die(csv_path, img_path, save_path, label_path, img_type, recipe)

    time_end = time.time()
    print("time cut die: {:.4f}s".format(time_cut_all))
    print("time inference: {:.4f}s".format(time_inference_all))
    print("time total: {:.4f}s".format((time_end-time_start)))

def args_config():
    parser = argparse.ArgumentParser(description="argparse for input.")
    parser.add_argument('--recipe', type=str, required=True, help="recipe")
    parser.add_argument('--image_folders', type=str, required=True, help="image folder with unified recipe")
    parser.add_argument('--label_path', type=str, required=True, help="file includes name of every class")
    parser.add_argument('--model_path', type=str, required=True, help="model to predict")
    parser.add_argument('--gpu', type=int, required=False, default=0, help="which gpu to use")
    parser.add_argument('--img_type', type=str, required=False, default="bmp", help="bmp or raw")

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    # recipe = "BGB1Q42E"    
    # image_folders = r"/var/cdy_data/aoyang/wafer/1Q42E-D/HNJ31N59780B11"
    # label_path = r"models_label/BOB1P42M.txt"
    # model_path = r"models_label/1Q42E-D_0702_b36.trt"
    # gpu = 0
    # img_type = "bmp"
    args = args_config()
    predict_save_csv_die(
        recipe=args.recipe,
        image_folders=args.image_folders,
        label_path=args.label_path,
        model_path=args.model_path,
        gpu=args.gpu,
        img_type=args.img_type
    )
