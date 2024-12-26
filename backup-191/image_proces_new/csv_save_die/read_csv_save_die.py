# coding=utf-8
import glob
import os
import sys
import time

print(os.getcwd())
sys.path.append(os.getcwd())

from cut.config.config import get_config, get_config_new
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

def predict_save_csv_die():
    time_start = time.time()
    recipe = "BGB1Q42E"
    # recipe = "3535"
    image_info, _, _ = get_config_new(recipe)

    # predict + save die by csv
    folder_paths = os.path.join(r"/var/cdy_data/aoyang/wafer/1Q42E-D/HNJ31N59780B11")
    label_path = r"models_label/BOB1P42M.txt"
    model_path = r"models_label/trt_models/1Q42E-D_0702_b24.trt"
    gpu = 0
    img_type = "bmp"

    time_cut_all = 0
    time_inference_all = 0
    for folder_path in glob.glob(folder_paths):
        refer_paths = glob.glob(os.path.join(folder_path, "*" + image_info["refer_light"] + f"*{img_type}"))

        base_path, file_name = os.path.split(folder_path)
        save_path_csv = base_path
        if not os.path.exists(save_path_csv):
            os.makedirs(save_path_csv)
        dest_csv_path = os.path.join(save_path_csv, file_name + ".csv")

        # predict images
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

def csv_die():
    # S-08HGAUD-C
    # save die by csv
    recipe = "0945A"
    img_type = "raw"
    result_folders = os.path.join("/media/data/code_and_dataset/wafer/1008_2/result/K*")
    for folder in glob.glob(result_folders):
        txt_path = os.path.join(folder, "predict_out.txt")

        base_path_1, file_name = os.path.split(folder)
        base_path_2, _ = os.path.split(base_path_1)
        img_path = os.path.join(base_path_2, file_name)

        save_path_die = os.path.join(base_path_1, file_name + "_die")

        label_path = "models_label/0945.txt"

        save_read_csv.save_die(txt_path, img_path, save_path_die, label_path, img_type, recipe)


if __name__ == "__main__":
    predict_save_csv_die()

    # csv_die()
