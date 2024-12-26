import cv2 as cv
import glob
import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())

from cut.config.config import get_config
from predict_image import partition_path, predict
from predict_image import read_images


def plt(light_image, total_names, total_pre, R_C, save_path_plt):
    del light_image['R_C']
    for light in light_image:
        img = light_image[light]

        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for name, pre in zip(total_names, total_pre):
            if R_C in name.decode() and light in name.decode():
                # 特殊缺陷判断
                die_flag_2 = os.path.splitext(name.decode())[0].split("#")[-1].split("_")[5]
                if int(die_flag_2) != 0:
                    pre = int(die_flag_2)

                x1, y1, x2, y2, *_ = (
                    os.path.splitext(name.decode())[0].split("#")[-1].split("_")
                )
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)
                x_locate = int((x1 + x2) / 2) - 15
                y_locate = int((y1 + y2) / 2) + 15

                if pre != 0:
                    cv.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv.putText(
                        img,
                        str(pre),
                        (x_locate, y_locate),
                        cv.FONT_HERSHEY_COMPLEX,
                        1.2,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv.putText(
                        img,
                        str(pre),
                        (x_locate, y_locate),
                        cv.FONT_HERSHEY_COMPLEX,
                        1.2,
                        (0, 255, 0),
                        2,
                    )

        dst_path = os.path.join(save_path_plt, R_C + light + ".bmp")
        cv.imwrite(dst_path, img)


def pre_plt_die(model_path, gpu, refer_paths, save_path_plt, light_list, recipe):
    if not os.path.exists(save_path_plt):
        os.mkdir(save_path_plt)

    # 划分数据
    total_name = []
    total_pre = []
    total_paths = partition_path.partition_path_list(refer_paths)
    predict.test_patch_predict(
        model_path, gpu, total_paths, total_name, total_pre, recipe
    )

    for refer_path in refer_paths:
        print(refer_path)
        light_image = read_images.read_img(refer_path, light_list)

        R_C = light_image["R_C"]
        plt(light_image, total_name, total_pre, R_C, save_path_plt)


def pre_plt_die_main():
    gpu = "0"
    recipe = "0945Y"
    img_type = "bmp"
    model_path = "models_label/0945Y_1105_40_epoch.h5"

    image_info, _, _ = get_config(recipe)
    folder_path = os.path.join("/var/cdy_data/jucan/wafer/rejudge/0945Y/11.5")
    for folder in glob.glob(folder_path):
        refer_paths = glob.glob(
            os.path.join(folder, "*" + image_info["refer_light"] + f"*{img_type}")
        )
        save_path_plt = folder + "_p"
        light_list = image_info["light_list"]
        pre_plt_die(model_path, gpu, refer_paths, save_path_plt, light_list, recipe)


if __name__ == "__main__":
    pre_plt_die_main()
