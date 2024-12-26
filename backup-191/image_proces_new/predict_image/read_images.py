import cv2 as cv
import os
import numpy as np


# 读取图片 返回图片字典
def read_img(refer_path, light_list):
    # print(refer_path)
    light_image = {}
    base_path, refer_name = os.path.split(refer_path)
    
    if ".bmp" in refer_name:
        if "L" in refer_name:
            rc = refer_name.split("L")[0]
            for light in light_list:
                img_path = os.path.join(base_path, rc + light + ".bmp")
                if not os.path.exists(img_path):
                    print(f"缺失图片:{img_path}")
                else:
                    light_image[light] = cv.imread(img_path, 0)
        else:
            rc = os.path.splitext(refer_name)[0].split("_")[-1]
            for light in light_list:
                img_path = os.path.join(base_path, light + "_" + rc + ".bmp")
                if not os.path.exists(img_path):
                    print(f"缺失图片:{img_path}")
                else:
                    light_image[light] = cv.imread(img_path, 0)

    elif ".raw" in refer_name:
        rc = os.path.splitext(refer_name)[0].split("_")[-1]
        for light in light_list:
            img_path = os.path.join(base_path, light + "_" + rc + ".raw")
            if not os.path.exists(img_path):
                print(f"缺失图片:{img_path}")
            else:
                img = np.fromfile(img_path, dtype="uint8")
                if img.size == 5120 * 5120:
                    img = img.reshape(5120, 5120, 1)
                    light_image[light] = img
                else:
                    print("raw image 解密失败")
    else:
        print("没有对应的图片类型")

    light_image["R_C"] = rc

    return light_image
