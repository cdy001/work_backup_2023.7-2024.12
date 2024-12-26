import cv2
import numpy as np


def process_image_06(img):
    _, binary_img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
    return binary_img


def judge_die_06(binary_img, light_image, die_left_x, die_left_y, die_right_x, die_right_y):
    # die_img = binary_img[die_left_y:die_right_y, die_left_x:die_right_x]
    
    die_img_L4 = light_image["L4"][die_left_y:die_right_y, die_left_x:die_right_x]
    
    if np.sum(die_img_L4)<(255*54*54)*0.8:
        return False

    # if np.sum(die_img) < 400:
    #     return False

    return True


def process_image(light_image, wafer_type):
    if "AUTO-FGLR05EL" in wafer_type:
        img = light_image["L3"]
        binary_img = process_image_06(img)
        is_judge = True
        return binary_img, is_judge
    else:
        return None, False


def is_die(
    binary_img,
    light_image,
    die_left_x,
    die_left_y,
    die_right_x,
    die_right_y,
    wafer_type,
):
    if "AUTO-FGLR05EL" in wafer_type:
        t_f = judge_die_06(
            binary_img, light_image, die_left_x, die_left_y, die_right_x, die_right_y
        )
        return t_f
    else:
        return False
