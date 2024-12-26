import cv2
import numpy as np


def check_is_right_die(die, img, offset=200):
    is_die = False
    img_height, img_width = img.shape

    # remove in edge die
    die_left_x = die[2]
    die_left_y = die[3]
    die_right_x = die[4]
    die_right_y = die[5]

    # remove in edge die
    if abs(die_right_x - img_height) < 20 or abs(die_right_y - img_height) < 20 or die_left_x < 20 or die_left_y < 20:
        return False, None

    # 电极面积
    die_img = img[die_left_y:die_right_y, die_left_x:die_right_x]
    if np.count_nonzero(die_img) < offset:
        return False, None

    contours_pad, hierarchy_pad = cv2.findContours(image=die_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    if len(contours_pad) == 2:
        return True, contours_pad

    return is_die, None


def cal_l1_offset(die, L1_img, l4_contours_pad):
    die_left_x = die[2]
    die_left_y = die[3]
    die_right_x = die[4]
    die_right_y = die[5]

    # 现在L4的基础上扩大一定的范围，保证能够找到一组pad
    die_img_l1 = L1_img[die_left_y - 10:die_right_y + 10, die_left_x - 15:die_right_x + 15]

    _, thread_l1 = cv2.threshold(die_img_l1, 160, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thread_l1 = cv2.morphologyEx(thread_l1, cv2.MORPH_CLOSE, kernel)
    l1_contours_pad, hierarchy_pad = cv2.findContours(image=thread_l1, mode=cv2.RETR_EXTERNAL,
                                                      method=cv2.CHAIN_APPROX_NONE)

    l1_pads = []
    for i, contorPoint in enumerate(l1_contours_pad):
        x, y, w, h = cv2.boundingRect(contorPoint)
        wh_rate = min(h, w) / max(h, w)
        if w * h > 200 and 0.7 < wh_rate:
            l1_pads.append([x + int(w / 2), y + int(h / 2), x, y, x + w, x + h])
    if len(l1_pads) != 2:
        return False, None

    l4_pads = []
    for contorPoint in l4_contours_pad:
        x, y, w, h = cv2.boundingRect(contorPoint)
        l4_pads.append([x + int(w / 2), y + int(h / 2), x, y, x + w, x + h])
    if len(l4_pads) != 2:
        return False, None

    is_up_left_l4 = True
    if l4_pads[0][1] > l4_pads[1][1]:
        if l4_pads[0][0] > l4_pads[1][0]:
            is_up_left_l4 = True
        else:
            is_up_left_l4 = False
    else:
        if l4_pads[0][0] > l4_pads[1][0]:
            is_up_left_l4 = False
        else:
            is_up_left_l4 = True

    is_up_left_l1 = True
    if l1_pads[0][1] > l1_pads[1][1]:
        if l1_pads[0][0] > l1_pads[1][0]:
            is_up_left_l1 = True
        else:
            is_up_left_l1 = False
    else:
        if l1_pads[0][0] > l1_pads[1][0]:
            is_up_left_l1 = False
        else:
            is_up_left_l1 = True

    # 计算偏移量
    if is_up_left_l1 == is_up_left_l4:
        l4_die_center = (int((l4_pads[0][0] + l4_pads[1][0]) / 2), int((l4_pads[0][1] + l4_pads[1][1]) / 2))
        l1_die_center = (int((l1_pads[0][0] + l1_pads[1][0]) / 2), int((l1_pads[0][1] + l1_pads[1][1]) / 2))

        offset = (l1_die_center[0] - (l4_die_center[0] + 15), l1_die_center[1] - (l4_die_center[1] + 10))
        return True, offset
    else:
        return False, None


def calculation_l1_offset(refer_dies, electrode_die_img_L4, L1_img):
    l1_offset = [0, 0]
    count_die_true = 0

    for i, die in enumerate(refer_dies):
        is_suit_die, pads = check_is_right_die(die, electrode_die_img_L4)
        if not is_suit_die:
            continue

        is_right, offset = cal_l1_offset(die, L1_img, pads)

        if is_right:
            l1_offset[0] += offset[0]
            l1_offset[1] += offset[1]
            count_die_true += 1

        if count_die_true > 2:
            break

    if count_die_true != 0:
        l1_offset[0] = - int(l1_offset[0] / count_die_true)
        l1_offset[1] = - int(l1_offset[1] / count_die_true)
    return l1_offset
