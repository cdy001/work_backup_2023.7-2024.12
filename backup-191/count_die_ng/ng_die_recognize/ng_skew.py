import numpy as np

from utils.z_score import z_score_outliers

def wh_ratio_abnormal(dies, die_wh_ratio):
    '''
    args:
        dies: 识别到的所有die
    return:
        indexes: 判断为歪斜的die的索引
    '''
    indexes = []
    for i, die in enumerate(dies):
        x_min, y_min, x_max, y_max, contour_area, rect_area = die
        ratio_min = min(die_wh_ratio, (x_max - x_min) / (y_max - y_min))
        ratio_max = max(die_wh_ratio, (x_max - x_min) / (y_max - y_min))
        if ratio_min < 1 < ratio_max:
            indexes.append(i)

    return indexes

def compute_angle(dies, die_wh_ratio):
    '''
    give: die_height, die_width, rect_height, rect_width
    for: alpha
    rect_height * rect_width = die_height * die_height + die_height * cos(alpha) * die_height * sin(alpha) + die_width * cos(alpha) * die_width * sin(alpha)
    sin(alpha) * cos(alpha) = rect_height * rect_width / (die_height * die_width * 2) - 1 / 2
    sin(2 * alpha) = rect_height * rect_width / (die_height * die_width) - 1
    alpha = arcsin(rect_height * rect_width / (die_height * die_width) - 1) / 2
    rect_height * rect_width = rect_area
    die_height * die_width = contour_area
    0 <= alpha < pi / 2

    rect_height = new_hieght = die_width * sin(alpha) + die_height * cos(alpha)
    # new_width = die_width * cos(alpha) - die_height * sin(alpha)
    # new_height ^ 2 + new_width ^ 2 = die_height ^ 2 + die_width ^ 2
    '''
    angles = []
    for i, die in enumerate(dies):
        x_min, y_min, x_max, y_max, contour_area, rect_area = die
        ratio_min = min(die_wh_ratio, (x_max - x_min) / (y_max - y_min))
        ratio_max = max(die_wh_ratio, (x_max - x_min) / (y_max - y_min))
        if -1 <= rect_area / contour_area - 1 <= 1:
            angle = np.arcsin(rect_area / contour_area - 1) / 2
        else:
            # print(f"rect_area: {rect_area}, contour_area: {contour_area}, die: {[x_min, y_min, x_max, y_max]}")
            angle = 0
        # 弧度转角度
        angle = np.degrees(angle)
        # 超过45度的情况
        if ratio_min < 1 < ratio_max:
            angle = 90 - angle
        angles.append(angle)
    # 角度相对值
    angle_mean = np.mean(angles)
    angles -= angle_mean
    return angles

def angle_abnormal(dies, die_wh_ratio, threshold_angle=10):
    indexes_skew = []
    angles = compute_angle(dies, die_wh_ratio)
    for i, angle in enumerate(angles):
        angle = abs(angle)
        if angle > threshold_angle:
            print(f"die {i} angle: {angle: .0f}°")
            indexes_skew.append(i)
        # else:
        #     print(f"die {i} angle: {angle}")
    return indexes_skew # 角度歪斜