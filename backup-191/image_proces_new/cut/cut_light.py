import cv2 as cv

from cut.offset_calculation_L1 import calculation_l1_offset
from cut.calculate_coordinates import calculate_coordinates
from cut.calculate_coordinates import (
    calculate_coordinates_L1,
    calculate_coordinates_L1_06,
)
from cut.judge_die import process_image, is_die
from cut.special_defects import process_image2, specific_defect_uv, specific_defect_sy


# 将图片与图片名压入列表
def append_list(
    img,
    x1,
    y1,
    x2,
    y2,
    i_i,
    R_C,
    light,
    die_flag_1,
    die_flag_2,
    die_flag_3,
    die_flag_4,
    resize_x,
    resize_y,
    die_name_l,
    die_mat_l,
):
    name_die = (
        R_C
        + "_"
        + str(i_i)
        + "_"
        + light
        + "#"
        + str(x1)
        + "_"
        + str(y1)
        + "_"
        + str(x2)
        + "_"
        + str(y2)
        + "_"
        + str(die_flag_1)
        + "_"
        + str(die_flag_2)
        + "_"
        + str(die_flag_3)
        + "_"
        + str(die_flag_4)
    )
    img_die = img[y1:y2, x1:x2]

    img_die = cv.resize(img_die, (resize_x, resize_y))

    die_name_l.append(name_die)
    die_mat_l.append(img_die)


def cut_die_light(
    light_image=None,
    light_list=None,
    refer_dies=None,
    die_para=None,
    model_para=None,
    recipe=None,
):
    margin_x = die_para.get("margin_x")
    margin_y = die_para.get("margin_y")
    L1_offset = die_para.get("L1_offset")[1]
    resize_x = model_para.get("resize_x")
    resize_y = model_para.get("resize_y")

    die_name_single = []
    die_mat_single = []

    # if len(light_image.get(light_list[0]).shape) == 2:
    #     img_height, img_width = light_image.get(light_list[0]).shape
    # else:
    #     img_height, img_width, _ = light_image.get(light_list[0]).shape
    img_height, img_width = light_image.get(light_list[0]).shape[:2]

    R_C = light_image.get("R_C")

    # 06产品l1光源偏移严重
    l1_offset_new = [0, 0]
    if "S-06" in recipe:
        if "C00" in R_C:
            _, electrode_die_img_L4 = cv.threshold(
                light_image["L4"], 185, 255, cv.THRESH_BINARY
            )
            l1_offset_new = calculation_l1_offset(
                refer_dies, electrode_die_img_L4, light_image["L1"]
            )

    # 是否需要判断die的产品
    binary_img, is_judge = process_image(light_image, recipe)

    for i, die in enumerate(refer_dies):
        die_left_x_s = die[2]
        die_left_y_s = die[3]
        die_right_x_s = die[4]
        die_right_y_s = die[5]
        die_flag_1 = die[6]
        die_flag_2 = die[7]
        die_flag_3 = die[8]
        die_flag_4 = die[9]

        # 判断是否为die
        if is_judge:
            is_continue = is_die(
                binary_img,
                light_image,
                die_left_x_s,
                die_left_y_s,
                die_right_x_s,
                die_right_y_s,
                recipe,
            )
            if not is_continue:
                continue

        # 除l1的其它光源坐标
        x1_n, y1_n, x2_n, y2_n = calculate_coordinates(
            die_left_x_s,
            die_left_y_s,
            die_right_x_s,
            die_right_y_s,
            margin_x,
            margin_y,
            img_width,
            img_height,
        )

        # 针对l1光源， 还有06产品
        if "S-06" in recipe:
            if l1_offset_new[0] == 0 and l1_offset_new[1] == 0:
                x1_n_l1, y1_n_l1, x2_n_l1, y2_n_l1 = calculate_coordinates_L1(
                    die_left_x_s,
                    die_left_y_s,
                    die_right_x_s,
                    die_right_y_s,
                    R_C,
                    L1_offset,
                    margin_x,
                    margin_y,
                    img_width,
                    img_height,
                )
            else:
                x1_n_l1, y1_n_l1, x2_n_l1, y2_n_l1 = calculate_coordinates_L1_06(
                    die_left_x_s,
                    die_left_y_s,
                    die_right_x_s,
                    die_right_y_s,
                    l1_offset_new[0],
                    l1_offset_new[1],
                    margin_x,
                    margin_y,
                )
        else:
            x1_n_l1, y1_n_l1, x2_n_l1, y2_n_l1 = calculate_coordinates_L1(
                die_left_x_s,
                die_left_y_s,
                die_right_x_s,
                die_right_y_s,
                R_C,
                L1_offset,
                margin_x,
                margin_y,
                img_width,
                img_height,
            )

        for light in light_list:

            img = light_image[light]
            if light == "L1":
                append_list(
                    img,
                    x1_n_l1,
                    y1_n_l1,
                    x2_n_l1,
                    y2_n_l1,
                    i,
                    R_C,
                    light,
                    die_flag_1,
                    die_flag_2,
                    die_flag_3,
                    die_flag_4,
                    resize_x,
                    resize_y,
                    die_name_single,
                    die_mat_single,
                )
            else:
                append_list(
                    img,
                    x1_n,
                    y1_n,
                    x2_n,
                    y2_n,
                    i,
                    R_C,
                    light,
                    die_flag_1,
                    die_flag_2,
                    die_flag_3,
                    die_flag_4,
                    resize_x,
                    resize_y,
                    die_name_single,
                    die_mat_single,
                )

    return die_name_single, die_mat_single


# # 将图片与图片名压入列表
# def append_list(img, x1, y1, x2, y2, i_i, R_C, light, die_flag_1, die_flag_2, die_flag_3, die_flag_4, resize_x,
#                 resize_y, die_name_l, die_mat_l, template, recipe):
#     img_die = img[y1:y2, x1:x2]

#     # 特殊缺陷卡控
#     if "0945" in recipe:
#         if light == "IMAGE2":
#             template, is_ok = process_image2(img_die)
#             if not is_ok:
#                 die_flag_2 = 28
#             else:
#                 die_flag_2 = 0
#         if light == "IMAGE3":
#             is_d_t = specific_defect_sy(img_die, template)
#             if not is_d_t:
#                 die_flag_2 = 28
#             else:
#                 die_flag_2 = 0
#         if light == "IMAGE1":
#             is_d_t = specific_defect_uv(img_die, template)
#             if not is_d_t:
#                 die_flag_2 = 28
#             else:
#                 die_flag_2 = 0
#         if light == "IMAGE4":
#             die_flag_2 = 0

#     name_die = (R_C + "_" + str(i_i) + "_" + light + "#" + str(x1) + "_" + str(y1) + "_" + str(x2) + "_" + str(
#         y2) + "_" + str(die_flag_1) + "_" + str(die_flag_2) + "_" + str(die_flag_3) + "_" + str(die_flag_4))

#     img_die = cv.resize(img_die, (resize_x, resize_y))

#     die_name_l.append(name_die)
#     die_mat_l.append(img_die)

#     return template

# def cut_die_light(light_image=None, light_list=None, refer_dies=None, die_para=None, model_para=None, recipe=None):
#     margin_x = die_para.get("margin_x")
#     margin_y = die_para.get("margin_y")
#     L1_offset = die_para.get("L1_offset")[1]
#     resize_x = model_para.get("resize_x")
#     resize_y = model_para.get("resize_y")

#     die_name_single = []
#     die_mat_single = []

#     if len(light_image.get(light_list[0]).shape) == 2:
#         img_height, img_width = light_image.get(light_list[0]).shape
#     else:
#         img_height, img_width, _ = light_image.get(light_list[0]).shape

#     R_C = light_image.get("R_C")

#     # 06产品l1光源偏移严重
#     l1_offset_new = [0, 0]
#     if "S-06" in recipe:
#         if "C00" in R_C:
#             _, electrode_die_img_L4 = cv.threshold(
#                 light_image["L4"], 185, 255, cv.THRESH_BINARY
#             )
#             l1_offset_new = calculation_l1_offset(
#                 refer_dies, electrode_die_img_L4, light_image["L1"]
#             )

#     # 是否需要判断die的产品
#     binary_img, is_judge = process_image(light_image, recipe)

#     for i, die in enumerate(refer_dies):
#         die_left_x_s = die[2]
#         die_left_y_s = die[3]
#         die_right_x_s = die[4]
#         die_right_y_s = die[5]
#         die_flag_1 = die[6]
#         die_flag_2 = die[7]
#         die_flag_3 = die[8]
#         die_flag_4 = die[9]

#         # 判断是否为die
#         if is_judge:
#             is_continue = is_die(binary_img, light_image, die_left_x_s, die_left_y_s, die_right_x_s, die_right_y_s,
#                                  recipe)
#             if not is_continue:
#                 continue

#         # 除l1的其它光源坐标
#         x1_n, y1_n, x2_n, y2_n = calculate_coordinates(die_left_x_s, die_left_y_s, die_right_x_s, die_right_y_s,
#                                                        margin_x, margin_y, img_width, img_height)

#         # 针对l1光源， 还有06产品
#         if "S-06" in recipe:
#             if l1_offset_new[0] == 0 and l1_offset_new[1] == 0:
#                 x1_n_l1, y1_n_l1, x2_n_l1, y2_n_l1 = calculate_coordinates_L1(die_left_x_s, die_left_y_s, die_right_x_s,
#                                                                               die_right_y_s, R_C, L1_offset, margin_x,
#                                                                               margin_y, img_width, img_height)
#             else:
#                 x1_n_l1, y1_n_l1, x2_n_l1, y2_n_l1 = calculate_coordinates_L1_06(die_left_x_s, die_left_y_s,
#                                                                                  die_right_x_s, die_right_y_s,
#                                                                                  l1_offset_new[0], l1_offset_new[1],
#                                                                                  margin_x, margin_y)
#         else:
#             x1_n_l1, y1_n_l1, x2_n_l1, y2_n_l1 = calculate_coordinates_L1(die_left_x_s, die_left_y_s, die_right_x_s,
#                                                                           die_right_y_s, R_C, L1_offset, margin_x,
#                                                                           margin_y, img_width, img_height)
#         template = None
#         light_list.remove("IMAGE2")
#         light_list.insert(0, "IMAGE2")
#         for light_n in range(len(light_list)):
#             light = light_list[light_n]
#             img = light_image[light]
#             if light == "L1":
#                 template = append_list(img, x1_n_l1, y1_n_l1, x2_n_l1, y2_n_l1, i, R_C, light, die_flag_1, die_flag_2,
#                                        die_flag_3, die_flag_4, resize_x, resize_y, die_name_single, die_mat_single,
#                                        template, recipe)
#             else:
#                 template = append_list(img, x1_n, y1_n, x2_n, y2_n, i, R_C, light, die_flag_1, die_flag_2, die_flag_3,
#                                        die_flag_4, resize_x, resize_y, die_name_single, die_mat_single, template,
#                                        recipe)

#     return die_name_single, die_mat_single