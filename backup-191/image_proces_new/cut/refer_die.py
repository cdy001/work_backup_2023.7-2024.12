import cv2

from cut.retrieve_contours_die import retrieve_contours


def find_dies(light_image=None, die_para=None, binary_para=None, recipe=None):
    die_height = die_para.get("die_height")
    die_width = die_para.get("die_width")

    img = light_image[die_para["refer_light"]]
    img_width = img.shape[1]
    img_height = img.shape[0]

    # return variable
    dies = []
    dies_long_vertical = []
    dies_long_horizon = []
    dies_long_big = []
    dies_rotate = []
    die_twin = []

    # 标志位
    die_flag_1 = 0
    die_flag_2 = 0
    die_flag_3 = 0
    die_flag_4 = 0
    # count
    die_nums = 0

    spec_twins_h = int(die_height * 1.6)
    spec_twins_w = int(die_width * 1.6)
    spec_h = int(die_height * 0.9)
    spec_w = int(die_width * 0.9)
    contours, remove_die_area = retrieve_contours(img, binary_para)

    for i, contorPoint in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contorPoint)
        if x < 2 or y < 2 or x + w > img_width - 2 or y + h > img_height - 2:
            continue

        # remove die
        is_remove = False
        for remove_die in remove_die_area:
            if (
                remove_die[0] < int(x + w / 2) < remove_die[2]
                and remove_die[1] < int(y + h / 2) < remove_die[3]
            ):
                is_remove = True
                break
        if is_remove:
            continue

        # 水平和垂直多胞
        if spec_twins_w < w < 5 * spec_twins_w and spec_twins_h < h < 5 * spec_twins_h:
            die_width_count = round(abs(w) / die_width)
            die_height_count = round(abs(h) / die_height)
            new_die_width = int(w / die_width_count)
            new_die_height = int(h / die_height_count)
            for j in range(die_height_count):
                new_die_left_y = y + j * new_die_height
                new_die_right_y = new_die_left_y + new_die_height
                for k in range(die_width_count):
                    new_die_left_x = x + k * new_die_width
                    new_die_right_x = new_die_left_x + new_die_width
                    new_die = [
                        int((new_die_left_x + new_die_right_x) / 2),
                        int((new_die_right_y + new_die_left_y) / 2),
                        new_die_left_x,
                        new_die_left_y,
                        new_die_right_x,
                        new_die_right_y,
                        2,
                        die_flag_2,
                        die_flag_3,
                        die_flag_4,
                    ]
                    dies_long_big.append(new_die)
                    die_nums += 1
        # 垂直双胞
        elif 2 * spec_w > w > spec_w and 5 * spec_twins_h > h > spec_twins_h:
            die_nums += 1
            die_count = round(abs(h) / die_height)
            new_die_height = int(abs(h) / die_count)
            for j in range(die_count):
                new_die_left_y = y + j * new_die_height
                new_die_right_y = new_die_left_y + new_die_height
                if die_count < 3:
                    new_die = [
                        x + int(w / 2),
                        int((new_die_left_y + new_die_right_y) / 2),
                        x,
                        new_die_left_y,
                        x + w,
                        new_die_right_y,
                        2,
                        die_flag_2,
                        die_flag_3,
                        die_flag_4,
                    ]
                else:
                    new_die = [
                        x + int(w / 2),
                        int((new_die_left_y + new_die_right_y) / 2),
                        x,
                        new_die_left_y,
                        x + w,
                        new_die_right_y,
                        2,
                        die_flag_2,
                        die_flag_3,
                        die_flag_4,
                    ]
                dies_long_vertical.append(new_die)
        # 水平多胞
        elif 2 * spec_h > h > spec_h and 5 * spec_twins_w > w > spec_twins_w:
            die_count = round(abs(w) / die_width)
            new_die_width = int(abs(w) / die_count)
            for j in range(die_count):
                new_die_left_x = x + j * new_die_width
                new_die_right_x = new_die_left_x + new_die_width
                if die_count < 3:
                    new_die = [
                        int((new_die_left_x + new_die_right_x) / 2),
                        y + int(h / 2),
                        new_die_left_x,
                        y,
                        new_die_right_x,
                        y + h,
                        2,
                        die_flag_2,
                        die_flag_3,
                        die_flag_4,
                    ]
                else:
                    new_die = [
                        int((new_die_left_x + new_die_right_x) / 2),
                        y + int(h / 2),
                        new_die_left_x,
                        y,
                        new_die_right_x,
                        y + h,
                        2,
                        die_flag_2,
                        die_flag_3,
                        die_flag_4,
                    ]
                dies_long_horizon.append(new_die)
                die_nums += 1
        # 正常的die
        elif 1.5 * spec_w > w > spec_w and 1.5 * spec_h > h > spec_h:
            tilt_angle = cv2.minAreaRect(contorPoint)[2]

            if 80 < tilt_angle:
                dies.append(
                    [
                        x + int(w / 2),
                        y + int(h / 2),
                        x,
                        y,
                        x + w,
                        y + h,
                        die_flag_1,
                        die_flag_2,
                        die_flag_3,
                        die_flag_4,
                    ]
                )
                die_nums += 1
            else:
                dies_rotate.append(
                    [
                        x + int(w / 2),
                        y + int(h / 2),
                        x,
                        y,
                        x + w,
                        y + h,
                        die_flag_1,
                        die_flag_2,
                        die_flag_3,
                        die_flag_4,
                    ]
                )
        else:
            continue

        die_twin = dies_long_vertical + dies_long_horizon + dies_long_big

    return dies, die_twin, dies_rotate
