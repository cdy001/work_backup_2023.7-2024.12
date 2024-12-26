def calculate_coordinates(die_left_x, die_left_y, die_right_x, die_right_y, margin_x, margin_y, img_width, img_height):
    die_left_x = die_left_x - margin_x if die_left_x - margin_x > 0 else 0
    die_left_y = die_left_y - margin_y if die_left_y - margin_y > 0 else 0
    die_right_x = die_right_x + margin_x if die_right_x + margin_x < img_width else img_width - 1
    die_right_y = die_right_y + margin_y if die_right_y + margin_y < img_height else img_height - 1

    return die_left_x, die_left_y, die_right_x, die_right_y


def calculate_coordinates_L1(die_left_x, die_left_y, die_right_x, die_right_y, R_C, L1_offset, margin_x, margin_y,
                             img_width, img_height):
    r_num = int(R_C.split("R")[-1].split("C")[0])
    if r_num % 2 == 0:
        offset_y = -L1_offset
    else:
        offset_y = L1_offset

    if die_left_x - margin_x > 0:
        die_left_x = die_left_x - margin_x
    else:
        die_left_x = 0
    if die_left_y + offset_y - margin_y > 0:
        die_left_y = die_left_y + offset_y - margin_y
    else:
        die_left_y = 0

    if die_right_x + margin_x < img_width:
        die_right_x = die_right_x + margin_x
    else:
        die_right_x = img_width - 1

    if die_right_y + offset_y + margin_y < img_height:
        die_right_y = die_right_y + offset_y + margin_y
    else:
        die_right_y = img_height - 1

    return die_left_x, die_left_y, die_right_x, die_right_y


def calculate_coordinates_L1_06(die_left_x, die_left_y, die_right_x, die_right_y, L1_offset_x, L1_offset_y, margin_x,
                                margin_y):
    if die_left_x - L1_offset_x - margin_x > 0:
        die_left_x = die_left_x - L1_offset_x - margin_x
    else:
        die_left_x = 0

    if die_left_y - L1_offset_y - margin_y > 0:
        die_left_y = die_left_y - L1_offset_y - margin_y
    else:
        die_left_y = 0

    if die_right_x - L1_offset_x + margin_x < 5120:
        die_right_x = die_right_x - L1_offset_x + margin_x
    else:
        die_right_x = 5119

    if die_right_y - L1_offset_y + margin_y < 5120:
        die_right_y = die_right_y - L1_offset_y + margin_y
    else:
        die_right_y = 5119

    return die_left_x, die_left_y, die_right_x, die_right_y
