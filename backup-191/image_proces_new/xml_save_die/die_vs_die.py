import os
import cv2


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    
    Parameters:
    box1, box2: List or tuple of four integers [x1, y1, x2, y2], 
                where (x1, y1) is the top-left coordinate, 
                and (x2, y2) is the bottom-right coordinate.
    
    Returns:
    float: IoU value.
    """
    # Unpack the coordinates
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate the (x, y)-coordinates of the intersection rectangle
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    # Compute the area of the intersection rectangle
    inter_width = max(0, x_inter_max - x_inter_min)
    inter_height = max(0, y_inter_max - y_inter_min)
    inter_area = inter_width * inter_height
    
    # Compute the area of both bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Compute the area of the union
    union_area = box1_area + box2_area - inter_area
    
    # Compute the IoU
    iou = inter_area / union_area
    
    return iou


# calculate area of two die
def Cal_area(rec1, rec2):
    xmin1, ymin1, xmax1, ymax1 = rec1
    xmin2, ymin2, xmax2, ymax2 = rec2
    if (xmax1 <= xmin2 or xmax2 <= xmin1) or (ymax1 <= ymin2 or ymax2 <= ymin1):
        area = 0
        return area
    else:
        lens = min(xmax2, xmax1) - max(xmin1, xmin2)
        width = min(ymax2, ymax1) - max(ymin1, ymin2)
        area = lens * width
        return area


# 比较两个die
def compare_die(xmin1, ymin1, xmax1, ymax1, die_name_l, image_type, img, dst_path, thresh, label_cut_die):
    c_num = 0
    for die_name in die_name_l:
        if image_type in die_name:
            xmin2, ymin2, xmax2, ymax2, *_ = os.path.splitext(die_name)[0].split('#')[-1].split('_')
            xmin2, ymin2, xmax2, ymax2 = int(xmin2), int(ymin2), int(xmax2), int(ymax2)
            # area = Cal_area((xmin1, ymin1, xmax1, ymax1), (xmin2, ymin2, xmax2, ymax2))
            iou = calculate_iou((xmin1, ymin1, xmax1, ymax1), (xmin2, ymin2, xmax2, ymax2))

            if iou > thresh:
                img_die = img[ymin2: ymax2, xmin2: xmax2]
                img_path = os.path.join(dst_path, die_name + '.bmp')
                cv2.imwrite(img_path, img_die)

                label_cut_die += 1

                c_num += 1
                if c_num >= 2:
                    print('number of c', c_num)
    
    return label_cut_die
