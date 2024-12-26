# coding=gbk
import numpy as np
import cv2

# 加载原始图像与模板图像
def load_src_templ(src_path, templ_path):
    """
    args:
        src_path: 需要匹配的原始图像路径
        templ_path: 模板图像路径
    return:
        img_src: 原始图像
        img_templ: 模板图像
    """
    img_src = cv2.imread(src_path, flags=-1)
    img_templ = cv2.imread(templ_path, flags=-1)
    return img_src, img_templ

# NMS
def nms(bounding_boxes, confidences, threshold):
    """
    Args:
        bounding_boxes: np.array([(x1, y1, x2, y2), ...])
        confidences: np.array(conf1, conf2, ...),数量需要与bounding box一致,并且一一对应
        threshold: IOU阀值,若两个bounding box的交并比大于该值，则置信度较小的box将会被抑制

    Returns:
        bounding_boxes: 经过NMS后的bounding boxes
        confidences: 与bounding_boxes对应的分数
    """
    len_bound = bounding_boxes.shape[0]
    len_conf = confidences.shape[0]
    if len_bound != len_conf:
        raise ValueError("Bounding box 与 Confidence 的数量不一致")
    if len_bound == 0:
        return np.array([]), np.array([])
    bounding_boxes, confidences = bounding_boxes.astype(int), np.array(confidences)

    x1, y1, x2, y2 = bounding_boxes[:, 0], bounding_boxes[:, 1], bounding_boxes[:, 2], bounding_boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(confidences)

    pick = []
    while len(idxs) > 0:
        # 因为idxs是从小到大排列的，last_idx相当于idxs最后一个位置的索引
        last_idx = len(idxs) - 1
        # 取出最大值在数组上的索引
        max_value_idx = idxs[last_idx]
        # 将这个添加到相应索引上
        pick.append(max_value_idx)

        xx1 = np.maximum(x1[max_value_idx], x1[idxs[: last_idx]])
        yy1 = np.maximum(y1[max_value_idx], y1[idxs[: last_idx]])
        xx2 = np.minimum(x2[max_value_idx], x2[idxs[: last_idx]])
        yy2 = np.minimum(y2[max_value_idx], y2[idxs[: last_idx]])

        w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)

        iou = w * h / areas[idxs[: last_idx]]
        # 删除最大的value,并且删除iou > threshold的bounding boxes
        idxs = np.delete(idxs, np.concatenate(([last_idx], np.where(iou > threshold)[0])))

    # bounding box 返回一定要int类型,否则Opencv无法绘制
    return np.array(bounding_boxes[pick, :]).astype(int), np.array(confidences[pick])