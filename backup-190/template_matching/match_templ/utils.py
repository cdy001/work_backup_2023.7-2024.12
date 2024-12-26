# coding=gbk
import numpy as np
import cv2

# ����ԭʼͼ����ģ��ͼ��
def load_src_templ(src_path, templ_path):
    """
    args:
        src_path: ��Ҫƥ���ԭʼͼ��·��
        templ_path: ģ��ͼ��·��
    return:
        img_src: ԭʼͼ��
        img_templ: ģ��ͼ��
    """
    img_src = cv2.imread(src_path, flags=-1)
    img_templ = cv2.imread(templ_path, flags=-1)
    return img_src, img_templ

# NMS
def nms(bounding_boxes, confidences, threshold):
    """
    Args:
        bounding_boxes: np.array([(x1, y1, x2, y2), ...])
        confidences: np.array(conf1, conf2, ...),������Ҫ��bounding boxһ��,����һһ��Ӧ
        threshold: IOU��ֵ,������bounding box�Ľ����ȴ��ڸ�ֵ�������ŶȽ�С��box���ᱻ����

    Returns:
        bounding_boxes: ����NMS���bounding boxes
        confidences: ��bounding_boxes��Ӧ�ķ���
    """
    len_bound = bounding_boxes.shape[0]
    len_conf = confidences.shape[0]
    if len_bound != len_conf:
        raise ValueError("Bounding box �� Confidence ��������һ��")
    if len_bound == 0:
        return np.array([]), np.array([])
    bounding_boxes, confidences = bounding_boxes.astype(int), np.array(confidences)

    x1, y1, x2, y2 = bounding_boxes[:, 0], bounding_boxes[:, 1], bounding_boxes[:, 2], bounding_boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(confidences)

    pick = []
    while len(idxs) > 0:
        # ��Ϊidxs�Ǵ�С�������еģ�last_idx�൱��idxs���һ��λ�õ�����
        last_idx = len(idxs) - 1
        # ȡ�����ֵ�������ϵ�����
        max_value_idx = idxs[last_idx]
        # �������ӵ���Ӧ������
        pick.append(max_value_idx)

        xx1 = np.maximum(x1[max_value_idx], x1[idxs[: last_idx]])
        yy1 = np.maximum(y1[max_value_idx], y1[idxs[: last_idx]])
        xx2 = np.minimum(x2[max_value_idx], x2[idxs[: last_idx]])
        yy2 = np.minimum(y2[max_value_idx], y2[idxs[: last_idx]])

        w, h = np.maximum(0, xx2 - xx1 + 1), np.maximum(0, yy2 - yy1 + 1)

        iou = w * h / areas[idxs[: last_idx]]
        # ɾ������value,����ɾ��iou > threshold��bounding boxes
        idxs = np.delete(idxs, np.concatenate(([last_idx], np.where(iou > threshold)[0])))

    # bounding box ����һ��Ҫint����,����Opencv�޷�����
    return np.array(bounding_boxes[pick, :]).astype(int), np.array(confidences[pick])