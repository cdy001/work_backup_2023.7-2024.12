from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from numpy import ndarray

import time
import torch

# image suffixs
SUFFIXS = ('.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff',
           '.webp', '.pfm')

# angle scale
ANGLE_SCALE = 1 / np.pi * 180.0


def letterbox(im: ndarray,
              new_shape: Union[Tuple, List] = (640, 640),
              color: Union[Tuple, List] = (114, 114, 114)) \
        -> Tuple[ndarray, float, Tuple[float, float]]:
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # new_shape: [width, height]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])
    # Compute padding [width, height]
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[
        1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im,
                            top,
                            bottom,
                            left,
                            right,
                            cv2.BORDER_CONSTANT,
                            value=color)  # add border
    return im, r, (dw, dh)


def blob(im: ndarray, return_seg: bool = False) -> Union[ndarray, Tuple]:
    seg = None
    if return_seg:
        seg = im.astype(np.float32) / 255
    im = im.transpose([2, 0, 1])  # HWC to CHW
    im = np.ascontiguousarray(im) / 255
    if return_seg:
        return im, seg
    else:
        return im


def sigmoid(x: ndarray) -> ndarray:
    return 1. / (1. + np.exp(-x))


def path_to_list(images_path: Union[str, Path]) -> List:
    if isinstance(images_path, str):
        images_path = Path(images_path)
    assert images_path.exists()
    if images_path.is_dir():
        images = [
            i.absolute() for i in images_path.iterdir() if i.suffix in SUFFIXS
        ]
    else:
        assert images_path.suffix in SUFFIXS
        images = [images_path.absolute()]
    return images


def crop_mask(masks: ndarray, bboxes: ndarray) -> ndarray:
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(bboxes[:, :, None], [1, 2, 3],
                              1)  # x1 shape(1,1,n)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    c = np.arange(h, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def box_iou(box1: ndarray, box2: ndarray) -> float:
    x11, y11, x21, y21 = box1
    x12, y12, x22, y22 = box2
    x1 = max(x11, x12)
    y1 = max(y11, y12)
    x2 = min(x21, x22)
    y2 = min(y21, y22)
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = (x21 - x11) * (y21 - y11) + (x22 - x12) * (y22 -
                                                            y12) - inter_area
    return max(0, inter_area / union_area)


def NMSBoxes(boxes: ndarray,
             scores: ndarray,
             labels: ndarray,
             iou_thres: float,
             agnostic: bool = False):
    num_boxes = boxes.shape[0]
    order = np.argsort(scores)[::-1]
    boxes = boxes[order]
    labels = labels[order]

    indices = []

    for i in range(num_boxes):
        box_a = boxes[i]
        label_a = labels[i]
        keep = True
        for j in indices:
            box_b = boxes[j]
            label_b = labels[j]
            if not agnostic and label_a != label_b:
                continue
            if box_iou(box_a, box_b) > iou_thres:
                keep = False
        if keep:
            indices.append(i)

    indices = np.array(indices, dtype=np.int32)
    return order[indices]


def xywh2xyxy(x):
    """
    Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
    top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.

    Args:
        x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.

    Returns:
        y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
    """
    assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
    y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
    xy = x[..., :2]  # centers
    wh = x[..., 2:] / 2  # half width-height
    y[..., :2] = xy - wh  # top left xy
    y[..., 2:] = xy + wh  # bottom right xy
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int, optional): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    t = time.time()
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == classes).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        boxes = x[:, :4] + c  # boxes (offset by class)
        i = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def preprocess(images, new_shape=(640, 640)):
    W, H = new_shape
    tensors = []
    ratios = []
    dwdhs = []
    for image in images:
        bgr = cv2.imread(str(image))
        bgr, ratio, dwdh = letterbox(bgr, (W, H))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor = blob(rgb, return_seg=False)
        dwdh = np.array(dwdh * 2, dtype=np.float32)
        # tensor = np.ascontiguousarray(tensor).astype(np.float32)
        ratios.append(ratio)
        dwdhs.append(dwdh)
        tensors.append(tensor)
    tensors = np.ascontiguousarray(tensors).astype(np.float32)
    return tensors, ratios, dwdhs


# def det_postprocess(preds,conf_thr=0.25, iou_thr=0.45, max_det=300):
#     preds = non_max_suppression(
#         prediction=preds,
#         conf_thres=conf_thr,
#         iou_thres=iou_thr,
#         max_det=max_det,
#     )
#     results = {}
#     for i, pred in enumerate(preds):
#         bboxes = pred[:, :4]
#         scores = pred[:, 4]
#         labels = pred[:, 5]
#         # check score negative
#         scores[scores < 0] = 1 + scores[scores < 0]
#         results[i] = (bboxes, scores, labels)
    
#     return results


def seg_postprocess(
        data: Tuple[ndarray],
        shape: Union[Tuple, List],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    assert len(data) == 2
    h, w = shape[0] // 4, shape[1] // 4  # 4x downsampling
    outputs, proto = (i[0] for i in data)
    bboxes, scores, labels, maskconf = np.split(outputs, [4, 5, 6], 1)
    scores, labels = scores.squeeze(), labels.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return np.empty((0, 4), dtype=np.float32), \
            np.empty((0,), dtype=np.float32), \
            np.empty((0,), dtype=np.int32), \
            np.empty((0, 0, 0, 0), dtype=np.int32)

    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    cvbboxes = np.concatenate([bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]],
                              1)
    labels = labels.astype(np.int32)
    v0, v1 = map(int, (cv2.__version__).split('.')[:2])
    assert v0 == 4, 'OpenCV version is wrong'
    if v1 > 6:
        idx = cv2.dnn.NMSBoxesBatched(cvbboxes, scores, labels, conf_thres,
                                      iou_thres)
    else:
        idx = cv2.dnn.NMSBoxes(cvbboxes, scores, conf_thres, iou_thres)
    bboxes, scores, labels, maskconf = \
        bboxes[idx], scores[idx], labels[idx], maskconf[idx]
    masks = sigmoid(maskconf @ proto).reshape(-1, h, w)
    masks = crop_mask(masks, bboxes / 4.)
    masks = masks.transpose([1, 2, 0])
    masks = cv2.resize(masks, (shape[1], shape[0]),
                       interpolation=cv2.INTER_LINEAR)
    masks = masks.transpose(2, 0, 1)
    masks = np.ascontiguousarray((masks > 0.5)[..., None], dtype=np.float32)
    return bboxes, scores, labels, masks


def pose_postprocess(
        data: Union[Tuple, ndarray],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[ndarray, ndarray, ndarray]:
    if isinstance(data, tuple):
        assert len(data) == 1
        data = data[0]
    outputs = np.transpose(data[0], (1, 0))
    bboxes, scores, kpts = np.split(outputs, [4, 5], 1)
    scores, kpts = scores.squeeze(), kpts.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no bounding boxes or seg were created
        return np.empty((0, 4), dtype=np.float32), np.empty(
            (0, ), dtype=np.float32), np.empty((0, 0, 0), dtype=np.float32)
    bboxes, scores, kpts = bboxes[idx], scores[idx], kpts[idx]
    xycenter, wh = np.split(bboxes, [
        2,
    ], -1)
    cvbboxes = np.concatenate([xycenter - 0.5 * wh, wh], -1)
    idx = cv2.dnn.NMSBoxes(cvbboxes, scores, conf_thres, iou_thres)
    cvbboxes, scores, kpts = cvbboxes[idx], scores[idx], kpts[idx]
    cvbboxes[:, 2:] += cvbboxes[:, :2]
    return cvbboxes, scores, kpts.reshape(idx.shape[0], -1, 3)


def obb_postprocess(
        data: Union[Tuple, ndarray],
        conf_thres: float = 0.25,
        iou_thres: float = 0.65) \
        -> Tuple[ndarray, ndarray, ndarray]:
    if isinstance(data, tuple):
        assert len(data) == 1
        data = data[0]
    outputs = np.transpose(data[0], (1, 0))
    num_cls = outputs.shape[-1] - 5
    bboxes, scores, angles = np.split(outputs, [4, num_cls + 4], 1)
    scores, labels = scores.max(-1), scores.argmax(-1)
    scores, labels, angles = scores.squeeze(), labels.squeeze(
    ), angles.squeeze()
    idx = scores > conf_thres
    if not idx.any():  # no obbs were created
        return np.empty((0, 4, 2), dtype=np.float32), np.empty(
            (0, ), dtype=np.float32), np.empty((0, ), dtype=np.int32)
    bboxes, scores, labels, angles = bboxes[idx], scores[idx], labels[
        idx], angles[idx] * ANGLE_SCALE
    cvrbboxes = [[(xc, yc), (w, h), a]
                 for (xc, yc, w, h), a in zip(bboxes, angles)]
    idx = cv2.dnn.NMSBoxesRotated(cvrbboxes, scores, conf_thres, iou_thres)
    points = np.array([cv2.boxPoints(cvrbboxes[i]) for i in idx],
                      dtype=np.float32)
    return points, scores, labels