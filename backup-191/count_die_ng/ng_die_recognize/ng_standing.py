import os
import glob
import cv2
import time
import numpy as np
import onnxruntime as ort

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    im = img.copy()
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, agnostic=False):
    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    
    max_nms = 30000
    max_wh = 7680
    max_det = 300    

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = prediction.shape[1] - 4  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = np.amax(prediction[:, 4:mi], axis=1) > conf_thres  # candidates

    if prediction.shape[-1] == 6:  # end-to-end model (BNC, i.e. 1,300,6)
        # output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        prediction = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        output = [np.zeros(shape=(0, 6 + nm))] * bs
        for xi, x in enumerate(prediction):
            scores = x[:, 4]  # scores
            boxes = x[:, :4]  # boxes (offset by class)
            # boxes from xyxy to xywh
            for box in boxes:
                box[2] -= box[0]
                box[3] -= box[1]
            indexes = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)  # NMS
            indexes = indexes[:max_det]  # limit detections
            # boxes from xywh to xyxy
            for box in boxes:
                box[2] += box[0]
                box[3] += box[1]
            output[xi] = x[indexes]
        output = prediction
        return output

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)

    t = time.time()
    output = [np.zeros(shape=(0, 6 + nm))] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = np.concatenate((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

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

        # # Experimental
        # merge = False  # use merge-NMS
        # if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
        #     # Update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
        #     from .metrics import box_iou
        #     iou = box_iou(boxes[i], boxes) > iou_thres  # IoU matrix
        #     weights = iou * scores[None]  # box weights
        #     x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
        #     redundant = True  # require redundant detections
        #     if redundant:
        #         i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]

    return output


class OnnxYolo:
    def __init__(self, path_weight, device, conf=0.25, iou=0.7):
        self.path_weight = path_weight
        self.device = device
        self.conf = conf
        self.iou = iou

        providers = ort.get_all_providers()
        if 'CUDAExecutionProvider' in providers and self.device != 'cpu':
            sess_options = ort.SessionOptions()
            self.sess = ort.InferenceSession(
                self.path_weight,
                providers=[("CUDAExecutionProvider", {'device_id': self.device}), ('CPUExecutionProvider')],
                sess_options=sess_options
            )
        else:
            self.sess = ort.InferenceSession(
                self.path_weight,
                providers=['CPUExecutionProvider']
            )

    def predict(self, im0, input_shapes=(640, 640)):
        input_names = self.sess.get_inputs()[0].name
        output_names = self.sess.get_outputs()[0].name
        # input_shapes = self.sess.get_inputs()[0].shape

        im1, ratio, dwdh = letterbox(im0, new_shape=input_shapes[-2:])

        im2 = im1.transpose((2, 0, 1))
        im2 = np.expand_dims(im2, 0)
        im2 = np.ascontiguousarray(im2)
        im2 = im2.astype(np.float32)
        im2 /= 255
        # print(im2.shape)
        # print(type(im2))

        pred_onnx = self.sess.run([output_names], {input_names: im2})[0]
        pred = non_max_suppression(pred_onnx, self.conf, self.iou)[0]

        boxes = pred[:, :4]
        idxs = pred[:, 5]
        scores = pred[:, 4]
        boxes -= np.array(dwdh * 2)
        boxes /= ratio
        boxes[boxes < 0] = 0

        results = np.concatenate((boxes, scores[:, np.newaxis], idxs[:, np.newaxis]), axis=1)
        # results = results[np.argsort(results[:, 0])]
        results = results[np.argsort(-results[:, -2])]
        return results
    

def inferencePerPatch(model, path_img, save_img=True):
    time_start = time.perf_counter()
    # model = OnnxYolo("ng_die_recognize/onnx_model/yolov10-0925.onnx", 0, conf=0.25)
    # path_imgs = glob.glob(os.path.join("/var/cdy_data/aoyang/counter_img/0922-scatter_standing/HAEE240919025703BA051_BPB0J49C_1745.JPG"))
    
    sub_img_height, sub_img_width = 1280, 1280
    sub_num_height, sub_num_width = 8, 8

    save_root_path, img_name = os.path.split(path_img)
    save_path = os.path.join(save_root_path, img_name.split("_")[0])
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    time_start_img = time.perf_counter()
    results_all = []
    img = cv2.imread(path_img, flags=0)
    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_height, img_width = img.shape[:2]
    gap_height = sub_img_height - (sub_num_height * sub_img_height - img_height) // (sub_num_height - 1)
    gap_width = sub_img_width - (sub_num_width * sub_img_width - img_width) // (sub_num_width - 1)
    for i in range(sub_num_height):
        for j in range(sub_num_width):
            img_i_j = img[i*gap_height:i*gap_height+sub_img_height, j*gap_width:j*gap_width+sub_img_width]
            img_i_j = cv2.cvtColor(img_i_j, cv2.COLOR_GRAY2BGR)
            results = model.predict(img_i_j, input_shapes=(sub_img_height, sub_img_width))
            results = [result for result in results if int(result[5]) == 1]  # 过滤，仅保留standing芯粒
            for result in results:
                result[0] += j * gap_width
                result[1] += i * gap_height
                result[2] += j * gap_width
                result[3] += i * gap_height
            results_all.extend(results)
    results_all = np.array(results_all)
    output = []
    if results_all.shape[0] > 0:
        img_plot = None
        scores = results_all[:, 4]  # scores
        boxes = results_all[:, :4]  # boxes
        # boxes from xyxy to xywh
        for box in boxes:
            box[2] -= box[0]
            box[3] -= box[1]
        indexes = cv2.dnn.NMSBoxes(boxes, scores, model.conf, model.iou)  # NMS
        # boxes from xywh to xyxy
        for box in boxes:
            box[2] += box[0]
            box[3] += box[1]
        output = results_all[indexes]
        # output = results_all

        for arr in output:
            xmin, ymin, xmax, ymax = map(int, arr[:4])
            img_plot = cv2.rectangle(
                img_display,
                (xmin, ymin),
                (xmax, ymax),
                (255, 0, 0),
                3
            )
            cv2.putText(img_display, f"{int(arr[5])} {arr[4]:.2f}", (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
            print(f"boxes:,{[xmin, ymin,xmax, ymax]} ids: {int(arr[-1])}, score: {round(arr[-2], 4)}")
        
        if img_plot is not None and save_img:
            cv2.imwrite(os.path.join(save_path, "yolo_result.jpg"), cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
    else:
        print(f"no objects were detected")
    print(f"time yolo detect: {time.perf_counter() - time_start_img: .4f}")

    return output



if __name__ == "__main__":
    model_path = "ng_die_recognize/onnx_model/yolov10-0925.onnx"
    device = 0    
    img_paths = glob.glob(os.path.join("/var/cdy_data/aoyang/counter_img/0922-scatter_standing/HAEE240919025703BA051_BPB0J49C_1745.JPG"))
    model = OnnxYolo(model_path, device)
    for i in range(10):
        for img_path in img_paths:
            output = inferencePerPatch(model, img_path)
            print(output)