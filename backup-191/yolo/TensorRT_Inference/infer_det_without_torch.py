import argparse
from pathlib import Path
import time
import cv2
import numpy as np

from models.utils import path_to_list, preprocess, non_max_suppression


def main(args: argparse.Namespace) -> None:
    if args.method == 'cudart':
        from models.cudart_api import TRTEngine
    elif args.method == 'pycuda':
        from models.pycuda_api import TRTEngine
    else:
        raise NotImplementedError

    Engine = TRTEngine(args.engine)
    H, W = 640, 640

    images = path_to_list(args.imgs)
    save_path = Path(args.out_dir)

    if not args.show and not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    time_start = time.perf_counter()
    # preprocess
    tensors, ratios, dwdhs = preprocess(images, new_shape=(W, H))
    time_preprocess_end = time.perf_counter()

    # inference
    preds = Engine(tensors)
    time_inference_end = time.perf_counter()
    
    # postprocess
    # results = det_postprocess(preds=preds)
    preds = non_max_suppression(preds)
    time_postprocess_end = time.perf_counter()

    print(f"preprocess time:{time_preprocess_end - time_start: .4f}s, inference time:{time_inference_end - time_preprocess_end: .4f}s, postprocess time:{time_postprocess_end - time_inference_end: .4f}s")

    for image, pred, ratio, dwdh in zip(images, preds, ratios, dwdhs):
        save_image = save_path / image.name
        if pred.size == 0:
            print(f"{image}: no object!")
            continue
        bboxes = pred[:, :4]
        scores = pred[:, 4]
        labels = pred[:, 5]
        # check score negative
        scores[scores < 0] = 1 + scores[scores < 0]
        bboxes -= dwdh
        bboxes /= ratio
        draw = cv2.imread(str(image))
        for (bbox, score, label) in zip(bboxes, scores, labels):
            bbox = bbox.round().astype(np.int32).tolist()
            cls_id = int(label)
            color = (0, 0, 255)
            text = f'{cls_id} {score:.3f}'
            x1, y1, x2, y2 = bbox
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw, text, (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
        if args.show:
            cv2.imshow('result', draw)
            cv2.waitKey(0)
        else:
            cv2.imwrite(str(save_image), draw)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--engine', type=str, help='Engine file', default="runs/detect/train-official-pretrain/weights/best.engine")
    parser.add_argument('--imgs', type=str, help='Images file', default="test_imgs/1Q42E")
    parser.add_argument('--show',
                        action='store_true',
                        help='Show the detection results')
    parser.add_argument('--out-dir',
                        type=str,
                        default='./output',
                        help='Path to output file')
    parser.add_argument('--method',
                        type=str,
                        default='cudart',
                        help='CUDART pipeline')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)