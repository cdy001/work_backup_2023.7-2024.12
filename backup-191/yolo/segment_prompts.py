from ultralytics import SAM, YOLO
from PIL import Image


if __name__ == "__main__":
    det_model = "models_and_labels/counter_standing.pt"
    sam_model = "pretrained/sam_b.pt"
    data = "/data/chendeyang/code/yolo/test_imgs/counter_standing/HAE2240625045735CA146-3_4.jpg"

    det_model = YOLO(model=det_model)
    sam_model = SAM(model=sam_model)


    det_results = det_model.predict(
        source=data,
        imgsz=(1024, 1024),  # image_size (h, w)
        device=[0, ],
        conf=0.25,
        )

    for i, result in enumerate(det_results):
        class_ids = result.boxes.cls.int().tolist()  # noqa
        if len(class_ids):
            boxes = result.boxes.xyxy  # Boxes object for bbox outputs
            sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=False)
            im = Image.fromarray(sam_results[0].plot(line_width=1, font_size=0.05, labels=True, conf=True))
            im.save(f"test_imgs/counter_standing/seg_{i}.jpg")