from ultralytics import SAM
from PIL import Image

def segment_prompt():
    # Load a model
    model = SAM("pretrained/sam2_b.pt")

    # Display model information (optional)
    model.info()

    # Run inference with bboxes prompt
    # results = model("test_imgs/1Q42E/R00C03_6_L1_1721609030.bmp", bboxes=[426, 316, 467, 357])
    # Run inference with points
    results = model("test_imgs/1Q42E/R00C03_6_L1_1721609030.bmp", points=[[73, 126],], labels=[1])
    # results = model("test_imgs/1Q42E/R00C03_6_L1_1721609030.bmp")
    print(results[0].masks.xyn)
    im = Image.fromarray(results[0].plot())
    im.save('seg_result.bmp')


if __name__ == "__main__":
    # segment_prompt()

    import torch
    weight = "pretrained/yolov10s.pt"
    model = torch.load(weight)
    print(type(model))
    print(model.keys())
    model_2 = model["model"]
    print(type(model_2))