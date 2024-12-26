from ultralytics import YOLO
import torch

model = YOLO(model="configs/models/yolov10s-aoyang.yaml")

# model = YOLO(model='configs/models/yolov10s-aoyang.yaml').load(weights='pretrained/yolov10s.pt')

state_dict = torch.load("self_supervised_pretrain/models/0811/epoch_350.pt", map_location="cpu")
# print(dir(state_dict["model"]))    # 查看属性
print(type(state_dict))
# model.model.load_state_dict(state_dict)

# for key, val in state_dict["model"].state_dict().items():
#     print(key, val)

# for item in state_dict.items():
#     print(item)

# key, val = list(model.state_dict().items())[0]
# print(key)
# for key, val in model.model.state_dict().items():
#     print(key)

new_state_dict = {f'model.{k}': v for k, v in state_dict.items()}
torch.save(new_state_dict, f"yolov10_pretrained_custom.pt")