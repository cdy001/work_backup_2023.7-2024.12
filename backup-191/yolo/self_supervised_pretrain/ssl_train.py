import os
import sys
sys.path.append(os.getcwd())

import torch
from ultralytics import YOLO
from pytorch_metric_learning.losses import NTXentLoss

import torchvision
torchvision.disable_beta_transforms_warning()

from torch.utils.data import DataLoader

from self_supervised_pretrain.dataset import ImageFolderCustom, data_transform
from self_supervised_pretrain.ssl_model import SimYOLOv10



batch_size = 256
num_epochs = 1000
device_id = 1
trained_layers = 11

print(f'batch_size : {batch_size} num_epochs : {num_epochs}')

root = "/var/cdy_data/aoyang/data/*/*"
train_dir = os.path.join(root, "Good/")
print(f"train_dir: {train_dir}")

train_data = ImageFolderCustom(
    targ_dir=train_dir,
    transform=data_transform,
    )

print(f"Length of train_data: {len(train_data)}")

train_dataloader = DataLoader(
    dataset=train_data, 
    batch_size=batch_size, 
    num_workers=20, 
    shuffle=True
    ) 
print(f"Length of train_dataloader: {len(train_dataloader)}")

    
# InfoNCE Noise-Contrastive Estimation
loss_func = NTXentLoss(temperature=0.25)

# higher batch sizes return better results usually from 256 to 8192 etc
# for batch size 1024, we get 1022 negative samples to model contrast against within a batch + our poisitive pair

device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
assert str(device) == f'cuda:{device_id}'

model = SimYOLOv10()
# load weight
state_dict = torch.load("self_supervised_pretrain/models/0809/epoch_250.pt", map_location="cpu")
state_dict_new = dict()
for key, value in state_dict.items():
    key_new = key.replace("model.", "")
    if int(key_new.split(".")[0]) <= trained_layers:
        state_dict_new[key_new] = value
model.backbone.load_state_dict(state_dict_new)

device = torch.device(device)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)


for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, data in enumerate(train_dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        # Get data representations
        h_1, h_2, compact_h_1, compact_h_2 = model(data)
        # Prepare for loss
        embeddings = torch.cat((compact_h_1, compact_h_2), dim = 0)
        # The same index corresponds to a positive pair
        indices = torch.arange(0, compact_h_1.size(0), device=compact_h_2.device)
        labels = torch.cat((indices, indices))
        loss = loss_func(embeddings, labels)
        loss.backward()
        total_loss += loss.item() * data.size(0)  
        optimizer.step()
        if step % 10 == 9:
            print(f"Epoch {epoch:03d}/{num_epochs}, Step {step+1}/{len(train_dataloader)}, Loss: {loss.item()*data.size(0)/len(train_data):.4f}")

    loss = total_loss / len(train_data) 
    print(f'Epoch {epoch:03d}/{num_epochs}, Loss: {loss:.4f}')
    scheduler.step()
    if epoch % 50 == 49:    
        # Extracting Backbone
        backbone = model.backbone
        # print(backbone , backbone.state_dict())

        model_yolo = YOLO(model="configs/models/yolov10s-aoyang.yaml")  # build a new model from scratch
        model_children_list = list(model_yolo.model.children())
        head_layers = model_children_list[0][trained_layers:]

        full_state_dict = {**backbone.state_dict(), **head_layers.state_dict()}
        full_state_dict = {f'model.{k}': v for k, v in full_state_dict.items()}

        model_save_path = "self_supervised_pretrain/models/0811"
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(full_state_dict, os.path.join(model_save_path, f"epoch_{epoch+1}.pt"))
        print(f"ephch_{epoch+1} is saved at {model_save_path}")