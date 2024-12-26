from torch import nn
import torchvision
from torchvision.transforms import v2
from ultralytics import YOLO


augmentation = v2.Compose([
    v2.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
    #v2.RandomEqualize(p=1.0),
    torchvision.transforms.functional.equalize,
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.RandomRotation(degrees=90),
    v2.RandomAffine(degrees=0, translate=(0, 0.1)),
    v2.GaussianBlur(kernel_size=(9, 9)),
    v2.ConvertImageDtype(),
])

trained_layers = 11 

model = YOLO(model="configs/models/yolov10s-aoyang.yaml")  # build a new model from scratch
model_children_list = list(model.model.children())
backbone = model_children_list[0][:trained_layers]

# Defining Model
class SimYOLOv10(nn.Module):
    def __init__(self):
        super().__init__()
        # Feature extraction
        self.backbone = backbone
        # Projection head
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # This will make the output (batch_size, 512, 1, 1)
            nn.Flatten(),  # This will make the output (batch_size, 512)
            nn.Linear(512, 256),
        )

    def forward(self, x, train=True):
        if train:
            # Get 2 augmentations of the batch
            augm_1 = augmentation(x)
            augm_2 = augmentation(x)

            # Get representations for first augmented view
            h_1 = self.backbone(augm_1)

            # Get representations for second augmented view
            h_2 = self.backbone(augm_2)
        else:
            h = self.backbone(x)
            return h

        # Transformation for loss function
        compact_h_1 = self.mlp(h_1)
        compact_h_2 = self.mlp(h_2)
        return h_1, h_2, compact_h_1, compact_h_2