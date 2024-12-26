import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import v2



batch_size = 256
num_epochs = 1000
img_size = 224
device_id = 1

print(f'batch_size : {batch_size} num_epochs : {num_epochs} img_size : {img_size}')

class ImageFolderCustom(Dataset):
    
    
    def __init__(self, targ_dir: str, transform=None) -> None:
        # self.paths = list(sorted(Path(targ_dir).glob("*.bmp"))) 
        self.paths = list(sorted(glob.glob(os.path.join(targ_dir, "*.bmp"))))
        self.transform = transform

    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    
    def __getitem__(self, index: int):
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index) # load image

        if img.mode != "RGB":
            img = img.convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img
    
data_transform = v2.Compose([
    v2.Resize(size=(img_size, img_size)),
    v2.ToImageTensor(),
    v2.ConvertImageDtype(dtype = torch.uint8),
])


