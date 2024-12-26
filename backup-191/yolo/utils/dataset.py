# coding=gbk
import torch  
from torch.utils.data import Dataset, DataLoader  
from PIL import Image  
import os
import numpy as np
import torchvision
import glob
import cv2
  
# 定义一个自定义的Dataset类，继承自Dataset  
class ImageDataset(Dataset):  
    def __init__(self, directory, transform=None):  
        self.directory = directory  
        self.transform = transform  
        self.images = glob.glob(os.path.join(directory, "*.bmp"))  # 获取目录下所有文件名  
  
    def __len__(self):  
        return len(self.images)  # 返回数据集大小  
  
    def __getitem__(self, idx):  
        # img_path = self.images[idx]  # 文件路径  
        # img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        # image = Image.open(img_path).convert('RGB')  # 打开图像，转换为RGB格式  
        # if self.transform:  
        #     image = self.transform(image)  # 如果有定义转换，就应用转换  
        # return torch.from_numpy(np.array(image))
        return self.images[idx]
  
if __name__ == '__main__':

    transform = torchvision.transforms.Compose([  
        torchvision.transforms.Resize((160, 640)),
        torchvision.transforms.ToTensor(), 
    ])
    # 创建Dataset实例  
    dataset = ImageDataset(directory='/var/cdy_data/jucan/data/0945/0945U/Electrode_dirty', )  
    
    # 创建DataLoader实例，传入Dataset实例和批处理大小（batch_size）  
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for batch in data_loader:
        print(batch)