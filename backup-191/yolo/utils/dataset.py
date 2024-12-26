# coding=gbk
import torch  
from torch.utils.data import Dataset, DataLoader  
from PIL import Image  
import os
import numpy as np
import torchvision
import glob
import cv2
  
# ����һ���Զ����Dataset�࣬�̳���Dataset  
class ImageDataset(Dataset):  
    def __init__(self, directory, transform=None):  
        self.directory = directory  
        self.transform = transform  
        self.images = glob.glob(os.path.join(directory, "*.bmp"))  # ��ȡĿ¼�������ļ���  
  
    def __len__(self):  
        return len(self.images)  # �������ݼ���С  
  
    def __getitem__(self, idx):  
        # img_path = self.images[idx]  # �ļ�·��  
        # img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
        # image = Image.open(img_path).convert('RGB')  # ��ͼ��ת��ΪRGB��ʽ  
        # if self.transform:  
        #     image = self.transform(image)  # ����ж���ת������Ӧ��ת��  
        # return torch.from_numpy(np.array(image))
        return self.images[idx]
  
if __name__ == '__main__':

    transform = torchvision.transforms.Compose([  
        torchvision.transforms.Resize((160, 640)),
        torchvision.transforms.ToTensor(), 
    ])
    # ����Datasetʵ��  
    dataset = ImageDataset(directory='/var/cdy_data/jucan/data/0945/0945U/Electrode_dirty', )  
    
    # ����DataLoaderʵ��������Datasetʵ�����������С��batch_size��  
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for batch in data_loader:
        print(batch)