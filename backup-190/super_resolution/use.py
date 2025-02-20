

import argparse
from pickle import TRUE
from numpy.lib.function_base import average
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from model import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
import cv2

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file',default='model/best.pth', type=str)
    parser.add_argument('--image', type=str, default='low_pixels.png')
    parser.add_argument('--scale', type=int, default=1)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image).convert('RGB')
    #image = cv2.imread(args.image).convert('RGB')
    #print(image.shape)
    
    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)
    #image.save(args.image.replace('.', '_bicubic_x{}.'.format(args.scale)))
    print('finish BICUBIC')
    
    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    new_images = []
    psnr = []
    for i in range(3):
        y = ycbcr[..., i]
        y /= 255.
        y = torch.from_numpy(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            preds = model(y).clamp(0.0, 1.0)
        
        psnr.append(calc_psnr(y, preds).cpu())
        
        preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)
        new_images.append(preds)
    

    print('PSNR: {:.2f}'.format(average(psnr)))
    
    #output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.array([new_images[0], new_images[1], new_images[2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    output.save(args.image.replace('.', '_srcnn_x{}.'.format(args.scale)))
    print('successfully save the image')