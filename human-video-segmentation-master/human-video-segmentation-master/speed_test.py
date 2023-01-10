from hvsnet import HVSNet, HVSNetV2
from tqdm import tqdm
from torchvision.transforms import functional as F

import torch
import cv2
import numpy as np
import argparse
import time
import thop

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', type=int, default=1)
    parser.add_argument('--type', type=str, default='segmentation')
    parser.add_argument('--inference-only', type=bool)
    parser.add_argument('--print-flops', type=bool)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.version == 1:
        model = HVSNet(num_classes=2, mode='eval', type=args.type)
    else:
        model = HVSNetV2(args.type)
    model.eval()

    image = torch.randn((1, 3, 512, 512))
    if args.print_flops:
        flops, params = thop.profile(model, (image,))
        print(f"flops: {flops / 1e9}G, params: {params / 1e6}M")

    model = torch.jit.trace(model, image)
    model = torch.jit.freeze(model)


    if args.inference_only:
        print('inference only...')
        for _ in tqdm(range(10)):
            output = model(image)
    else:
        print('processing time included...')
        origin_image = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)
        for _ in tqdm(range(10)):
            H, W = origin_image.shape[:2]
            image = origin_image[:, :, ::-1]
            image = cv2.resize(image, (512, 512), cv2.INTER_AREA)
            image = F.to_tensor(image)
            image = F.normalize(image, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            image = image.unsqueeze(0)
            if args.type == 'segmentation':
                output = model(image).squeeze(0).argmax(0)
            else:
                output = model(image).squeeze(0).clamp(0, 1)
            output = output.to(dtype=torch.uint8).numpy()
            output = cv2.resize(output, (W, H), cv2.INTER_AREA)

if __name__ == '__main__':
    main()