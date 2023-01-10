from hvsnet import HVSNet, HVSNetV2
from torchvision.transforms import functional as F
from tqdm import tqdm

import torch
import cv2
import numpy as np
import argparse
import os

def listdir(path):
    filelist = []
    for filepath, _, filenames in os.walk(path):
        filelist.extend([os.path.join(filepath, filename) for filename in filenames])
    return filelist

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path')
    parser.add_argument('--output-path')
    parser.add_argument('--type', default='segmentation')
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    model_names = listdir('./saved/models')
    video_names = listdir(args.video_path)
    for model_name in model_names:
        base = os.path.basename(model_name).replace('.pt', '')
        if args.type not in base: continue

        fg_output_path = os.path.join(args.output_path, base, 'fg')
        mt_output_path = os.path.join(args.output_path, base, 'matte')

        if not os.path.exists(fg_output_path):
            os.makedirs(fg_output_path)
        if not os.path.exists(mt_output_path):
            os.makedirs(mt_output_path)

        model = HVSNet(type=args.type, mode='train')
        weights = torch.load(model_name, map_location='cpu')
        model.load_state_dict(weights, strict=True)
        model.eval()
        model.mode = 'eval'

        image = torch.randn((1, 3, 512, 512))
        model = torch.jit.trace(model, image)
        model = torch.jit.freeze(model)

        for video_name in tqdm(video_names):
            cap = cv2.VideoCapture(video_name)
            video_name = os.path.basename(video_name)

            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            fps = cap.get(cv2.CAP_PROP_FPS)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fg_writer = cv2.VideoWriter(os.path.join(fg_output_path, video_name), fourcc, fps, (W, H))
            mt_writer = cv2.VideoWriter(os.path.join(mt_output_path, video_name), fourcc, fps, (W, H))

            success = True
            while success:
                success, origin_image = cap.read()
                if not success: break

                H, W = origin_image.shape[:2]
                origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)

                image = cv2.resize(origin_image, (512, 512), cv2.INTER_AREA)
                image = F.to_tensor(image)
                image = F.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                image = image.unsqueeze(0)

                if args.type == 'segmentation':
                    output = model(image).argmax(1)
                else:
                    output = model(image)
                output = F.resize(output, [H, W])
                output = output.repeat(3, 1, 1).permute((1, 2, 0)).numpy()

                fg = origin_image * output + 255 * (1 - output)
                fg = fg.clip(0, 255).astype(np.uint8)[:, :, ::-1]
                mt = (output * 255).clip(0, 255).astype(np.uint8)

                fg_writer.write(fg)
                mt_writer.write(mt)

            cap.release()
            fg_writer.release()
            mt_writer.release()


if __name__ == '__main__':
    main()