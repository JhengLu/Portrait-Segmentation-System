import cv2

from models import *
from models import _make_divisible, _init_weight
from hvsnet import *
from torchvision.models import *
from dataset import *
from train_eval_utils import evaluate

import unittest
import time
import thop

import torch
import numpy as np


def _test_time(f, *args, **kwargs):
    def wrapper():
        start = time.time()
        ret = f(*args, **kwargs)
        end = time.time()
        print(f"time used: {(end - start) * 1000}ms")
        return ret

    return wrapper


def _batch_test_time(loops, f, *args, **kwargs):
    def wrapper():
        start = time.time()
        for i in range(loops):
            f(*args, **kwargs)
        end = time.time()
        print(f"time used per loop: {(end - start) / loops * 1000}ms")

    return wrapper


class MyTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(MyTestCase, self).__init__(*args, **kwargs)
        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # self.size = (256, 256)
        self.size = (512, 512)

        # model = HVSNet(num_classes=2, mode='train', stem='mobilenet_v3')
        self.model = HVSNet(num_classes=2, mode='train')
        # self.model = HVSNet(num_classes=2, mode='train', type='matting')
        # self.model = HVSNetV2(type='matting')

        # weights = torch.load('./saved/models/model_29_matting_v2.pt', map_location='cpu')
        # weights = torch.load('./saved/models/model_45_matting_v1.pt', map_location='cpu')
        weights = torch.load('./saved/models/model_45_segmentation_v1.pt', map_location='cpu')
        self.model.load_state_dict(weights)
        self.model.eval()
        self.model.mode = 'eval'

        image = torch.randn((1, 3, 512, 512))
        model = torch.jit.trace(self.model, image)
        model.save('test.torchscript')


    def test_model(self):
        image = torch.randn((1, 3, 512, 512))
        # model = HVSNet(num_classes=2, mode='eval', stem='mobilenet_v3')
        # model = HVSNet(num_classes=2, mode='eval')
        model = HVSNet(num_classes=2, mode='eval', type='matting')
        # model = HVSNetV2(type='matting')
        model.eval()
        flops, params = thop.profile(model, (image,))
        print(flops / 1e9, params / 1e6)

        model = torch.jit.trace(model, image)
        model = torch.jit.freeze(model)

        # _batch_test_time(1000, lambda x: model(x).max(1), image)()

    def test_model_video(self):
        model = self.model
        cap = cv2.VideoCapture('videos/test02.mp4')
        # cap = cv2.VideoCapture(0)
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        # writer = cv2.VideoWriter('videos/me_seg.mp4', fourcc, fps, (W, H))

        success, image = cap.read()
        while success and cv2.waitKey(1) == -1:
            image = cv2.resize(image, self.size, cv2.INTER_AREA)
            origin_image = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.to_tensor(image).unsqueeze(0)

            output = model(image)
            if model.type == 'segmentation':
                label = output.argmax(1) \
                    .repeat(3, 1, 1) \
                    .to(dtype=torch.uint8) \
                    .permute((1, 2, 0)) \
                    .numpy()
                origin_image *= label
                origin_image[label == 0] = 255

            elif model.type == 'matting':
                matte = output.repeat(3, 1, 1).permute((1, 2, 0)).detach().numpy()
                origin_image = origin_image * matte + 255 * (1 - matte)
                origin_image = np.clip(origin_image, 0, 255)
                origin_image = origin_image.astype(np.uint8)

                _matte = (matte * 255).clip(0, 255).astype(np.uint8)

            origin_image = cv2.resize(origin_image, (512, 512), cv2.INTER_AREA)
            # origin_image = cv2.resize(origin_image, (W, H), cv2.INTER_AREA)
            cv2.imshow('test', origin_image)
            # writer.write(origin_image)
            success, image = cap.read()


    def test_model_image(self):
        model = self.model

        for i in range(14, 22 + 1):
            image = cv2.imread(f'images/test{i:02d}.png')
        # for i in range(1, 18 + 1):
        #     image = cv2.imread(f'images/test{i:02d}.jpg')
            H, W = image.shape[:2]
            image = cv2.resize(image, self.size, cv2.INTER_AREA)
            origin_image = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.to_tensor(image).unsqueeze(0)

            output = model(image)
            if model.type == 'segmentation':
                label = output.argmax(1) \
                    .repeat(3, 1, 1) \
                    .to(dtype=torch.uint8) \
                    .permute((1, 2, 0)) \
                    .numpy()
                origin_image *= label
                origin_image[label == 0] = 255

            elif model.type == 'matting':
                matte = output.repeat(3, 1, 1).permute((1, 2, 0)).detach().numpy()
                origin_image = origin_image * matte + 255 * (1 - matte)
                origin_image = np.clip(origin_image, 0, 255)
                origin_image = origin_image.astype(np.uint8)

            # origin_image = cv2.resize(origin_image, (512, 512), cv2.INTER_CUBIC)
            origin_image = cv2.resize(origin_image, (W, H), cv2.INTER_CUBIC)
            cv2.imwrite(f'images/seg/test{i:02d}.jpg', origin_image)
            cv2.imshow('test', origin_image)
            cv2.waitKey()


    def test_make_divisible(self):
        for i in range(1, 33):
            print(i, _make_divisible(i))

    def test_catBottleneck(self):
        x = torch.randn((1, 128, 64, 64))
        model = CatBottleneck(128, 256, True)
        x = _test_time(lambda x: model(x), x)()
        print(x.size())

    def test_catConvX(self):
        x = torch.randn((1, 64, 512, 512))
        model = CatConvX(64, 64, False)
        x = model(x)
        print(x.size())

    def test_dataset(self):
        path = r"D:\Documents\Python\CV\Datasets\Human-Matting\train"
        dataset = HumanSegmentationDataset(path)
        for image, label, label_edge, trimap in dataset:
            label_edge = label_edge.to(dtype=torch.uint8).numpy()
            label_edge[label_edge == 1] = 255
            label_edge[label_edge == 2] = 128
            cv2.imshow('test', label_edge)
            cv2.waitKey(0)
            break

    def test_matting_dataset(self):
        path = r"D:\Documents\Python\CV\Datasets\Human-Matting\train"
        dataset = HumanMattingDataset(path, mode='eval')
        for image, matte, trimap in dataset:
            matte *= 255
            matte = matte.to(dtype=torch.uint8).numpy()
            cv2.imshow('test', matte)
            cv2.waitKey()
            break

    def test_trimap(self):
        matte = cv2.imread('images/matte01.jpg', cv2.IMREAD_GRAYSCALE)
        trimap = random_dilate(matte, 1, 5)
        cv2.imshow('test', trimap)
        cv2.waitKey(0)

    def test_crop(self):
        image = cv2.imread('images/test01.jpg')
        image = opencv_to_pillow(image)
        image = T.RandomCrop(512, 512)(image)
        image = pillow_to_opencv(image)
        cv2.imshow('test', image)
        cv2.waitKey(0)

    def test_interpolate(self):
        image = cv2.imread('images/test01.jpg')
        image = cv2.resize(image, (128, 128), cv2.INTER_AREA)
        image = cv2.resize(image, (1024, 1024), cv2.INTER_AREA)
        cv2.imshow('test', image)
        cv2.waitKey()

    def test_miou(self):
        root = r'D:\Documents\Python\CV\Datasets\mini_supervisely'
        val_dataset = HumanSegmentationDataset(root, mode='eval')
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=4, pin_memory=True)
        confmat = evaluate(self.model, val_loader, device='cpu', num_classes=2, matting=False)
        print(str(confmat))

if __name__ == '__main__':
    unittest.main()
