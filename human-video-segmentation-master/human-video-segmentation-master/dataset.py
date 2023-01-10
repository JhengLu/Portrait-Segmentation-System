from torch.utils.data.dataset import Dataset, T_co
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

import cv2
import os
import numpy as np
import torch
import transforms


def random_dilate(alpha, low=1, high=5, mode='constant'):
    iterations = np.random.randint(1, 20)
    erode_ksize = np.random.randint(low=low, high=high)
    dilate_ksize = np.random.randint(low=low, high=high)
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_ksize, erode_ksize))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_ksize, dilate_ksize))
    alpha_eroded = cv2.erode(alpha, erode_kernel, iterations=iterations)
    alpha_dilated = cv2.dilate(alpha, dilate_kernel, iterations=iterations)
    if mode == 'constant':
        alpha_noise = 128 * np.ones_like(alpha)
        alpha_noise[alpha_eroded >= 255] = 255
        alpha_noise[alpha_dilated <= 0] = 0
    else:
        value = np.random.randint(low=100, high=255)
        alpha_noise = value * ((alpha_dilated - alpha_eroded) / 255.)
        alpha_noise += alpha_eroded
    return alpha_noise


def listdir(path):
    filelist = []
    for filepath, _, filenames in os.walk(path):
        filelist.extend([os.path.join(filepath, filename) for filename in filenames])
    return filelist


def getBase(path):
    return os.path.basename(path)\
        .replace('.jpg', '')\
        .replace('.png', '')\
        .replace('.jpeg', '')


def opencv_to_pillow(*args):
    if len(args) == 1:
        return Image.fromarray(args[0])
    return [Image.fromarray(image) for image in args]


def pillow_to_opencv(*args):
    if len(args) == 1:
        return np.asarray(args[0])
    return [np.asarray(image) for image in args]


class HumanSegmentationDataset(Dataset):
    def __init__(self, root: str, size=(512, 512), mode='train',
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 dilate_high=5):
        super(HumanSegmentationDataset, self).__init__()
        assert mode in ('train', 'eval')

        self.size = size
        self.mode = mode
        self.inter_size = (int(size[0] * 1.5), int(size[1] * 1.5))
        self.dilate_high = dilate_high

        image_path = os.path.join(root, 'image')
        matte_path = os.path.join(root, 'matte')

        self.image_names = listdir(image_path)
        self.matte_names = listdir(matte_path)
        self.image_names.sort()
        self.matte_names.sort()

        self.to_tensor = T.Compose([
            T.ToTensor(),
            T.Normalize(mean, std)
        ])
        self.crop = transforms.RandomCrop(size[0])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        matte_name = self.matte_names[idx]
        if getBase(image_name) != getBase(matte_name):
            print(image_name, matte_name)
        assert getBase(image_name) == getBase(matte_name)

        image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
        matte = cv2.imread(matte_name, cv2.IMREAD_GRAYSCALE)

        if self.mode == 'train':
            return self._preprocessing_train(image, matte)
        return self._preprocessing_eval(image, matte)

    def _resize_and_crop(self, image, matte):
        if np.random.rand() < 0.2:
            image = cv2.resize(image, self.size, cv2.INTER_AREA)
            matte = cv2.resize(matte, self.size, cv2.INTER_AREA)
        else:
            image = cv2.resize(image, self.inter_size, cv2.INTER_AREA)
            matte = cv2.resize(matte, self.inter_size, cv2.INTER_AREA)
            image, matte = opencv_to_pillow(image, matte)
            image, matte = self.crop(image, matte)
            image, matte = pillow_to_opencv(image, matte)
        return image, matte

    def _preprocessing_train(self, image, matte):
        image, matte = self._resize_and_crop(image, matte)
        trimap = random_dilate(matte, high=self.dilate_high)

        label = matte.copy()
        label[label <= 127] = 0
        label[label > 127] = 255

        label_edge = cv2.Laplacian(label, cv2.CV_8U, ksize=5)
        label_bak = label_edge.copy()
        label_edge[label == 0] = 0
        label_edge[label == 255] = 1
        label_edge[label_bak == 255] = 2

        label[label == 255] = 1

        image = self.to_tensor(image)
        label = torch.from_numpy(label).to(dtype=torch.long)
        label_edge = torch.from_numpy(label_edge).to(dtype=torch.long)
        trimap = torch.from_numpy(trimap)

        return image, label, label_edge, trimap

    def _preprocessing_eval(self, image, matte):
        image = cv2.resize(image, self.size, cv2.INTER_AREA)
        matte = cv2.resize(matte, self.size, cv2.INTER_AREA)

        label = matte.copy()
        label[label <= 127] = 0
        label[label > 127] = 1
        image = self.to_tensor(image)
        label = torch.from_numpy(label).to(dtype=torch.long)
        return image, label


class HumanMattingDataset(HumanSegmentationDataset):
    def __init__(self, root: str, size=(512, 512), mode='train',
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 dilate_high=5):
        super(HumanMattingDataset, self).__init__(root, size, mode, mean, std, dilate_high)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        matte_name = self.matte_names[idx]
        if getBase(image_name) != getBase(matte_name):
            print(image_name, matte_name)
        assert getBase(image_name) == getBase(matte_name)

        image = cv2.cvtColor(cv2.imread(image_name), cv2.COLOR_BGR2RGB)
        matte = cv2.imread(matte_name, cv2.IMREAD_GRAYSCALE)

        if self.mode == 'train':
            return self._preprocessing_train(image, matte)
        return self._preprocessing_eval(image, matte)

    def _preprocessing_train(self, image, matte):
        image, matte = self._resize_and_crop(image, matte)
        trimap = random_dilate(matte, high=self.dilate_high)

        image = self.to_tensor(image)
        matte = matte / 255
        matte = torch.from_numpy(matte)
        trimap = torch.from_numpy(trimap)
        return image, matte, trimap
