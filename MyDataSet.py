import torch
from torch.utils.data import Dataset
import cv2
import torch.utils.data
import torchvision.transforms as transforms
import os
import numpy as np
import random
import math

from utils import calculate_batch_iou

def random_perspective(img, gray, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    """combination of img transform"""
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]
    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if gray is None:
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
        return img
    
    else:
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if perspective:
                img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
                gray = cv2.warpPerspective(gray, M, dsize=(width, height), borderValue=0)
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
                gray = cv2.warpAffine(gray, M[:2], dsize=(width, height), borderValue=0)
        return img, gray

def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    if len(img.shape) == 2:  # 如果是灰度图像，直接调整亮度（Value）
        # 增加亮度
        gain = np.random.uniform(1 - vgain, 1 + vgain)
        img = np.clip(img * gain, 0, 255).astype(np.uint8)
        return img
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    
    img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    return img_bgr


class MyDataset_all(Dataset):
    def __init__(self, images_path: str, image_type1: str, image_type2: str):
        self.root = images_path
        self.image_type1 = image_type1
        self.image_type2 = image_type2
        self.Tensor = transforms.ToTensor()

        self.W = 224
        self.H = 224

        if self.image_type1 == "c":
            self.folders = ["ID_b", "ID_i"]
        elif self.image_type1 == "s":
            self.sub_folders = ["image", "ContourMask", "NoduleMask"]
        elif self.image_type1 not in ["c", "s"]:
            raise ValueError("Invalid image_type1. Should be 's', 'c'.")

        if self.image_type2 not in ["train", "test"]:
            raise ValueError("Invalid image_type2. Should be 'train', 'test'.")

    def __len__(self):
        if self.image_type1 == "c":
            total_length = 0
            for folder in self.folders:
                folder_path = os.path.join(self.root, folder)
                total_length += len(os.listdir(folder_path))
            return total_length
        elif self.image_type1 == "s":
            folder_path = os.path.join(self.root, "image")
            if not os.path.exists(folder_path):
                raise FileNotFoundError(f"Directory not found: {folder_path}")
            return len(os.listdir(folder_path))
        else:
            raise ValueError("Invalid image_type1. Should be 's', 'c'.")

    def __getitem__(self, idx):
        W = self.W
        H = self.H
        if self.image_type1 == "s":
            image_name = os.path.join(self.root, "image", os.listdir(os.path.join(self.root, "image"))[idx])
            NoduleMask_name = image_name.replace("image", "NoduleMask")
            image_cv2 = cv2.imread(image_name, 0)
            NoduleMask_cv2 = cv2.imread(NoduleMask_name, 0)

            if self.image_type2 == "train":
                if random.random() < 0.5:
                    image_cv2, NoduleMask_cv2= random_perspective(image_cv2, NoduleMask_cv2,degrees=10,translate=0.1,scale=0.25,shear=0.0)
                if random.random() < 0.5:
                    augment_hsv(image_cv2)
                if random.random() < 0.5:
                    image_cv2 = np.fliplr(image_cv2)
                    NoduleMask_cv2 = np.fliplr(NoduleMask_cv2)
            
            image = cv2.resize(image_cv2, (W, H))
            image = image[None, :, :]
            image = np.ascontiguousarray(image)
            image = torch.Tensor(image)

            _,NoduleMask = cv2.threshold(NoduleMask_cv2,1,255,cv2.THRESH_BINARY)
            NoduleMask = cv2.resize(NoduleMask, (W, H))
            NoduleMask = NoduleMask / 255.0
            NoduleMask = self.Tensor(NoduleMask)


            return image, NoduleMask

        elif self.image_type1 == "c":
            cumulative_length = 0
            for folder in self.folders:
                folder_path = os.path.join(self.root, folder)
                folder_length = len(os.listdir(folder_path))
                if idx < cumulative_length + folder_length:
                    image_name = os.path.join(folder_path, os.listdir(folder_path)[idx - cumulative_length])
                    image_cv2 = cv2.imread(image_name, 0)
                    
                    if self.image_type2 == "train":
                        if random.random() < 0.5:
                            image_cv2= random_perspective(image_cv2,None,degrees=10,translate=0.1,scale=0.25,shear=0.0)
                        if random.random() < 0.5:
                            augment_hsv(image_cv2)
                        if random.random() < 0.5:
                            image_cv2 = np.fliplr(image_cv2)

                    image = cv2.resize(image_cv2, (W, H))
                    image = image[None, :, :]
                    image = np.ascontiguousarray(image)
                    image = torch.Tensor(image)

                    label_map = {"ID_b": 0,"ID_i": 1}
                    label = label_map[os.path.basename(os.path.dirname(image_name))] 

                    return image, label
                
                cumulative_length += folder_length
