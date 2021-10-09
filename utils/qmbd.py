# -*- coding: utf-8 -*-
"""
Copyright 2021 HuHui, Inc. All Rights Reserved.
@author: HuHui
@software: PyCharm
@project: few-shot
@file: qmbd.py
@version: v1.0
@time: 2021/10/9 下午4:05
-------------------------------------------------
Description :
工程文件说明： 
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        # print("hh: size = ", size)
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img, padding_value=None):
        # print("hh: img.size = ", img.size)
        # sys.exit()
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class alignCollate(object):

    def __init__(self, imgH=28, imgW=28, padding_value=255):
        self.imgH = imgH
        self.imgW = imgW
        self.padding_value = padding_value

    def __call__(self, batch):
        # print(batch[0][0].shape, batch[0][1])

        imgH = self.imgH
        imgW = self.imgW

        new_images = []
        new_labels = []
        transform = resizeNormalize((imgW, imgH))
        for image, label in batch:
            image = np.array(image)
            _, h, w = image.shape
            if w >= h:
                new_image = np.ones((w, w), dtype=np.uint8) * self.padding_value
                padding = int((w - h) / 2)
                new_image[padding:padding + h, :] = image[0]
            else:
                new_image = np.ones((h, h), dtype=np.uint8) * self.padding_value
                padding = int((h - w) / 2)
                new_image[:, padding:padding + w] = image[0]
            # print(new_image.shape)
            new_images.append(Image.fromarray(new_image))
            new_labels.append(label)

        out_images = [transform(image, self.padding_value) for image in new_images]
        images = torch.cat([t.unsqueeze(0) for t in out_images], 0)
        labels = torch.tensor(new_labels)
        # print(images.shape, labels.shape)
        return [images, labels]

