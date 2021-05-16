import cv2
import numpy as np
import json
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms

def imshow(img):
    for i in range(len(img)):
        tmp_img = img[i] / 2 + 0.5     # unnormalize
        npimg = tmp_img.numpy()
        plt.figure()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def collate_fn(batch):
    return tuple(zip(*batch))    