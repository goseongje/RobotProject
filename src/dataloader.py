import cv2
import numpy as np
import json
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms

class CustomDataset(Dataset):
    def __init__(self, data_root, split, CLASSES):
        self.data_root = data_root
        self.split = split  # train, test
        self.classes = CLASSES        
        self.img_list = sorted(os.listdir(os.path.join(self.data_root, 'img')))
        self.ann_list = sorted(os.listdir(os.path.join(self.data_root, 'ann')))
        assert len(self.img_list) == len(self.ann_list)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        # load RGB image
        img_name = self.img_list[idx]
        img_file = os.path.join(self.data_root, 'img', img_name)
        img = Image.open(img_file).convert("RGB")
        # transfrom from PIL.Image to Torch.Tensor
        img = self.transform(img)

        # load annotation
        ann_name = self.ann_list[idx]
        ann_file = os.path.join(self.data_root, 'ann', ann_name)
        # read json file
        ann = json.load(open(ann_file))

        # extract class id and bboxes
        labels, boxes = [], []
        for obj in ann["objects"]:
            # get label
            label = self.classes[obj['classTitle']]
            labels.append(label)
            # get bbox
            pts_x, pts_y = [], []
            for x,y in obj['points']['exterior']:
                pts_x.append(x)
                pts_y.append(y)
            x_min, x_max = min(pts_x), max(pts_x)
            y_min, y_max = min(pts_y), max(pts_y)
            boxes.append([x_min, y_min, x_max, y_max])
        # labels = np.array(labels)
        # boxes = np.array(boxes)
        # tensor data format
        labels = torch.as_tensor(labels, dtype=torch.int64)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {}
        target["image_id"] = image_id
        target["labels"] = labels
        target["boxes"] = boxes
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        return img, target
    