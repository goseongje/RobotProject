import cv2
import numpy as np
import json
import os
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.models as models
import torchvision.transforms as transforms

from dataloader import CustomDataset
from utils import imshow, collate_fn

if __name__ == "__main__":
    # number of class for detection
    # including the background as 0
    # if model classifies three classes, num_class = 4
    NUM_CLASS = 4
    CLASSES = {
        "coke": 1,
        "green": 2,
        "white": 3
    }
    output_folder = './results'
    os.makedirs(output_folder, exist_ok=True)

    # hyper parameter
    BATCH_SIZE = 2
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.00005
    MAX_EPOCH = 20
    SAVE_INTERVAL = 5 # saving per epoch

    # get device (GPU or CPU)
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # load model
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=NUM_CLASS)
    model = model.to(device)
    model.train()
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # load dataset
    data_root = "./data/robotics_trash" # path/to/dataset
    train_dataset = CustomDataset(data_root, 'train', CLASSES)
    train_loader = torch.utils.data.DataLoader(
                            dataset=train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, collate_fn=collate_fn)

    for epoch in range(MAX_EPOCH):
        print("[Epoch {}]".format(epoch+1))
        loss_sum = 0
        for idx, (images, targets) in enumerate(train_loader):
            # get data
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # forward
            loss_dict = model(images, targets)
            losses = loss_dict['loss_classifier'] + loss_dict['loss_box_reg'] \
                     + loss_dict['loss_objectness'] + loss_dict['loss_rpn_box_reg'] 
            
            # backporp
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # logging 
            loss_sum += losses
            print("Epoch [{}/{}] mini-batch [{}/{}] loss {:.5f} (avg {:.5f})".format(
                    epoch+1, MAX_EPOCH, idx, len(train_loader), losses, loss_sum/(idx+1)))
                    
        if (epoch +1) % SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), '{}/epoch_{}.tar'.format(output_folder, epoch+1))
