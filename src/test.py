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
    trained_model = './results/epoch_20.tar'

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
    model.load_state_dict(torch.load(trained_model))
    model = model.to(device)
    model.eval()

    # load dataset
    data_root = "./data/robotics_trash" # path/to/dataset
    test_dataset = CustomDataset(data_root, CLASSES)
    test_loader = torch.utils.data.DataLoader(
                            dataset=test_dataset, batch_size=1, 
                            shuffle=False, collate_fn=collate_fn)
                            
    coco = get_coco_api_from_dataset(test_loader.dataset)
    coco_evaluator = CocoEvaluator(coco, iou_types=["bbox"])

    for idx, (images, targets) in enumerate(test_loader):
        # get data
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # forward
        outputs = model(images)
        outputs = [{k: v.to(torch.device("cpu")) for k, v in t.items()} for t in outputs]
        results = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(results)

    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()