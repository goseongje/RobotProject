import cv2 
import time 

import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
import pafy
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def draw_bboxes(img_arr, boxes, labels, scores, classes):
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_arr, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img_arr, classes[int(label)], 
                    (x1+10, y1+10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
    return img_arr


if __name__ == "__main__":
    # load custom trained model if True
    # load COCO trained model if False
    load_custom = False
    trained_model = './results/epoch_10.tar'
    NUM_CLASS = 4
    CUSTOM_CATEGORY_NAMES = ['__background__', "coke", "green", "white"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_custom:
        # load model trained on custom dataset
        #model = models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=True, num_classes=NUM_CLASS)
        model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained_backbone=True, num_classes=NUM_CLASS)
        model.load_state_dict(torch.load(trained_model))
        model = model.to(device)  
        classes = CUSTOM_CATEGORY_NAMES
    else:
        # load model trained on COCO public dataset
        #model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        model = model.to(device)
        classes = COCO_INSTANCE_CATEGORY_NAMES

    model.eval()
    transforms_composed = transforms.Compose([
            transforms.ToTensor()])
       
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
    url = "https://www.youtube.com/watch?v=wqctLW0Hb_0"
    play = pafy.new(url).streams[-1]
    assert play is not None    

    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(play.url)
    assert cap.isOpened(), 'Cannot capture source'

    while cap.isOpened():
        start_time = time.time()

        ret, img_arr = cap.read()
    
        img_tensor = transforms_composed(img_arr).unsqueeze(0)
        outputs = model(img_tensor.to(device))[0]

        probs = outputs['scores'].cpu().detach().numpy()
        boxes = outputs["boxes"].cpu().detach().numpy()
        labels = outputs["labels"].cpu().detach().numpy()
        
        keep = probs > 0.5
        probs = probs[keep]
        labels = labels[keep]
        boxes = boxes[keep]

        draw_arr = draw_bboxes(img_arr, boxes, labels, probs, classes)

        cv2.imshow("frame", draw_arr)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        print("FPS is {:.2f}".format(1 / (time.time() - start_time)))