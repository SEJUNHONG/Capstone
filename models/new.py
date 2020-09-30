from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np
import pandas as pd
from time import time
import sys, os
import glob

from models.mtcnn import MTCNN, fixed_image_standardization
from models.inception_resnet_v1 import InceptionResnetV1, get_torch_home
mtcnn = MTCNN(keep_all=True)
img = [
    Image.open('C:/Users/mmlab/PycharmProjects/facenet-pytorch-master-s/data/abc.jpg'),
    Image.open('C:/Users/mmlab/PycharmProjects/facenet-pytorch-master-s/data/abc.jpg')
]
batch_boxes, batch_probs = mtcnn.detect(img)

mtcnn(img, save_path=['C:/Users/mmlab/PycharmProjects/facenet-pytorch-master-s/data/tmp1.png', 'C:/Users/mmlab/PycharmProjects/facenet-pytorch-master-s/data/tmp1.png'])
tmp_files = glob.glob('data/tmp*')
for f in tmp_files:
    os.remove(f)
