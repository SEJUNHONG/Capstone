import numpy as np
import os
import mmcv,cv2
from PIL import Image, ImageDraw
from IPython import display
import glob
from facenet_pytorch import MTCNN, training
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import time

from facenet_pytorch import InceptionResnetV1
location = []
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
current_path = os.getcwd()
mtcnn = MTCNN(keep_all=True, device=device)

def save_data(file):

    video = mmcv.VideoReader(file)
    video.cvt2frames('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/out_dir')
    frames_1 = glob.glob('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/out_dir/*.jpg')

    image=glob.glob('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/out_dir/*.jpg')
    print(len(image))
    frames= [

        Image.open('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/out_dir/000000.jpg'),
        Image.open('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/out_dir/000000.jpg')
    ]

    for i in range(1,len(frames_1)):
        frames.append(Image.open(image[i]))

    frames_tracked = []
    frames_tracked_no=[]
    for i, frame in enumerate(frames):
        print('\rTracking frame: {}'.format(i + 1), end='')

        # Detect faces
        boxes, _ = mtcnn.detect(frame)

        # Draw faces
        frame_draw = frame.copy()
        frame_draw_1=frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        if boxes is not None:
            for box in boxes:
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

                #for a in box:
                 #   location.append(a)

        # Add to frame list
            frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
            frames_tracked_no.append(frame_draw_1.resize((640, 360), Image.BILINEAR))
            mtcnn(frames_tracked_no[i], save_path=['C:/Users/mmlab/PycharmProjects/UI_pyqt/models/test_file/tm{}.jpg'.format(i+1),
                              'C:/Users/mmlab/PycharmProjects/UI_pyqt/models/test_file/tm{}.jpg'.format(i+1)])
            tmp_files = glob.glob('data/tm*')

            for f in tmp_files:
                os.remove(f)
        else:
            frames_tracked.append(frame.resize((640, 360), Image.BILINEAR))
            frames_tracked_no.append(frame.resize((640, 360), Image.BILINEAR))

    print('\nDone')
    #for i in range(12):
     #   print(location[i])

    dim = frames_tracked[0].size
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    video_tracked = cv2.VideoWriter('C:/Users/mmlab/PycharmProjects/UI_pyqt/models/video_tracked.mp4', fourcc, 25.0, dim)
    for frame in frames_tracked:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()


