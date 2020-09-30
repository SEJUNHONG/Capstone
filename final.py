# MIT License
#
# Copyright (c) 2017 PXL University College
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Clusters similar faces from input folder together in folders based on euclidean distance matrix

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import clu
import make
import src.classifier
import mosaic

#//////////////////////////////dataset만들기
make.main()
# 이전에 한게 있다면 src/out_dir과 src/test_file지우기
# 1. make.py가서 17줄의 동영상 이름 바꾸기
# 2. make.py의 49,50줄의 저장 경로 바꾸기







#/////////////////////////////clustering하기
if __name__ == '__main__':
    project_root_path = os.path.join(os.path.abspath(__file__),"C:/Users/mmlab/PycharmProjects/UI_pyqt/")
        # Feel free to replace this and use actual commandline args instead, the main method will still work
    args = lambda: None
    args.data_dir = project_root_path + 'src/test_file'
    args.model = project_root_path + '20180402-114759/20180402-114759.pb'
    args.out_dir = project_root_path + 'cluster_people'
    args.largest_cluster_only=False
    args.image_size=160
    args.margin=44
    args.min_cluster_size=3
    args.cluster_threshold=0.67
    args.gpu_memory_fraction=1.0
    clu.main(args)

# 1. 이전에 한게 있다면 cluster_people가서 지우기
# 2. 동영상별로 args.cluster_threshold=0.65 바꾸기 수동으로(보통은 0.65가 가장 적절한 경우가 많다.)
# 3. args.min_cluster_size=8 이 부분은 최소 클러스터 사이즈로 이 이하로 클러스터링되는 경우 (예외적으로 잘못 찾아낸 얼굴 등에 대해서 지우는 코드 이다.)-특별한 경우 아니면 바꾸지 않아도됨



#///////////////////////////train하기
if __name__ == '__main__':

    # Project root path to make it easier to reference files
    project_root_path = os.path.join(os.path.abspath(__file__), "C:/Users/mmlab/PycharmProjects/UI_pyqt/")

    # Feel free to replace this and use actual commandline args instead, the main method will still work
    args = lambda: None
    args.data_dir = project_root_path + 'cluster_people'
    args.seed = None
    args.use_split_dataset = False
    args.model = project_root_path + '20180402-114759/20180402-114759.pb'
    args.mode = 'TRAIN'
    args.batch_size = 460
    args.image_size = 160
    args.classifier_filename = project_root_path + 'trained_classifier/video_new_name_test4.pkl'
    src.classifier.main(args)
# 1. args.data_dir = project_root_path + 'cluster_people' 클러스터링된 폴더를 학습하는 과정으로 학습하고자 하는 폴더명 적기 -특별히 경로를 바꾸지 않는 이상 바꾸지 않아도됨
# 2. args.classifier_filename = project_root_path + 'trained_classifier/video_new_name_test.pkl' 학습모델을 pkl파일로 저장하는 것으로 해당 학습파일 이름이 뒤의 mosaic.py의 66번줄과 이름이 동일해야함



#////////////////////////mosaic실행
if __name__ == "__main__":
    args = lambda : None
    args.video = True
    args.youtube_video_url = ''
    args.video_speedup = 3
    args.webcam = False
    mosaic.main(args)

# 1. mosaic.py의 65번째 줄과 해당 final.py의 85번째 줄과 이름이 같은지 확인
# 2. make.py에서 dataset을 만든 동영상의 이름과 mosaic.py의 95번째줄의 이름과 같은지 확인(96번째 불에서 ,mp4를 붙이기 때문에 .mp4빼고 동영상 이름만 쓰기)
# 3. mosaic.py의 106번째 줄에서 결과를 동영상으로 저장할때 원하는 이름 설정