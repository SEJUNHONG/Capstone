from PyQt5.QtCore import QDir, Qt, QUrl , Qt, QThread, QObject, QThreadPool, pyqtSignal, pyqtSlot, QCoreApplication
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

import sys
from PyQt5 import uic
from PyQt5 import QtWidgets
from PIL import Image
import os
import shutil
import glob
from math import ceil
import make,clu,mosaic,emoji,swap,mosaic2
from src import classifier
import facenet
from models import cluster_image_save
from models import dataset_image
import torch
from facenet_pytorch import MTCNN, training

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# mtcnn = MTCNN(keep_all=True, device=device)


class VideoWindow(QMainWindow):

    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setFixedSize(900, 600)
        self.setWindowTitle("Automatic mosaic editing of videos using deep learning")
        self.setWindowIcon(QIcon('kwicon.jpg'))
        #self._mutex = QtCore.QMutex()

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer_1 = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        self.videoWidget = QVideoWidget()
        self.videoWidget_1 = QVideoWidget()

        p = self.videoWidget.palette()
        p_1 = self.videoWidget_1.palette()
        p.setColor(self.backgroundRole(), Qt.black)
        p_1.setColor(self.backgroundRole(), Qt.black)
        self.videoWidget.setPalette(p)
        self.videoWidget_1.setPalette(p_1)

        self.video_path = video_path
        self.tmp_result_name = 'tmp_result.mp4'
        self.tmp_result_path = "C:/Users/mmlab/PycharmProjects/UI_pyqt/" + self.tmp_result_name

        self.playButton = QPushButton()
        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)
        self.playButton.setStyleSheet('background:white')

        self.pauseButton=QPushButton()
        self.pauseButton.setEnabled(True)
        self.pauseButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pauseButton.clicked.connect(self.pause)
        self.pauseButton.setStyleSheet('background:white')

        self.stopButton=QPushButton()
        self.stopButton.setEnabled(True)
        self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stopButton.clicked.connect(self.stop)
        self.stopButton.setStyleSheet('background:white')

        self.openButton=QPushButton()
        self.openButton.setEnabled(True)
        self.openButton.setText("OPEN")
        self.openButton.setFixedHeight(25)
        self.openButton.setFixedWidth(50)
        self.openButton.clicked.connect(lambda : self.openFile(self.grid))
        self.openButton.setStyleSheet('background:white')

        self.saveButton=QPushButton()
        self.saveButton.setEnabled(True)
        self.saveButton.setText("SAVE")
        self.saveButton.setFixedHeight(25)
        self.saveButton.setFixedWidth(50)
        self.saveButton.clicked.connect(self.saveFile)
        self.saveButton.setStyleSheet('background:white')

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)



        # Image ScrollArea
        self.grid = QtWidgets.QGridLayout()  # Create Grid Layout
        self.bg = QtWidgets.QButtonGroup(self)  # Create Button Group

        flen = len(os.listdir('C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people'))
        self.f = []
        self.cont01 = 0
        mod = sys.modules[__name__]
        for i in range(0, flen):
            self.f.append('C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people/{}'.format(
                i) + 'human/{}'.format(i) + 'human1.png')
        a = 0
        check = []
        vBox = QtWidgets.QHBoxLayout()
        self._databases_checked = []
        positions = [(0, j) for j in range(flen)]
        for position, title in zip(positions, self.f):
            if title == '':
                continue
            QLabel = QtWidgets.QLabel()
            QLabel.setFixedWidth(150)
            QLabel.setFixedHeight(150)
            image = self.f[self.cont01]
            num = []
            num.append(self.cont01)

            self.qp = QPixmap()
            self.qp.load(image)
            self.qp = self.qp.scaled(150, 150)
            QLabel.setPixmap(self.qp)

            vBox.addWidget(QLabel)
            cb = QCheckBox(str(a))
            cb.stateChanged.connect(self.on_stateChanged)

            vBox.addWidget(cb)
            self.grid.addLayout(vBox, *position)
            self.cont01 += 1
            a += 1


        setattr
        #RUN 버튼
        self.okButton = QPushButton()
        self.okButton.setEnabled(True)
        self.okButton.clicked.connect(self.RunFile)
        self.okButton.setFixedWidth(50)
        self.okButton.setFixedHeight(25)
        self.okButton.setText("RUN")
        self.okButton.setStyleSheet('background:yellow')
        self.okButton.setFont(QFont("굴림", 10, QFont.Bold))

        # 확인버튼
       # grid2 = QtWidgets.QGridLayout()



        wid = QWidget(self)
        self.setCentralWidget(wid)
        # CREATION OF SCROLL AREA
        frame = QFrame()
        frame.setLayout(self.grid)
        scroll = QScrollArea()
        scroll.setWidget(frame)
        scroll.setWidgetResizable(False)
        scroll.setFixedHeight(210)
        self.scrollbar = scroll.verticalScrollBar()
        # Create a QVBoxLayout
        layout_scroll = QVBoxLayout(self)
        # Add the Widget QScrollArea to the QVBoxLayout
        layout_scroll.addWidget(scroll)


        # Create new action
        openAction = QAction(QIcon('open.png'), '&Open', self)
        openAction.setShortcut('Ctrl+O')
        openAction.setStatusTip('Open video')
        openAction.triggered.connect(self.openFile)

        # Create save action
        saveAction=QAction(QIcon('save.png'), '&Save', self)
        saveAction.setShortcut('Ctrl+S')
        saveAction.setStatusTip('Save video')
        saveAction.triggered.connect(self.saveFile)

        # Create exit action
        exitAction = QAction(QIcon('exit.png'), '&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(self.exitCall)



        # Create menu bar and add action
        menuBar = self.menuBar()
        fileMenu = menuBar.addMenu('&File')
        fileMenu.addAction(openAction)
        fileMenu.addAction(saveAction)
        fileMenu.addAction(exitAction)
        editMenu=menuBar.addMenu('&Edit')
        viewMenu=menuBar.addMenu('&View')
        toolMenu=menuBar.addMenu('&Tools')
        helpMenu=menuBar.addMenu('&Help')

        # Create a widget for window contents
        wid = QWidget(self)
        self.setCentralWidget(wid)

        # Create layouts to place inside widget
        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.pauseButton)
        controlLayout.addWidget(self.stopButton)
        controlLayout.addWidget(self.positionSlider)
        controlLayout.addWidget(self.openButton)
        controlLayout.addWidget(self.okButton)
        controlLayout.addWidget(self.saveButton)


        self.videoLayout = QHBoxLayout()
        self.videoLayout.setContentsMargins(0, 0, 0, 0)
        self.videoLayout.addWidget(self.videoWidget)
        self.videoLayout.addWidget(self.videoWidget_1)
        #bLayout = QHBoxLayout()
        #bLayout.setContentsMargins(0, 0, 0, 0)
        #bLayout.addWidget(self.okButton)

        layout = QVBoxLayout()
        layout.addLayout(self.videoLayout)
        layout.addLayout(controlLayout)
        layout.addLayout(layout_scroll)
        #layout.addLayout(bLayout)
        #layout.addWidget(self.errorLabel)


        # Set widget to contain window contents
        wid.setLayout(layout)

        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        # self.mediaPlayer.error.connect(self.handleError)

        self.mediaPlayer_1.setVideoOutput(self.videoWidget_1)
        self.mediaPlayer_1.positionChanged.connect(self.positionChanged)
        self.mediaPlayer_1.durationChanged.connect(self.durationChanged)
        # self.mediaPlayer_1.error.connect(self.handleError)
        fileName2 = "C:/Users/mmlab/PycharmProjects/UI_pyqt/tmp_result1.mp4"

        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.mediaPlayer_1.setMedia(QMediaContent(QUrl.fromLocalFile(fileName2)))

    def on_stateChanged(self, state):
        checkbox = self.sender()
        text = checkbox.text() + 'human'
        if state == Qt.Checked:
            self._databases_checked.append(text)
        else:
            self._databases_checked.remove(text)
        print(self._databases_checked)

    def model_image(self , grid):
        flen = len(os.listdir('C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people'))
        self.f = []
        self.cont01 = 0

        for i in range(0, flen):
            self.f.append('C:/Users/mmlab/PycharmProjects/UI_pyqt/cluster_people/{}'.format(
                i) + 'human/{}'.format(i) + 'human1.png')
        a=0
        self.check=[]
        positions = [(0, j) for j in range(flen)]
        for position, title in zip(positions, self.f):
            if title == '':
                continue
            vBox = QtWidgets.QVBoxLayout()
            QLabel = QtWidgets.QLabel()
            QLabel.setFixedWidth(150)
            QLabel.setFixedHeight(150)
            image = self.f[self.cont01]
            num = []
            num.append(self.cont01)

            self.qp = QPixmap()
            self.qp.load(image)
            self.qp = self.qp.scaled(150, 150)
            QLabel.setPixmap(self.qp)

            vBox.addWidget(QLabel)
            self.check.append(QCheckBox())


            vBox.addWidget(self.check[a])
            self.check[a].setText(str(a))
            self.check[a].clicked.connect(self.c(self.check[a]))

            grid.addLayout(vBox, *position)
            self.cont01 += 1
            a+=1

        # if self.check.isChecked():
        #     print(self.check.text())
        #     print(self.check.text())

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer_1 = QMediaPlayer(None, QMediaPlayer.VideoSurface)

        p = self.videoWidget.palette()
        p_1 = self.videoWidget_1.palette()
        p.setColor(self.backgroundRole(), Qt.black)
        p_1.setColor(self.backgroundRole(), Qt.black)
        self.videoWidget.setPalette(p)
        self.videoWidget_1.setPalette(p_1)

        self.video_path = ''
        self.tmp_result_name = 'tmp_result.mp4'
        self.tmp_result_path = "C:/Users/mmlab/PycharmProjects/UI_pyqt/" + self.tmp_result_name

        self.playButton = QPushButton()
        self.playButton.setEnabled(True)
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.clicked.connect(self.play)
        self.playButton.setStyleSheet('background:white')

        self.pauseButton = QPushButton()
        self.pauseButton.setEnabled(True)
        self.pauseButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pauseButton.clicked.connect(self.pause)
        self.pauseButton.setStyleSheet('background:white')

        self.stopButton = QPushButton()
        self.stopButton.setEnabled(True)
        self.stopButton.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stopButton.clicked.connect(self.stop)
        self.stopButton.setStyleSheet('background:white')

        self.openButton = QPushButton()
        self.openButton.setEnabled(True)
        self.openButton.setText("OPEN")
        self.openButton.setFixedHeight(25)
        self.openButton.setFixedWidth(50)
        self.openButton.clicked.connect(lambda : self.openFile( grid))
        self.openButton.setStyleSheet('background:white')

        self.saveButton = QPushButton()
        self.saveButton.setEnabled(True)
        self.saveButton.setText("SAVE")
        self.saveButton.setFixedHeight(25)
        self.saveButton.setFixedWidth(50)
        self.saveButton.clicked.connect(self.saveFile)
        self.saveButton.setStyleSheet('background:white')

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.bg = QtWidgets.QButtonGroup(self)  # Create Button Group

        self.okButton = QPushButton()
        self.okButton.setEnabled(True)
        self.okButton.clicked.connect(self.RunFile)
        self.okButton.setFixedWidth(50)
        self.okButton.setFixedHeight(25)
        self.okButton.setText("RUN")
        self.okButton.setStyleSheet('background:yellow')
        self.okButton.setFont(QFont("굴림", 10, QFont.Bold))

        wid = QWidget(self)
        self.setCentralWidget(wid)
        frame = QFrame()
        frame.setLayout(grid)
        scroll = QScrollArea()
        scroll.setWidget(frame)
        scroll.setWidgetResizable(False)
        scroll.setFixedHeight(210)
        self.scrollbar = scroll.verticalScrollBar()
        layout_scroll = QVBoxLayout(self)
        layout_scroll.addWidget(scroll)



        controlLayout = QHBoxLayout()
        controlLayout.setContentsMargins(0, 0, 0, 0)
        controlLayout.addWidget(self.playButton)
        controlLayout.addWidget(self.pauseButton)
        controlLayout.addWidget(self.stopButton)
        controlLayout.addWidget(self.positionSlider)
        controlLayout.addWidget(self.openButton)
        controlLayout.addWidget(self.okButton)
        controlLayout.addWidget(self.saveButton)

        self.videoLayout = QHBoxLayout()
        self.videoLayout.setContentsMargins(0, 0, 0, 0)
        self.videoLayout.addWidget(self.videoWidget)
        self.videoLayout.addWidget(self.videoWidget_1)

        layout = QVBoxLayout()
        layout.addLayout(self.videoLayout)
        layout.addLayout(controlLayout)
        layout.addLayout(layout_scroll)

        # Set widget to contain window contents
        wid.setLayout(layout)

        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)

        self.mediaPlayer_1.setVideoOutput(self.videoWidget_1)
        self.mediaPlayer_1.positionChanged.connect(self.positionChanged)
        self.mediaPlayer_1.durationChanged.connect(self.durationChanged)

    def openFile(self ,grid):
        if os.path.exists('C:/Users/mmlab/PycharmProjects/UI_pyqt/src/test_file'):
            for file in os.scandir('C:/Users/mmlab/PycharmProjects/UI_pyqt/src/test_file'):
                os.remove(file.path)
        if os.path.exists('C:/Users/mmlab/PycharmProjects/UI_pyqt/src/out_dir'):
            for file in os.scandir('C:/Users/mmlab/PycharmProjects/UI_pyqt/src/out_dir'):
                os.remove(file.path)
        window2 = MyWindow()
        window2.OnOpenDocument()
        video_p(window2.video_path)
        tmp_result_name = 'tmp_result.mp4'
        tmp_result_path = "C:/Users/mmlab/PycharmProjects/UI_pyqt/" + tmp_result_name

        if video_path != "":
            print(video_path)

            # make.main(video_path)
            # clu.main()
            # classifier.main()
            # mosaic2.main(video_path, tmp_result_name)
        player.close()
        player2 = VideoWindow()
        player2.show()

    def RunFile(self):
        checkdemo()

    def saveFile(self, grid):
        fileName1=QFileDialog.getSaveFileName(self, "Save Video", "*.mp4", QDir.homePath())

    def exitCall(self):
        sys.exit(app.exec_())

    def play(self):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
            self.mediaPlayer_1.pause()
        else:
            self.mediaPlayer.play()
            self.mediaPlayer_1.play()

    def pause(self):
        if self.mediaPlayer.state() == QMediaPlayer.PausedState:
            self.mediaPlayer.play()
            self.mediaPlayer_1.play()
        else:
            self.mediaPlayer.pause()
            self.mediaPlayer_1.pause()

    def stop(self):
        if self.mediaPlayer.state()==QMediaPlayer.StoppedState:
            self.mediaPlayer.play()
            self.mediaPlayer_1.play()
        else:
            self.mediaPlayer.stop()
            self.mediaPlayer_1.stop()

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

    def setPosition(self, position):
        self.mediaPlayer.setPosition(position)
        self.mediaPlayer_1.setPosition(position)

    def handleError(self):
        self.playButton.setEnabled(False)
        self.errorLabel.setText("Error: " + self.mediaPlayer.errorString())


class checkdemo(QWidget):
    def __init__(self, parent=None):
        super(checkdemo, self).__init__(parent)
        self.setGeometry(600,200,200,250)
        self.setWindowIcon(QIcon('kwicon.jpg'))

        self.mo = 0
        self.em = 0
        self.sw = 0

        layout = QGridLayout()
        self.b1 = QCheckBox("Mosaic")
        self.b1.toggled.connect(lambda: self.btnstate(self.b1, 1))
        layout.addWidget(self.b1 ,0,0)

        self.b2 = QCheckBox("Face Swap")
        self.b2.toggled.connect(lambda: self.btnstate(self.b2, 2))
        layout.addWidget(self.b2 , 1,0)

        self.b3 = QCheckBox("Face Emoticon")
        self.b3.toggled.connect(lambda: self.btnstate(self.b3, 3))
        layout.addWidget(self.b3, 2, 0)

        self.NextButton = QPushButton()
        self.NextButton.setEnabled(True)
        self.NextButton.setText("Confirm")
        self.NextButton.setFixedWidth(100)
        self.NextButton.setFixedHeight(20)
        self.NextButton.clicked.connect(lambda: self.confirm())
        self.NextButton.setStyleSheet('background:white')
        layout.addWidget(self.NextButton , 3,0)

        self.setLayout(layout)
        self.setWindowTitle("Select Function")
        self.show()

    def btnstate(self, b, a):
        if a == 1:
            if b.isChecked():
                self.mo = 1;
            else:
                self.mo = 0;
        elif a == 2:
            if b.isChecked():
                self.em = 1;
            else:
                self.em = 0;
        elif a == 3:
            if b.isChecked():
                self.sw = 1;
            else:
                self.sw = 0;

    def confirm(self):
        if (self.mo + self.em + self.sw) != 1:
            print("check one")
            print(self.mo, self.em, self.sw)

        elif self.mo == 1:
            #player.close()
            #os.remove(tmp_result_path)
            list = player._databases_checked
            mosaic.check(video_path,tmp_result_name,list)
            player.mediaPlayer_1.setMedia(QMediaContent(QUrl.fromLocalFile("C:/Users/mmlab/PycharmProjects/UI_pyqt/final_video_mosaic.avi")))
            self.close()
            #player.show()

        elif self.em == 1:
            list = player._databases_checked
            emoji.main(video_path,tmp_result_name,list)
            player.mediaPlayer_1.setMedia(
                QMediaContent(QUrl.fromLocalFile("C:/Users/mmlab/PycharmProjects/UI_pyqt/final_video_emoji.avi")))
            self.close()
            #self.mediaPlayer_1.setMedia(QMediaContent(QUrl.fromLocalFile("C:/Users/mmlab/PycharmProjects/UI_pyqt/audio_vlog2_emoji.mp4")))

        elif self.sw == 1:
            list = player._databases_checked
            swap.main(video_path,tmp_result_name,list)
            player.mediaPlayer_1.setMedia(
                QMediaContent(QUrl.fromLocalFile("C:/Users/mmlab/PycharmProjects/UI_pyqt/final_video_swap.avi")))
            self.close()
            #self.mediaPlayer_1.setMedia(QMediaContent(QUrl.fromLocalFile("C:/Users/mmlab/PycharmProjects/UI_pyqt/audio_vlog2_emoji.mp4")))




class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.video_path = ""
        self.setupUI()

    def setupUI(self):
        self.setGeometry(400, 400, 200, 100)
        self.setWindowTitle("Automatic mosaic editing of videos using deep learning")
        self.setWindowIcon(QIcon('kwicon.jpg'))

        self.OnOpenDocument_Button = QPushButton("File Open")
        self.OnOpenDocument_Button.clicked.connect(self.OnOpenDocument)
        self.label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.OnOpenDocument_Button)
        layout.addWidget(self.label)

        self.setLayout(layout)

    def OnOpenDocument(self):
        self.video_path, _ = QFileDialog.getOpenFileName(self, "Open Video", "*.mp4",QDir.homePath())
        if self.video_path:
            QCoreApplication.instance().quit

        else:
            QMessageBox.about(self, "Warning", "파일을 선택하지 않았습니다.")
            self.video_path = ""


def video_p(path):
    global video_path
    video_path = path

app = QApplication(sys.argv)

window = MyWindow()
window.OnOpenDocument()
video_path = 0
video_p(window.video_path)
tmp_result_name = 'tmp_result1.mp4'
tmp_result_path = "C:/Users/mmlab/PycharmProjects/UI_pyqt/" + tmp_result_name


if video_path != "":
    print(video_path)

    # make.main(video_path)
    # clu.main()
    # classifier.main()
    #mosaic2.main(video_path, tmp_result_name)

player = VideoWindow()
#player.resize(640, 480)
player.show()
sys.exit(app.exec_())