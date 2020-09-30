from PyQt5.QtCore import QDir, Qt, QUrl
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QMainWindow, QWidget, QPushButton, QAction, QFrame, QScrollArea
from PyQt5.QtGui import *
import sys
from PyQt5 import uic
from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
import os
import glob
from math import ceil

class VideoWindow(QMainWindow):

    def __init__(self, parent=None):
        super(VideoWindow, self).__init__(parent)
        self.setWindowTitle("PyQt Video Player Widget Example")

        grid = QtWidgets.QGridLayout()  # Create Grid Layout
        self.bg = QtWidgets.QButtonGroup(self)  # Create Button Group

        dir = "C:/Users/mmlab/PycharmProjects/UI_pyqt/Kate/"  # Image direction
        self.f = sorted(glob.glob(dir + '*.jpg'))
        flen = len(self.f)
        self.titles = []
        self.cont01 = 0

        positions = [(i, j) for i in range(int(ceil(flen / 4))) for j in range(4)]
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
            # self.qp.loadFromData(image)
            self.qp.load(image)
            self.qp = self.qp.scaled(150, 150)
            QLabel.setPixmap(self.qp)

            vBox.addWidget(QLabel)

            self.check = QCheckBox()
            chk = []
            for i in range(len(num)):
                vBox.addWidget(self.check)

            grid.addLayout(vBox, *position)

            self.cont01 += 1


        wid = QWidget(self)
        self.setCentralWidget(wid)

        frame = QtWidgets.QFrame()
        # Put the GridLayout in a Widget QFrame
        frame.setLayout(grid)
        # Create a Widget QScrollArea()
        scroll = QtWidgets.QScrollArea()
        # Put the Widget QFrame in the QScrollArea()
        scroll.setWidget(frame)
        scroll.setWidgetResizable(False)
        scroll.setFixedHeight(420)
        self.scrollbar = scroll.verticalScrollBar()
        # Create a QVBoxLayout
        layout = QtWidgets.QVBoxLayout(self)
        # Add the Widget QScrollArea to the QVBoxLayout
        layout.addWidget(scroll)


        wid.setLayout(layout)

    def checkbox(self, num):
        chk = []
        for i in range(len(num)):
            chk.append(QCheckBox())

            # self.check.setGeometry(QtCore.QRect(10 + 100 * (i + 1), 250, 95, 95))
            i += 1

app = QtWidgets.QApplication(sys.argv)
main = VideoWindow()
main.show()
sys.exit(app.exec_())