from pathlib import Path  # python 3.5+
import sys
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot
import glob
import re
from math import ceil
import time
import mutagen.mp3  # Install python3-mutagen
from mutagen.id3 import ID3
from config import *  # This way you can use global variables from config.py directly
from pygame import \
    mixer  # In order to install PYGAME PYTHON3 see README.md and https://www.pygame.org/wiki/GettingStarted#Debian/Ubuntu/Mint

mixer.init()  # initializing the mixer


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle('PyQt5 player')
        self.move(300, 60)
        self.form_widget = FormWidget(self)
        self.setCentralWidget(self.form_widget)
        self.setFixedSize(670, 565)


class FormWidget(QtWidgets.QWidget):
    def __init__(self, parent):
        super(FormWidget, self).__init__(parent)
        grid = QtWidgets.QGridLayout()  # Create Grid Layout
        self.bg = QtWidgets.QButtonGroup(self)  # Create Button Group
        self.qp2 = QtGui.QPixmap("noartwork.png")
        homedir = str(Path.home())  # Python 3.5+
        fullmp3dir = "C:/Users/mmlab/PycharmProjects/UI_pyqt/music/" # homedir + mp3dir  # With expansion tilde is expanded to "/home/username/dir_name/"
        self.f = sorted(glob.glob(fullmp3dir + '*.mp3'))
        flen = len(self.f)
        self.titles = []
        self.cont01 = 0
        self.cont02 = 0
        self.k = 8
        self.firsttime = True
        self.flag01 = True
        self.paused = False
        self.playing = True
        self.frombg = False
        self.multipleofeight = 0

        # Create a number of Vertical Box Layouts corresponding to the number of MP3 files in your music directory
        positions = [(i, j) for i in range(int(ceil(flen / 4))) for j in range(4)]
        for position, title in zip(positions, self.f):
            if title == '':
                continue
            vBox = QtWidgets.QVBoxLayout()
            QLabel = QtWidgets.QLabel()
            QLabel.setFixedHeight(150)
            mp3 = self.f[self.cont01]
            infomp3 = self.apic_extract(mp3)
            image_data = infomp3[0]
            if image_data == None:
                self.qp = QtGui.QPixmap("noartwork_bg.png")
            else:
                self.qp = QPixmap()
                self.qp.loadFromData(image_data)
            self.qp = self.qp.scaled(145, 145, Qt.KeepAspectRatio, Qt.FastTransformation)
            QLabel.setPixmap(self.qp)
            # Add a QLabel with cover artwork to the QVBoxLayout
            vBox.addWidget(QLabel)
            # If meta tags Author and Title are empty, the text on the button will be the name of the MP3
            # split in two
            if infomp3[1] == "" and infomp3[2] == "":
                mp3 = re.split(r'[/.]', mp3)
                mp3 = mp3[4]
                firstpart, secondpart = mp3[:len(mp3) // 2], mp3[len(mp3) // 2:]
                self.name = firstpart + "\n" + secondpart
            # Otherwise the text on the button will be the content of the tag Author and of the tag Title
            else:
                self.name = infomp3[1] + "\n" + infomp3[2]
            self.titles.append(self.name)
            self.button = QtWidgets.QPushButton(self.name)
            self.button.setFlat(True)
            self.button.setFixedWidth(150)
            self.button.setCheckable(True)
            # Button has been set to checked programmatically and consequentely changes style
            self.button.setStyleSheet("QPushButton:checked { background-color: #2CA7F8; color: white; border: none}")
            self.bg.addButton(self.button)
            self.bg.setId(self.button, self.cont01)
            # Add a QPushButton to the QVBoxLayout
            vBox.addWidget(self.button)
            # Add the QVBoxLayout to the QGridLayout
            grid.addLayout(vBox, *position)

            self.bg.buttonClicked['QAbstractButton *'].connect(self.on_button_clicked)
            self.bg.buttonClicked['int'].connect(self.on_button_clicked)
            self.cont01 += 1
            #### E N D  of creation of grid with buttons and labels corresponding to MP3s

        #### CREATION OF SCROLL AREA
        # How to put the grid with buttons and labels in a QScrollArea?
        # Create a Widget QFrame
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
        #### E N D  of creation of the scroll area

        # PLAYER BOX
        PlayerBox = QtWidgets.QHBoxLayout()

        self.qp2 = self.qp2.scaled(100, 100, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.coverlabel = QtWidgets.QLabel()
        self.coverlabel.setFixedWidth(150)
        self.coverlabel.setPixmap(self.qp2)
        PlayerBox.addWidget(self.coverlabel)

        self.titlelabel = QtWidgets.QLabel()
        self.titlelabel.setStyleSheet("QLabel { font-size: 14pt; text-align: left}")
        PlayerBox.addWidget(self.titlelabel)

        self.playbutton = QtWidgets.QPushButton()
        self.playbutton.setFlat(True)
        self.playbutton.setFixedWidth(60)
        self.playbutton.setCheckable(True)
        self.pauseicon = QtGui.QIcon("pauseicon.png")
        self.playicon = QtGui.QIcon("playicon.png")
        self.playbutton.setIcon(self.pauseicon)
        self.playbutton.setIconSize(QtCore.QSize(50, 50))
        self.playbutton.setStyleSheet("QPushButton { image-align: right; border: none}")
        self.playbutton.clicked.connect(self.playbuttonHandler)
        self.playbutton.setEnabled(False)  # At app start the button is disabled
        PlayerBox.addWidget(self.playbutton)

        self.skipbutton = QtWidgets.QPushButton()
        self.skipbutton.setFlat(True)
        self.skipbutton.setFixedWidth(60)
        skipicon = QtGui.QIcon("skipicon.png")
        self.skipbutton.setIcon(skipicon)
        self.skipbutton.setIconSize(QtCore.QSize(50, 50))
        self.skipbutton.clicked.connect(self.skipbuttonHandler)
        self.skipbutton.setEnabled(False)  # At app start the button is disabled
        PlayerBox.addWidget(self.skipbutton)

        framePlayer = QtWidgets.QFrame()
        framePlayer.setLayout(PlayerBox)
        layout.addWidget(framePlayer)

        self.show()

    @pyqtSlot(QtWidgets.QAbstractButton)
    @pyqtSlot(int)
    def on_button_clicked(self, button_or_id):
        # Next four lines in order to prevent the lines after "if self.flag01 == True:" from executing more than once
        # and delaying the start of the playback
        if self.cont02 == len(self.titles):
            self.flag01 = True
            self.cont02 = 0
        self.cont02 += 1
        if self.flag01 == True:
            if self.firsttime == True:  # After first mp3 selection buttons of player are enabled
                self.firtsttime = False
                self.playbutton.setEnabled(True)
                self.skipbutton.setEnabled(True)
            if isinstance(button_or_id, int):
                self.track = button_or_id
                self.multipleofeight = (ceil((self.track + 1) / 8.00) * 8)
                # You have reached the bottom of a page
                if (self.track + 1) == self.multipleofeight:
                    self.k = self.multipleofeight + 1
                self.frombg = True
                # Playing selected track
                self.play(self.f[self.track])
                self.show_artwork(button_or_id)
            elif isinstance(button_or_id, QtWidgets.QAbstractButton):
                self.titlelabel.setText(button_or_id.text())
                self.titlelabel.update()
            self.flag01 = False
            self.paused = False
            self.playing = True
            self.playbutton.setIcon(self.pauseicon)
            self.playbutton.setChecked(False)  # Needed for right functioning of play pause button

    # P L A Y    B U T T O N
    def playbuttonHandler(self):
        if self.playbutton.isChecked():
            if self.playing == True:
                mixer.music.pause()
                self.playbutton.setIcon(self.playicon)
                self.paused = True
                self.playing = True
            elif self.playing == False:
                self.playbutton.setIcon(self.playicon)
                self.play(self.f[self.track])  ##
                self.paused = True
                self.playing = False

        else:
            if self.paused == True and self.playing == False:
                self.play(self.f[self.track])  ##
                self.playbutton.setIcon(self.pauseicon)
                self.paused = False
                self.playing = True
            elif self.paused == True and self.playing == True:
                mixer.music.unpause()
                self.playbutton.setIcon(self.pauseicon)
                self.paused = False
            elif self.paused == False and self.playing == False:
                self.play(self.f[self.track])  ##
                self.playbutton.setIcon(self.pauseicon)
                self.playing = True

    # S K I P   B U T T O N
    def skipbuttonHandler(self):
        self.frombg = False
        self.track += 1
        # At the end of mp3 collection, track number is set to 0 and the vertical scroll bar is set to top
        if self.track == self.cont02:
            self.track = 0
            self.scrollbar.setValue(self.scrollbar.minimum())
        self.show_artwork(self.track)
        self.titlelabel.setText(self.titles[self.track])
        self.titlelabel.update()
        self.button = self.bg.button(self.track)  # Returns the button of the next track
        # Button set to checked programmatically in order to make the button change style
        self.button.setChecked(True)
        self.multipleofeight = (ceil((self.track + 1) / 8.00) * 8)
        if (self.track + 1) == self.multipleofeight:
            self.k = self.multipleofeight + 1
        if (self.track + 1) == self.k:
            pages = self.k / 8
            self.scrollbar.setValue(self.scrollbar.minimum() + round(pages) * self.scrollbar.pageStep())
        if self.paused == False and self.playing == True and self.frombg == False:
            mixer.music.stop()
            self.play(self.f[self.track])  ##
        elif self.paused == True and self.playing == True and self.frombg == False:
            mixer.music.stop()
            self.playing = False
        # elif self.paused == True and self.playing == False and self.frombg == False:
        #   print("T E S T")    ########### FOR TEST
        elif self.paused == True and self.playing == True and self.frombg == True:
            mixer.music.stop()  ##
            self.play(self.f[self.track])
            self.frombg = False
        elif self.paused == False and self.playing == True and self.frombg == True:
            mixer.music.stop()
            self.play(self.f[self.track])  ##
            self.frombg = False
        elif self.paused == True and self.playing == False and self.frombg == True:
            self.play(self.f[self.track])  ##
            self.playbutton.setIcon(self.pauseicon)
            self.paused = False
            self.playing = True
            self.frombg = False
            # Extract Artist, Title and cover artwork from MP3 metadata

    def apic_extract(self, mp3):
        try:
            tags = mutagen.mp3.Open(mp3)
        except:
            return False
        data = ""
        for i in tags:
            if i.startswith("APIC"):
                data = tags[i].data  # data is the image of the cover artwork
                break
        if not data:
            data = None
        try:
            audio = ID3(mp3)
        except:
            artist = ""
            track = ""
            return [data, artist, track]
        artist = audio['TPE1'].text[0]
        track = audio['TIT2'].text[0]
        return [data, artist, track]

    def show_artwork(self, button_id):
        infomp3 = self.apic_extract(self.f[button_id])
        image_data = infomp3[0]
        if image_data == None:
            self.qp = QtGui.QPixmap("noartwork.png")
        else:
            self.qp = QPixmap()
            self.qp.loadFromData(image_data)
        self.qp = self.qp.scaled(100, 100, Qt.KeepAspectRatio, Qt.FastTransformation)
        self.coverlabel.setPixmap(self.qp)
        self.coverlabel.update()

    # Play MP3 with pygame.mixer.music
    def play(self, mediafile):
        mixer.music.load(
            mediafile)  # If a music stream is already playing it will be stopped. Not needed an explicit stop command.
        mixer.music.play()  ##


app = QtWidgets.QApplication(sys.argv)
main = MainWindow()
main.show()
sys.exit(app.exec_())