import sys
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QGridLayout,
    QGroupBox,
    QVBoxLayout,
)
databases = ["db{}".format(i) for i in range(10)]
class App(QDialog):
    def __init__(self):
        super().__init__()
        self.title = "MySQL Timing Discrepancies"
        self.left = 10
        self.top = 10
        self.width = 320
        self.height = 100
        self.initUI()
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self._databases_checked = []
        self.createGridLayout()
        windowLayout = QVBoxLayout(self)
        windowLayout.addWidget(self.horizontalGroupBox)
        self.show()
    @property
    def databases_checked(self):
        return self._databases_checked
    def createGridLayout(self):
        self.horizontalGroupBox = QGroupBox("Databases")
        layout = QGridLayout()
        self.horizontalGroupBox.setLayout(layout)
        for i in range(3):
            layout.setColumnStretch(i, 0)
        row_num = 0
        # Generating checkboxes here
        for i, database in enumerate(databases):
            col_num = i % 3
            row_num = i if col_num == 0 else row_num
            cb = QCheckBox(database)
            cb.setChecked(False)
            cb.stateChanged.connect(self.on_stateChanged)
            layout.addWidget(cb, row_num, col_num)
    @pyqtSlot(int)
    def on_stateChanged(self, state):
        checkbox = self.sender()
        text = checkbox.text()
        if state == Qt.Checked:
            self._databases_checked.append(text)
        else:
            self._databases_checked.remove(text)
        print(self.databases_checked)
if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())