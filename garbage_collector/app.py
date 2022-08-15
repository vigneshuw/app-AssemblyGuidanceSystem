from PyQt6.QtQml import QQmlApplicationEngine
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QPushButton
from PyQt6.QtCore import QObject, QStandardPaths, Qt
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtMultimedia import QMediaPlayer
import sys
import os
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Window settings
        self.setGeometry(0, 0, 640, 480)
        self.setWindowTitle("Real-time Inference")

        # Create buttons to load video
        load_video_btn = QPushButton('Load Video')

        # Create HBox Layout
        hbox = QHBoxLayout()
        hbox.addWidget(load_video_btn)
        self.setLayout(hbox)

        # The Video player
        self._player = QMediaPlayer()
        # self.open()
        return

    def open(self):
        self._ensure_stopped()
        file_dialog = QtWidgets.QFileDialog(self)

        # Set the mime types - only MP4
        file_dialog.selectMimeTypeFilter('video/mp4')

        # Get and open the standard paths
        # movies_location = QStandardPaths.writableLocation(QStandardPaths.MoviesLocation)
        # file_dialog.setDirectory(movies_location)
        if file_dialog.exec() == QtWidgets.QDialog.accept(self):
            url = file_dialog.selectedUrls()[0]
            self._player.setSource(url)
            self._player.play()

    def _ensure_stopped(self):
        return




        # self.frm = QtWidgets.QFrame(self)
        # self.frm.setStyleSheet("QWidget { background-color: #eeeeec; }")
        # self.lyt = QtWidgets.QVBoxLayout()
        # self.frm.setLayout(self.lyt)
        # self.setCentralWidget(self.frm)
        #
        # # Matplotlib Figure
        # self.myFig = MyFigureCanvas(x_len=200, y_range=[0, 100], interval=1)
        # self.lyt.addWidget(self.myFig)
        #
        # # 3. Show
        # self.show()
        # return


class MyFigureCanvas(FigureCanvas):

    def __init__(self, x_len: int, y_range: list, interval: int) -> None:
        '''
        :param x_len:       The nr of data points shown in one plot.
        :param y_range:     Range on y-axis.
        :param interval:    Get a new datapoint every .. milliseconds.

        '''
        super().__init__(Figure())
        # Range settings
        self._x_len_ = x_len
        self._y_range_ = y_range

        # Store two lists _x_ and _y_
        self._x_ = list(range(0, x_len))
        self._y_ = [0] * x_len

        # Store a figure ax
        self._ax_ = self.figure.subplots()
        self._ax_.set_ylim(ymin=self._y_range_[0], ymax=self._y_range_[1]) # added
        self._line_, = self._ax_.plot(self._x_, self._y_)                  # added
        self.draw()                                                        # added

        # Initiate the timer
        self._timer_ = self.new_timer(interval, [(self._update_canvas_, (), {})])
        self._timer_.start()
        return

    def _update_canvas_(self) -> None:
        '''
        This function gets called regularly by the timer.

        '''
        self._y_.append(round(get_next_datapoint(), 2))     # Add new datapoint
        self._y_ = self._y_[-self._x_len_:]                 # Truncate list y

        # Previous code
        # --------------
        # self._ax_.clear()                                   # Clear ax
        # self._ax_.plot(self._x_, self._y_)                  # Plot y(x)
        # self._ax_.set_ylim(ymin=self._y_range_[0], ymax=self._y_range_[1])
        # self.draw()

        # New code
        # ---------
        self._line_.set_ydata(self._y_)
        self._ax_.draw_artist(self._ax_.patch)
        self._ax_.draw_artist(self._line_)
        self.update()
        self.flush_events()
        return

# Data source
# ------------
n = np.linspace(0, 499, 500)
d = 50 + 25 * (np.sin(n / 8.3)) + 10 * (np.sin(n / 7.5)) - 5 * (np.sin(n / 1.5))
i = 0
def get_next_datapoint():
    global i
    i += 1
    if i > 499:
        i = 0
    return d[i]

qapp = QApplication(sys.argv)
app = ApplicationWindow()
sys.exit(qapp.exec())

