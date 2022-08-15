from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QStyle, QFileDialog, QLabel
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QThread, pyqtSignal
import sys
import cv2


class MainWindow(QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()

        # Get the file information
        self.file_name = None

        # Buttons
        # Load video
        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.setEnabled(True)
        self.load_video_btn.clicked.connect(self.open_file)
        # Play Button
        self.play_btn = QPushButton()
        self.play_btn.setEnabled(False)
        self.play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_btn.clicked.connect(self.start_inference)
        # Cancel Inference
        self.cancel_btn = QPushButton()
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.cancel_btn.clicked.connect(self.cancel_feed)

        self.setWindowTitle("Real-time Inference")
        self.setGeometry(350, 100, 640, 480)
        # Labels
        self.feed_label = QLabel()

        # Horizontal Box
        hbox = QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.addWidget(self.load_video_btn)
        hbox.addWidget(self.play_btn)
        hbox.addWidget(self.cancel_btn)
        # Vertical Box
        vbox = QVBoxLayout()
        vbox.addWidget(self.feed_label)
        vbox.addLayout(hbox)

        # Set a layout
        self.setLayout(vbox)

    def image_update_slot(self, image):
        self.feed_label.setPixmap(QPixmap.fromImage(image))

    def cancel_feed(self):
        self.Worker1.stop()

    def start_inference(self):
        # Others
        self.load_video_btn.setEnabled(False)

        # Threads
        self.Worker1 = Worker1(self.file_name)
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.image_update_slot)

    def open_file(self):

        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video")

        # Set media player to file name
        if file_name != '':
            self.file_name = file_name
            self.play_btn.setEnabled(True)
            self.cancel_btn.setEnabled(True)


class Worker1(QThread):

    # Signal to transmit data
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self, file_name):
        super(Worker1, self).__init__()

        # Video file to load
        self.file_name = file_name
        self.thread_active = False

    def run(self):

        self.thread_active = True
        cap = cv2.VideoCapture(self.file_name)

        while self.thread_active and cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            # Convert frame to QT6 format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)

            # Emit the thread
            self.ImageUpdate.emit(frame)

        cap.release()
        self.thread_active = False

    def stop(self):
        self.thread_active = False
        self.quit()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

