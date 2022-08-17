from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QStyle, QFileDialog, QLabel, QGridLayout
from PyQt6.QtGui import QIcon, QImage, QPixmap, QPainter
from PyQt6.QtCharts import QBarSet, QStackedBarSeries, QChart, QChartView, QBarCategoryAxis, QValueAxis, QBarSeries
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtCore import QThread, pyqtSignal, Qt
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

        # Window properties
        self.setWindowTitle("Real-time Inference")
        self.setGeometry(350, 100, 1920, 1280)
        # Labels
        self.feed_label = QLabel()

        # Plots
        # BarPlots
        self.time_sets_bysteps = []
        self.time_series_bystep = QBarSeries()
        self.chart_step_time = QChart()
        self.axis_x_step_time = QBarCategoryAxis()
        self.axis_y_step_time = QValueAxis()
        self.categories_step_time = ["Position Motherboard", "Attach Bracket", "Secure Motherboard", "Insert Card",
                                     "Attach Device", "Remove Battery", "Others"]
        self._chart_view_step_time = None
        self._chart_view_step_time_percentage = None

        # Testing
        self.initialize_charts(6)

        # For all buttons
        hbox_btns = QHBoxLayout()
        hbox_btns.setContentsMargins(0, 0, 0, 0)
        hbox_btns.addWidget(self.load_video_btn)
        hbox_btns.addWidget(self.play_btn)
        hbox_btns.addWidget(self.cancel_btn)

        # For the bar plots
        vbox_bar_plots = QVBoxLayout()
        vbox_bar_plots.setContentsMargins(0, 0, 0, 0)
        vbox_bar_plots.addWidget(self._chart_view_step_time)


        # Make a grid
        layout = QGridLayout()
        # Add Buttons
        layout.addLayout(hbox_btns, 0, 0, 1, 10)
        layout.addLayout(vbox_bar_plots, 1, 0, 5, 3)
        # Add the plots
        # layout.addWidget(self.feed_label, 1, 0, 3, 1)
        # layout.addWidget(self._chart_view_step_time, 1, 1, 3, 1)


        # For the video
        # hbox_plots = QHBoxLayout()
        # hbox_plots_sub = QHBoxLayout()
        # vbox_plots = QVBoxLayout()
        # vbox_plots.addWidget(self._chart_view_step_time)
        # vbox_plots.addWidget(self._chart_view_step_time)
        # hbox_plots_sub.addLayout(vbox_plots)
        # hbox_plots.addLayout(hbox_plots_sub)
        # hbox_plots.addWidget(self.feed_label)


        # # Vertical Box
        # vbox = QVBoxLayout()
        # vbox.addLayout(hbox_plots)
        # vbox.addLayout(hbox_btns)

        # Initialize all the threads
        self.Worker1 = Worker1()
        self.Worker1.ImageUpdate.connect(self.image_update_slot)

        # Set a layout
        self.setLayout(layout)

    def image_update_slot(self, image):
        self.feed_label.setPixmap(QPixmap.fromImage(image))

    def cancel_feed(self):
        self.Worker1.stop()

        # Update button states
        self.load_video_btn.setEnabled(True)
        self.play_btn.setEnabled(True)

    def start_inference(self):
        # Others
        self.load_video_btn.setEnabled(False)
        self.play_btn.setEnabled(False)

        # Start the updates
        self.time_sets_bysteps[0].replace(0, 10)

        # Start the worker operation
        self.Worker1.start()

    def open_file(self):

        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video")

        # Set media player to file name
        if file_name != '':
            self.file_name = file_name
            self.Worker1.file_name = file_name
            self.play_btn.setEnabled(True)
            self.cancel_btn.setEnabled(True)

    def initialize_charts(self, num_steps):

        # Add the 'Others' class to the step_names
        step_names = ["Step time", "Miscellaneous"]

        # Initialize timers
        timer_init = [1] * (num_steps + 1)
        print(timer_init)

        # data sets
        for step in range(2):
            # Create the set
            bar_set = QBarSet(step_names[step])
            bar_set.append(timer_init)
            # add to the list after initialization
            self.time_sets_bysteps.append(bar_set)

            # Append the sets to the Bar series
            self.time_series_bystep.append(self.time_sets_bysteps[step])

        # Initialize the charts
        self.chart_step_time.addSeries(self.time_series_bystep)
        self.chart_step_time.setTitle("Step Time in Seconds")

        self.axis_x_step_time.append(self.categories_step_time)
        self.chart_step_time.addAxis(self.axis_x_step_time, Qt.AlignmentFlag.AlignBottom)
        self.time_series_bystep.attachAxis(self.axis_x_step_time)

        self.axis_y_step_time.setRange(0, 15)
        self.chart_step_time.addAxis(self.axis_y_step_time, Qt.AlignmentFlag.AlignLeft)
        self.time_series_bystep.attachAxis(self.axis_y_step_time)

        self._chart_view_step_time = QChartView(self.chart_step_time)


class Worker1(QThread):

    # Signal to transmit data
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self):
        super(Worker1, self).__init__()

        # Video file to load
        self.file_name = None
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
            frame = frame.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)

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

