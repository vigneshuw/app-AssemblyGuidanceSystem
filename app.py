from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QStyle, QFileDialog, QLabel, QGridLayout
from PyQt6.QtGui import QIcon, QImage, QPixmap, QPainter
from PyQt6.QtCharts import QBarSet, QChart, QChartView, QBarCategoryAxis, QValueAxis, QHorizontalStackedBarSeries, \
    QHorizontalPercentBarSeries
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from InferenceModule.state_machine import StateMachine
from InferenceModule.inferences import C3DOpticalFlowRealTime
import sys
import cv2
import os
import numpy as np
from collections import Counter
from tensorflow import keras


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
        # BarPlot - Step Time
        self.time_sets_bysteps = []
        self.time_series_bystep = QHorizontalStackedBarSeries()
        self.chart_step_time = QChart()
        self.axis_y_step_time = QBarCategoryAxis()
        self.axis_x_step_time = QValueAxis()
        self.categories_step_time = ["Position Motherboard", "Attach Bracket", "Secure Motherboard", "Insert Card",
                                     "Attach Device", "Remove Battery", "Others"]
        self._chart_view_step_time = None
        # BarPlots - Percentage of Value added activities
        self.cycle_percent_sets = []
        self.cycle_percent_series = QHorizontalPercentBarSeries()
        self.chart_cycle_percent = QChart()
        self.axis_y_cycle_percent = QBarCategoryAxis()
        self.categories_cycle_percent = ["Cycle"]
        self._chart_view_cycle_percent = None

        # Testing
        self.initialize_charts(num_steps=6)

        # For all buttons
        hbox_btns = QHBoxLayout()
        hbox_btns.setContentsMargins(0, 0, 0, 0)
        hbox_btns.addWidget(self.load_video_btn)
        hbox_btns.addWidget(self.play_btn)
        hbox_btns.addWidget(self.cancel_btn)

        # For the bar plots
        vbox_bar_plots = QVBoxLayout()
        vbox_bar_plots.setContentsMargins(0, 0, 0, 0)
        vbox_bar_plots.addWidget(self._chart_view_cycle_percent)
        vbox_bar_plots.addWidget(self._chart_view_step_time)
        # For the video being playing
        vbox_video = QVBoxLayout()
        vbox_video.setContentsMargins(0, 0, 0, 0)
        vbox_video.addWidget(self.feed_label)

        # Make a grid
        layout = QGridLayout()
        # Add Buttons
        layout.addLayout(hbox_btns, 0, 0, 1, 10)
        layout.addLayout(vbox_bar_plots, 1, 0, 3, 3)
        layout.addLayout(vbox_video, 1, 3, 3, 5)

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

        # Initialize the state machines and assoc
        self.Worker1.classes_to_states = {
            5: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            6: 5,
            0: 6

        }
        # Create state dependencies
        state_dependencies = [
            [
                [3, 4, 5, 6], [1],  # S0
            ],

            [
                [0, 6], [2]  # S1
            ],

            [
                [1, 6], [3]  # S2
            ],

            [
                [2, 6], [4]  # S3
            ],

            [
                [3, 6], [5]  # S4
            ],

            [
                [4, 6], [0]  # S5
            ],

            [
                [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]  # S6 - The "Other" class
            ],
        ]
        # Model path
        model_path = os.path.join(os.path.dirname(os.getcwd()), "trained_models", "AssemblyDemo", "c3d",
                                  "C3D_scratch_best_optimo.hdf5")
        model = keras.models.load_model(model_path)

        # Real time class
        self.Worker1.c3d_realtime = C3DOpticalFlowRealTime(model, inference_length=30)
        # Initialize the inference state machine
        self.Worker1.inference_sm = StateMachine(state_dependencies=state_dependencies, num_classes=len(state_dependencies),
                                    timer=(3, 2))

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
        percent_names = ["Value Added", "Non Value Added"]

        # Initialize timers
        timer_init = [1] * (num_steps + 1)
        percent_init = [1] * 2

        # Chart creation for both bar charts
        for step in range(2):
            # Create sets and series for the step time bar chart
            bar_set = QBarSet(step_names[step])
            bar_set.append(timer_init)
            self.time_sets_bysteps.append(bar_set)
            self.time_series_bystep.append(self.time_sets_bysteps[step])

            # Create sets and series for the percentage bar chart
            bar_set_percent = QBarSet(percent_names[step])
            bar_set_percent.append(percent_init)
            self.cycle_percent_sets.append(bar_set_percent)
            self.cycle_percent_series.append(self.cycle_percent_sets[step])

        # Initialize the step time bar chart
        self.chart_step_time.addSeries(self.time_series_bystep)
        self.chart_step_time.setTitle("Step Time in Seconds")
        # Y-axis
        self.axis_y_step_time.append(self.categories_step_time)
        self.chart_step_time.addAxis(self.axis_y_step_time, Qt.AlignmentFlag.AlignLeft)
        self.time_series_bystep.attachAxis(self.axis_y_step_time)
        # X-axis
        self.axis_x_step_time.setRange(0, 15)
        self.chart_step_time.addAxis(self.axis_x_step_time, Qt.AlignmentFlag.AlignBottom)
        self.time_series_bystep.attachAxis(self.axis_x_step_time)
        # Chart View for the Step-time bar chart
        self._chart_view_step_time = QChartView(self.chart_step_time)

        # Initialize the cycle time percentage bar chart
        self.chart_cycle_percent.addSeries(self.cycle_percent_series)
        # Only Y-axis
        self.axis_y_cycle_percent.append(self.categories_cycle_percent)
        self.chart_cycle_percent.addAxis(self.axis_y_cycle_percent, Qt.AlignmentFlag.AlignLeft)
        self.cycle_percent_series.attachAxis(self.axis_y_cycle_percent)
        # Chart View for the Cycle percent chart
        self._chart_view_cycle_percent = QChartView(self.chart_cycle_percent)


class Worker1(QThread):

    # Signal to transmit data
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self):
        super(Worker1, self).__init__()

        # Video file to load
        self.file_name = None
        self.thread_active = False

        # The Inference module
        self.c3d_realtime = None
        self.inference_sm = None
        self.classes_to_states = None


    def run(self):

        self.thread_active = True
        cap = cv2.VideoCapture(self.file_name)

        while self.thread_active and cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            inference_packet = self.c3d_realtime.make_inference(frame)
            if inference_packet is None:
                continue
                # Unpack the inference packet
            inferences_list, inferences_prob = inference_packet
            # Get the array of inferences
            inferences_prob_summary = np.round_(np.array(inferences_prob).mean(axis=0), decimals=4)[0, :]
            # Rearrange the output
            inferences_prob_summary = np.array(inferences_prob_summary[list(self.classes_to_states.keys())])[np.newaxis, :]

            # Counter
            counter = Counter(inferences_list)
            # get the most common
            majority_vote = counter.most_common(1)[0][0]
            # TODO: Fix the step order
            majority_vote = self.classes_to_states[majority_vote]

            # Update the state machine
            status = self.inference_sm.update_state(majority_vote=majority_vote)

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

