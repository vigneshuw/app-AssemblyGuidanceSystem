from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QStyle, QFileDialog, QLabel, \
    QGridLayout, QComboBox, QDial, QGroupBox
from PyQt6.QtGui import QIcon, QImage, QPixmap, QPainter
from PyQt6.QtCharts import QBarSet, QChart, QChartView, QBarCategoryAxis, QValueAxis, QHorizontalStackedBarSeries, \
    QHorizontalPercentBarSeries, QLineSeries
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
        self.model_file_name = None

        # GroupBox
        self.model_gb = QGroupBox("Model parameters")
        self.video_gp = QGroupBox("Video Information")

        # Buttons
        # Load video
        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.setEnabled(True)
        self.load_video_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        self.load_video_btn.clicked.connect(self.open_file)
        # Play Button
        self.play_btn = QPushButton("Start Inference")
        self.play_btn.setEnabled(False)
        self.play_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_btn.clicked.connect(self.start_inference)
        # Cancel Inference
        self.cancel_btn = QPushButton("Stop Inference")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaStop))
        self.cancel_btn.clicked.connect(self.cancel_feed)
        # Load the model for inference
        self.load_model_btn = QPushButton("Load Model")
        self.load_model_btn.setEnabled(True)
        self.load_model_btn.setFixedWidth(200)
        self.load_model_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        self.load_model_btn.clicked.connect(self.open_file_model)
        # Initialization button
        self.initialize_all = QPushButton("Initialize Inference Module")
        self.initialize_all.setEnabled(False)
        self.initialize_all.setFixedHeight(100)
        self.initialize_all.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_CommandLink))
        self.initialize_all.clicked.connect(self.initialize_all_fn)

        # Combo boxes
        self.assembly_selection = QComboBox()
        self.assembly_selection.addItems(["Demo", "L10"])

        # Dials
        # The inference length dial
        inference_machine_dial_vbox = QVBoxLayout()
        self.inference_machine_dial_value_label = QLabel("30s")
        inference_machine_dial_label = QLabel("Inference length")
        inference_machine_dial_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inference_machine_dial_value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inference_machine_dial = QDial()
        self.inference_machine_dial.setRange(10, 60)
        self.inference_machine_dial.setSingleStep(10)
        self.inference_machine_dial.setValue(30)
        self.inference_machine_dial.setEnabled(False)
        self.inference_machine_dial.setNotchesVisible(True)
        inference_machine_dial_vbox.addWidget(inference_machine_dial_label)
        inference_machine_dial_vbox.addWidget(self.inference_machine_dial)
        inference_machine_dial_vbox.addWidget(self.inference_machine_dial_value_label)
        # Action for Dial
        self.inference_machine_dial.valueChanged.connect(self.inference_machine_dial_value_changed)
        # The state machine params dial
        sm_dial_vbox1 = QVBoxLayout()
        sm_dial_vbox2 = QVBoxLayout()
        sm_dial1_label = QLabel("State Transition")
        sm_dial1_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sm_dial2_label = QLabel("Cycle Reset")
        sm_dial2_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sm_dial1_label_value = QLabel("3.0s")
        self.sm_dial1_label_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sm_dial2_label_value = QLabel("2.0s")
        self.sm_dial2_label_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # SM Dial-1
        self.sm_dial1 = QDial()
        self.sm_dial1.setRange(5, 50)
        self.sm_dial1.setSingleStep(1)
        self.sm_dial1.setValue(30)
        self.sm_dial1.setEnabled(False)
        self.sm_dial1.setNotchesVisible(True)
        self.sm_dial1.valueChanged.connect(self.sm_dial1_value_changed)
        sm_dial_vbox1.addWidget(sm_dial1_label)
        sm_dial_vbox1.addWidget(self.sm_dial1)
        sm_dial_vbox1.addWidget(self.sm_dial1_label_value)
        # SM Dial-2
        self.sm_dial2 = QDial()
        self.sm_dial2.setRange(5, 50)
        self.sm_dial2.setSingleStep(1)
        self.sm_dial2.setValue(20)
        self.sm_dial2.setEnabled(False)
        self.sm_dial2.setNotchesVisible(True)
        self.sm_dial2.valueChanged.connect(self.sm_dial2_value_changed)
        sm_dial_vbox2.addWidget(sm_dial2_label)
        sm_dial_vbox2.addWidget(self.sm_dial2)
        sm_dial_vbox2.addWidget(self.sm_dial2_label_value)
        # Add to a horizontal box
        sm_dial_hbox = QHBoxLayout()
        sm_dial_hbox.addLayout(sm_dial_vbox1)
        sm_dial_hbox.addLayout(sm_dial_vbox2)

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
        self._chart_view_step_time = None
        # BarPlots - Percentage of Value added activities
        self.cycle_percent_sets = []
        self.cycle_percent_series = QHorizontalPercentBarSeries()
        self.chart_cycle_percent = QChart()
        self.axis_y_cycle_percent = QBarCategoryAxis()
        self.categories_step_time = None
        self.categories_cycle_percent = ["Cycle"]
        self.axis_x_cycle_percent = QValueAxis()
        self._chart_view_cycle_percent = None
        # Creating a line chart
        self.cycle_time_linechart_series = QLineSeries()
        self.chart_cycle_time_line = QChart()
        self.chart_cycle_time_line.addSeries(self.cycle_time_linechart_series)
        self.chart_cycle_time_line.createDefaultAxes()
        self._chart_view_cycle_line = QChartView(self.chart_cycle_time_line)

        # For all buttons
        hbox_btns = QHBoxLayout()
        hbox_btns.setContentsMargins(0, 0, 0, 0)
        hbox_btns.addWidget(self.load_video_btn)
        hbox_btns.addWidget(self.play_btn)
        hbox_btns.addWidget(self.cancel_btn)
        self.video_gp.setLayout(hbox_btns)
        # Other buttons
        hbox_btns_2 = QHBoxLayout()
        hbox_btns_2.setContentsMargins(0, 0, 0, 0)
        hbox_btns_2.addWidget(self.load_model_btn)
        hbox_btns_2.addWidget(self.assembly_selection)
        hbox_btns_2.addWidget(self.initialize_all)
        hbox_btns_2.addLayout(inference_machine_dial_vbox)
        hbox_btns_2.addLayout(sm_dial_hbox)
        self.model_gb.setLayout(hbox_btns_2)

        # For the video being playing
        self.vbox_video = QVBoxLayout()
        self.vbox_video.setContentsMargins(0, 0, 0, 0)
        self.vbox_video.addWidget(self.feed_label)

        # Make a grid
        self.layout = QGridLayout()
        # Add Buttons
        self.layout.addWidget(self.video_gp, 0, 0, 1, 10)
        self.layout.addWidget(self.model_gb, 1, 0, 1, 10)

        # Initialize all the threads
        self.Worker1 = Worker1()
        self.Worker1.ImageUpdate.connect(self.image_update_slot)

        # Set a layout
        self.setLayout(self.layout)

    def image_update_slot(self, image):
        self.feed_label.setPixmap(QPixmap.fromImage(image))

    def initialize_all_fn(self):

        # Initialize the model
        # Load and set the required model
        model = keras.models.load_model(self.model_file_name)
        self.Worker1.c3d_realtime = C3DOpticalFlowRealTime(model=model,
                                                           inference_length=self.inference_machine_dial.value())

        # Initialize the State Machine
        assembly_index = self.assembly_selection.currentIndex()
        # Assign the values appropriately
        if assembly_index == 1:
            sys.stdout.write("Not Implemented\n")
        elif assembly_index == 0:
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
            # Initialize the inference state machine
            self.Worker1.inference_sm = StateMachine(state_dependencies=state_dependencies,
                                                     num_classes=len(state_dependencies),
                                                     timer=(self.sm_dial1.value()/10, self.sm_dial2.value()/10))

            # Define the categories for step time plot
            self.categories_step_time = ["Position Motherboard", "Attach Bracket", "Secure Motherboard", "Insert Card",
                                         "Attach Device", "Remove Battery", "Others"]
            # Enable the charts initialization
            self.initialize_charts(num_steps=6)

        # Create plots appropriately
        # Add the plots to the vbox
        # Bar charts
        # For the bar plots
        # self.vbox_bar_plots = QVBoxLayout()
        # self.vbox_bar_plots.setContentsMargins(0, 0, 0, 0)
        # self.vbox_bar_plots.addWidget(self._chart_view_cycle_percent,
        #                               alignment=Qt.AlignmentFlag.AlignTop)
        # self.vbox_bar_plots.addWidget(self._chart_view_step_time, alignment=Qt.AlignmentFlag.AlignBottom)
        # Line chart
        self.vbox_video.addWidget(self._chart_view_cycle_line)
        # Transfer the plot control to Worker
        self.Worker1.time_sets_bysteps = self.time_sets_bysteps
        self.Worker1.cycle_percent_sets = self.cycle_percent_sets

        # Add the charts to the layout
        self.layout.addLayout(self.vbox_video, 2, 3, 3, 5)
        # self.layout.addLayout(self.vbox_bar_plots, 2, 0, 3, 3)
        self.layout.addWidget(self._chart_view_cycle_percent, 2, 0, 3, 5)
        self.layout.addWidget(self._chart_view_step_time, 5, 0, 5, 5)
        # Reset the layout
        # self.setLayout(self.layout)

        # Enable the inference button
        self.play_btn.setEnabled(True)
        self.cancel_btn.setEnabled(True)

    def cancel_feed(self):
        self.Worker1.stop()

        # Update button states
        self.load_video_btn.setEnabled(True)
        self.load_model_btn.setEnabled(True)
        self.play_btn.setEnabled(True)

    def inference_machine_dial_value_changed(self):

        # Set the values
        current_value = self.inference_machine_dial.value()
        self.Worker1.inference_length = current_value

        # Update the label
        self.inference_machine_dial_value_label.setText(str(current_value) + "s")

    def sm_dial1_value_changed(self):

        # Set the values
        current_value = self.sm_dial1.value() / 10
        self.Worker1.d1 = current_value

        # Update the label
        self.sm_dial1_label_value.setText(str(current_value) + "s")

    def sm_dial2_value_changed(self):

        # Set the values
        current_value = self.sm_dial2.value() / 10
        self.Worker1.d2 = current_value

        # Update the label
        self.sm_dial2_label_value.setText(str(current_value) + "s")

    def start_inference(self):
        # Others
        self.load_video_btn.setEnabled(False)
        self.play_btn.setEnabled(False)

        # Start the worker operation
        self.Worker1.start()

    def open_file(self):

        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video")

        # Set media player to file name
        if file_name != '':
            self.file_name = file_name
            self.Worker1.file_name = file_name
            if self.model_file_name is not None:
                # Enable all dials
                self.inference_machine_dial.setEnabled(True)
                self.sm_dial1.setEnabled(True)
                self.sm_dial2.setEnabled(True)
                self.initialize_all.setEnabled(True)
            # Disable the button
            self.load_video_btn.setEnabled(False)

    def open_file_model(self):

        file_name, _ = QFileDialog.getOpenFileName(self, "Open Model")

        # Set model to model file name
        if file_name != "":
            self.model_file_name = file_name
            self.Worker1.model_file_name = file_name
            if self.file_name is not None:
                # Enable all dials
                self.inference_machine_dial.setEnabled(True)
                self.sm_dial1.setEnabled(True)
                self.sm_dial2.setEnabled(True)
                self.initialize_all.setEnabled(True)
            # Disable the button
            self.load_model_btn.setEnabled(False)

    def initialize_charts(self, num_steps):

        # Add the 'Others' class to the step_names
        step_names = ["Step time", "Miscellaneous"]
        percent_names = ["Value Added", "Non Value Added"]

        # Initialize timers
        timer_init = [0] * (num_steps + 1)
        percent_init = [0] * 2

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
        self.axis_x_step_time.setRange(0, 100)
        self.axis_x_step_time.setTickCount(10)
        self.chart_step_time.addAxis(self.axis_x_step_time, Qt.AlignmentFlag.AlignBottom)
        self.time_series_bystep.attachAxis(self.axis_x_step_time)
        # Chart View for the Step-time bar chart
        self._chart_view_step_time = QChartView(self.chart_step_time)

        # Initialize the cycle time percentage bar chart
        self.chart_cycle_percent.addSeries(self.cycle_percent_series)
        # Only Y-axis
        self.axis_y_cycle_percent.append(self.categories_cycle_percent)
        self.chart_cycle_percent.addAxis(self.axis_y_cycle_percent, Qt.AlignmentFlag.AlignLeft)
        self.axis_x_cycle_percent.setTickCount(10)
        self.chart_cycle_percent.addAxis(self.axis_x_cycle_percent, Qt.AlignmentFlag.AlignBottom)
        self.cycle_percent_series.attachAxis(self.axis_y_cycle_percent)
        self.cycle_percent_series.attachAxis(self.axis_x_cycle_percent)
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

        # Model information
        self.model_file_name = None

        # The Inference module
        # Hyperparams
        self.inference_length = None
        self.d1 = None
        self.d2 = None

        self.c3d_realtime = None
        self.inference_sm = None
        self.classes_to_states = None

        # Transfer of plotting information
        self.time_sets_bysteps = None
        self.cycle_percent_sets = None

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

            # Update the Step time plot
            current_state = self.inference_sm.get_current_state(self.inference_sm.states)
            self.time_sets_bysteps[0].replace(current_state,
                                              self.inference_sm.class_occurrence_counter_normalized[0, current_state])
            self.time_sets_bysteps[1].remove(0, count=len(self.inference_sm.states))
            self.time_sets_bysteps[1].append(self.inference_sm.class_occurrence_counter_normalized_no_other.tolist()[0])
            # Update the Cycle time percentage plot
            self.cycle_percent_sets[0].replace(0,
                                               np.sum(self.inference_sm.class_occurrence_counter_normalized[0, 0:-1]))
            self.cycle_percent_sets[1].replace(0,
                                               self.inference_sm.class_occurrence_counter_normalized[0, -1])

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

