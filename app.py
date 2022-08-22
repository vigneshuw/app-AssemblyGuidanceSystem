from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QStyle, QFileDialog, QLabel, \
    QGridLayout, QComboBox, QDial, QGroupBox, QRadioButton
from PyQt6.QtGui import QIcon, QImage, QPixmap, QPainter, QFont
from PyQt6.QtCharts import QBarSet, QChart, QChartView, QBarCategoryAxis, QValueAxis, QHorizontalStackedBarSeries, \
    QHorizontalPercentBarSeries, QLineSeries
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPointF
from InferenceModule.state_machine import StateMachine
from InferenceModule.inferences import C3DOpticalFlowRealTime
import sys
import cv2
import os
import numpy as np
from collections import Counter
from tensorflow import keras
from database import StateMachineDB


class MainWindow(QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()

        # Get the file information
        self.file_name = None
        self.model_file_name = None

        # GroupBox
        self.video_gp = QGroupBox("Video Information")

        # Buttons
        # Load video
        vbox_load_items = QVBoxLayout()
        load_items_label = QLabel("(1) Load Data")
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
        self.load_model_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirIcon))
        self.load_model_btn.clicked.connect(self.open_file_model)
        # Initialization button
        self.initialize_all = QPushButton("Initialize")
        self.initialize_all.setEnabled(False)
        self.initialize_all.setFixedHeight(50)
        self.initialize_all.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_CommandLink))
        self.initialize_all.clicked.connect(self.initialize_all_fn)

        # Radio Buttons
        vbox_assembly_selection = QVBoxLayout()
        self.assembly_selection_1 = QRadioButton("AssemblyDemo")
        self.assembly_selection_1.setChecked(True)
        self.assembly_selection_2 = QRadioButton("L10 Assembly Line")

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
        self.sm_dial2_label_value = QLabel("1.0s")
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
        self.sm_dial2.setValue(10)
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
        self.chart_cycle_time_line.setTitle("Past Cycle Time")
        self.axis_x_cycle_time_line = QValueAxis()
        self.axis_x_cycle_time_line.setTitleText("Cycle Count")
        self.axis_x_cycle_time_line.setRange(1, 10)
        self.axis_x_cycle_time_line.setTickCount(10)
        self.chart_cycle_time_line.addAxis(self.axis_x_cycle_time_line, Qt.AlignmentFlag.AlignBottom)
        self.cycle_time_linechart_series.attachAxis(self.axis_x_cycle_time_line)
        self.axis_y_cycle_time_line = QValueAxis()
        self.axis_y_cycle_time_line.setTitleText("Time (s)")
        self.axis_y_cycle_time_line.setRange(10, 250)
        self.chart_cycle_time_line.addAxis(self.axis_y_cycle_time_line, Qt.AlignmentFlag.AlignLeft)
        self.cycle_time_linechart_series.attachAxis(self.axis_y_cycle_time_line)
        self._chart_view_cycle_line = QChartView(self.chart_cycle_time_line)
        # Line chart clicking functionalities
        self.cycle_time_linechart_series.doubleClicked.connect(self.line_chart_clicked_slot)

        # Assign buttons appropriately
        # Loading model and video
        load_items_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        load_items_label.setFont(QFont("Sanserif", 18))
        vbox_load_items.addWidget(load_items_label)
        vbox_load_items.addWidget(self.load_video_btn)
        vbox_load_items.addWidget(self.load_model_btn)
        vbox_load_items.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Setting the assembly operation
        select_assembly_ops_label = QLabel("(2) Assembly Type")
        select_assembly_ops_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        select_assembly_ops_label.setFont(QFont("Sanserif", 18))
        vbox_assembly_selection.addWidget(select_assembly_ops_label)
        vbox_assembly_selection.addWidget(self.assembly_selection_1)
        vbox_assembly_selection.addWidget(self.assembly_selection_2)
        vbox_assembly_selection.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox_assembly_selection.setSpacing(20)
        # Inference Initialization
        initialize_inference_label = QLabel("(3) Inference Module")
        initialize_inference_label.setFont(QFont("Sanserif", 18))
        vbox_initialize = QVBoxLayout()
        vbox_initialize.addWidget(initialize_inference_label)
        vbox_initialize.addWidget(self.initialize_all)
        vbox_initialize.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # The Start and Stop of Inference
        inference_control_label = QLabel("(3) Inference Control")
        inference_control_label.setFont(QFont("Sanserif", 18))
        vbox_inference_control = QVBoxLayout()
        vbox_inference_control.addWidget(inference_control_label)
        vbox_inference_control.addWidget(self.play_btn)
        vbox_inference_control.addWidget(self.cancel_btn)
        vbox_inference_control.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # For all buttons
        hbox_btns = QHBoxLayout()
        hbox_btns.setContentsMargins(0, 0, 0, 0)
        hbox_btns.addLayout(vbox_load_items)
        hbox_btns.addLayout(vbox_assembly_selection)
        hbox_btns.addLayout(vbox_initialize)
        hbox_btns.addLayout(inference_machine_dial_vbox)
        hbox_btns.addLayout(sm_dial_hbox)
        hbox_btns.addLayout(vbox_inference_control)
        self.video_gp.setLayout(hbox_btns)
        self.video_gp.setFont(QFont("Sanserif", 14))

        # For the video being playing
        self.vbox_video = QVBoxLayout()
        self.vbox_video.setContentsMargins(0, 0, 0, 0)
        self.vbox_video.addWidget(self.feed_label)

        # Set the maximum heights and stuff
        self.video_gp.setMaximumHeight(150)
        # Make a grid
        self.layout = QGridLayout()
        # Add Buttons
        self.layout.addWidget(self.video_gp, 0, 0, 1, 10)

        # Initialize all the threads
        self.Worker1 = Worker1()
        self.Worker1.ImageUpdate.connect(self.image_update_slot)

        # Hand-offs
        self.Worker1.cycle_time_line_series = self.cycle_time_linechart_series
        self.Worker1.axis_x_cycle_time_line = self.axis_x_cycle_time_line

        # Set a layout
        self.setLayout(self.layout)

    def image_update_slot(self, image):
        self.feed_label.setPixmap(QPixmap.fromImage(image))

    def line_chart_clicked_slot(self, point):
        print(point)

    def initialize_all_fn(self):

        # Initialize the model
        # Load and set the required model
        model = keras.models.load_model(self.model_file_name)
        self.Worker1.c3d_realtime = C3DOpticalFlowRealTime(model=model,
                                                           inference_length=self.inference_machine_dial.value())

        # Initialize the State Machine
        if self.assembly_selection_1.isChecked():
            assembly_index = 0
            assembly_op = "Demo"
        elif self.assembly_selection_2.isChecked():
            assembly_index = 1
            assembly_op = "L10"
        self.Worker1.assembly_op = assembly_op
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
        # Line chart
        self.vbox_video.addWidget(self._chart_view_cycle_line)
        # Transfer the plot control to Worker
        self.Worker1.time_sets_bysteps = self.time_sets_bysteps
        self.Worker1.cycle_percent_sets = self.cycle_percent_sets

        # Add the charts to the layout - Modify
        self.layout.addLayout(self.vbox_video, 2, 4, 8, 3)
        # self.layout.addLayout(self.vbox_bar_plots, 2, 0, 3, 3)
        self.layout.addWidget(self._chart_view_cycle_percent, 2, 0, 3, 4)
        self.layout.addWidget(self._chart_view_step_time, 5, 0, 5, 4)

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
        self.axis_x_step_time.setTitleText("Time(s)")
        self.time_series_bystep.attachAxis(self.axis_x_step_time)
        # Chart View for the Step-time bar chart
        self._chart_view_step_time = QChartView(self.chart_step_time)

        # Initialize the cycle time percentage bar chart
        self.chart_cycle_percent.addSeries(self.cycle_percent_series)
        self.chart_cycle_percent.setTitle("Cycle Time")
        # Only Y-axis
        self.axis_y_cycle_percent.append(self.categories_cycle_percent)
        self.chart_cycle_percent.addAxis(self.axis_y_cycle_percent, Qt.AlignmentFlag.AlignLeft)
        self.axis_x_cycle_percent.setTickCount(10)
        self.axis_x_cycle_percent.setTitleText("Percentage")
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
        self.cycle_time_line_series = None
        self.axis_x_cycle_time_line = None

        # Database connections
        self.sm_database = StateMachineDB()
        self.assembly_op = None

    def run(self):

        self.thread_active = True
        cap = cv2.VideoCapture(self.file_name)

        # Establish the DB connection
        # Connect the database
        self.sm_database.connect()
        self.sm_database.create_table(self.assembly_op)

        # Query the last 5 rows for the cycle-time line plot
        rows = self.sm_database.query_last_rows(assembly_op=self.assembly_op)
        if rows:
            for row in rows:
                # Get cycle time
                cycle_time = sum(eval(row[2]))
                print(cycle_time, row[0])
                # Update the plot
                self.cycle_time_line_series.append(row[0], cycle_time)

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

            # When there is a cycle reset happening
            if status:
                # Get the summary
                summary = self.inference_sm.summary[-1]

                # Insert data into the database
                # Convert items to appropriate for database
                row_id = self.construct_insert_sql_data(summary=summary)

                # Update plots
                # Get the cycle time and add to line plot
                cycle_time = sum(summary["class_occurrence_time"].tolist()[0])
                self.cycle_time_line_series.append(row_id, cycle_time)
                # Clear up other plots
                self.time_sets_bysteps[0].remove(0, count=len(self.inference_sm.states))
                self.time_sets_bysteps[0].append([0] * len(self.inference_sm.states))

                # Check the line chart count and update
                if self.cycle_time_line_series.count() >= 10:
                    # Update the line chart params
                    self.cycle_time_line_series.removePoints(0, 5)
                    # Move the axis
                    self.axis_x_cycle_time_line.setRange(row_id - 5, row_id + 5)

            # Convert frame to QT6 format for display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format.Format_RGB888)
            frame = frame.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)

            # Emit the thread
            self.ImageUpdate.emit(frame)

        # When we run out of video length
        self.construct_insert_sql_data()
        # Make a commit to database
        self.sm_database.db_conn.commit()
        # Close the DB connection
        self.sm_database.db_conn.close()

        cap.release()
        self.thread_active = False

    def stop(self):
        self.thread_active = False
        # Close the thread
        self.quit()

    def construct_insert_sql_data(self, summary=None):

        # When the cycle completes
        if summary is not None:
            # Insert data into the database
            # Convert items to appropriate for database
            temp_step_time = summary["class_occurrence_time"].tolist()[0]
            temp_step_time = [round(x, 2) for x in temp_step_time]
            step_time_str = repr(temp_step_time)
            sequence_break_flag = summary["sequence_break_flag"]
            if sequence_break_flag:
                sequence_break_items_str = repr(summary["sequence_break_list"])
            else:
                sequence_break_items_str = None
            missed_steps_str = summary["untouched_states"]
            if not missed_steps_str:
                missed_steps_str = None
            else:
                missed_steps_str = repr(missed_steps_str)
            states_sequence_str = repr(summary["state_changes"])

            # Insert data into database
            params = [step_time_str, sequence_break_items_str, sequence_break_flag, missed_steps_str,
                      states_sequence_str]
            return self.sm_database.insert_data(self.assembly_op, params)

        else:
            # First construct the summary
            summary = {
                "class_occurrence_time": self.inference_sm.class_occurrence_counter_normalized,
                "sequence_break_flag": self.inference_sm.sequence_break_flag,
                "sequence_break_list": self.inference_sm.sequence_break_list,
                "untouched_states": self.inference_sm.untouched_states,
                "state_changes": self.inference_sm.state_changes
            }
            self.construct_insert_sql_data(summary=summary)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

