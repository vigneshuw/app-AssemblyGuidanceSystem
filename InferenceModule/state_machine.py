import math
import numpy as np
from collections import deque, Counter


class StateMachine:

    def __init__(self, state_dependencies, num_classes, fps=30, cycle_summary_len=50, timer=(1.5, 2)):

        assert isinstance(state_dependencies, list), "The state dependencies should be a list of list"
        assert len(state_dependencies) == num_classes, "The number of states should match the number of classes"

        # Create variables based on number of classes
        # The first state is always true
        # TODO: Automated way of discovering current state
        self.states = [True if x == 6 else False for x in range(len(state_dependencies))]
        self.state_ids = list(range(len(self.states)))
        self.past_state = 0
        self.state_dependencies = state_dependencies
        # Create a class frame incrementer
        self.class_index_incrementer = np.eye(len(self.states), dtype=np.uint8)

        # Buttons for transitions
        self.d1 = timer[0]
        self.d2 = timer[1]
        # To be used for sequence break check
        self.max_timer = max(timer)

        # Keep track of inferences
        self.past_inference = 0
        self.past_inference_elapsed_time = 0
        self.current_inference = 0

        # The class occurrences counter
        self.class_occurrence_counter = np.zeros(shape=(1, 7), dtype=np.uint16)
        self.class_occurrence_counter_normalized = np.zeros(shape=(1, 7), dtype=np.float16)
        # Counter for associating the "Other" to a state
        self.class_occurrence_counter_no_other = np.zeros(shape=(1, 7), dtype=np.uint16)
        self.class_occurrence_counter_normalized_no_other = np.zeros(shape=(1, 7), dtype=np.float16)

        # General utils
        self.fps = fps

        # Managing errors and other
        self.untouched_states = list(range(len(self.states)))
        self.sequence_break_flag = False
        self.sequence_break_list = []
        self.repeated_states = []
        # Track of all state changes
        self.state_changes = []

        # Sequential timer for states
        self.states_visited = []
        self.time_states_visited = []
        self.current_state_counter = 0

        # Flag to reset the state machine
        self.reset_state_machine = False

        # Store results summary
        self.summary = deque(maxlen=cycle_summary_len)

        # State transition variables
        # An array of timer for each classes
        self.state_transition_timer = np.zeros(shape=(1, num_classes), dtype=np.uint16)
        self.cycle_reset_timer = 0
        self.touched_sequence_break_counter = 0
        
        # Quick fix
        self.first_inference = True

    def __update_class_occurrences(self, majority_vote):

        """
        Method that updates the counter for class occurrences in the inference. If "Other" class is detected the updates
        to the counter is done to the previous class, one before the "Other" class

        :param majority_vote: The class that has been identified by the majority voting
        :return: None
        """

        # Increment the class_counter for every inference
        self.class_occurrence_counter += self.class_index_incrementer[:, majority_vote][np.newaxis, :]
        # Normalize the counter to seconds
        self.class_occurrence_counter_normalized = self.class_occurrence_counter / self.fps

        # If the state is the "Other" state
        if majority_vote == self.state_ids[-1]:
            # Update the majority voting
            majority_vote = self.past_state

            # Increment the counter
            self.class_occurrence_counter_no_other += self.class_index_incrementer[:, majority_vote][np.newaxis, :]
            # Normalize the counter to seconds
            self.class_occurrence_counter_normalized_no_other = self.class_occurrence_counter_no_other / self.fps

    def __correct_class_occurrences(self, majority_vote):

        """
        The state changes were delayed, to ensure reliability. This method helps in correcting the timer appropriately.

        :param majority_vote: The class that has been identified by the majority voting
        :return: None
        """

        # Take time from previous state and add it to the current state
        # LATER: Remove the fix on fps and determine actual time
        self.class_occurrence_counter[0, majority_vote] += math.floor(self.state_transition_timer[0, majority_vote])
        self.class_occurrence_counter[0, self.past_state] -= math.floor(self.state_transition_timer[0, majority_vote])

        # Check if the past state is 'Other' state
        if self.past_state == self.state_ids[-1]:
            # Determine the past-past state
            try:
                past_past_state = self.state_changes[-3]
            except IndexError:
                # In case where the first state of the machine is "Other" - select the first state
                past_past_state = self.state_ids[0]

            # Decrement the 'Other' counter as well
            self.class_occurrence_counter_no_other[0, past_past_state] -= math.floor(self.state_transition_timer[0, majority_vote])
            # Normalize the counter to seconds
            self.class_occurrence_counter_normalized_no_other = self.class_occurrence_counter_no_other / self.fps

    def __check_cycle_completion(self, current_state):

        """
        This method checks to see if there is a cycle change for every matching inference.

        :param current_state: The current state that the state machine is in. This is different from the inference state
         or majority voting state
        :return: A boolean indicating cycle reset to be TRUE or FALSE
        """

        # Check if cycle is completed
        cycle_reset_condition = current_state == 0 and \
                                    (len(self.untouched_states) <= len(self.state_dependencies[0][0])) and \
                                    (self.cycle_reset_timer/self.fps) >= self.d2

        # If the cycle is to be reset
        if cycle_reset_condition:
            # Update the counters
            self.class_occurrence_counter[0, 0] -= self.cycle_reset_timer
            self.class_occurrence_counter_normalized = self.class_occurrence_counter / self.fps

        return cycle_reset_condition

    def __determine_sequence_break_states(self, mod_current_state):

        """
        This method contains the logic used to identify state breaks. This method is called only during state changes.

        :param actual_past_state: The past state, that does not include the "Other" state.
        :return: None
        """

        # TODO: What if the sequence breaks at first step itself

        # Determine all possible input states
        input_states = self.state_dependencies[mod_current_state][0]

        # Remove the touched states, starting state and "Other" state
        mod_input_states = []
        for state in input_states:
            if (state in self.untouched_states) and (state is not self.state_ids[0]) and \
                    (state is not self.state_ids[-1]):
                mod_input_states.append(state)
        # Add to the sequence break list - only if new
        for state in mod_input_states:
            if state not in self.sequence_break_list:
                self.sequence_break_list.append(state)

    def __update_sequence_break_status(self, current_state):

        """
        The updates to the sequence break states, if they are being reached in the past, after being broken.

        :param current_state:  The current state of the state machine, and not the majority voted inference state.
        :return: None
        """

        # Remove the states that were reached
        if current_state in self.sequence_break_list:
            self.sequence_break_list.remove(current_state)

        # When the whole list becomes empty
        if not self.sequence_break_list:
            self.sequence_break_flag = False

    def __repeated_state_inference_check(self, majority_vote):

        return (majority_vote not in self.untouched_states) and \
                (majority_vote != self.state_ids[0] and majority_vote != self.state_ids[-1]) and \
                self.class_occurrence_counter_normalized[0, majority_vote] > 10.0

    @staticmethod
    def get_current_state(states):

        """
        A state method to determine the current state of the state machine at any point in time.

        :param states: A list of booleans containing the status of each state instance.
        :return: The identified state.
        """

        # The max value
        max_val = max(states)
        max_val_id = states.index(max_val)
        return max_val_id

    def status_update(self):

        """
        This method is exposed to the program that is using the class. It provides an update on the status of the
        state machine

        :return: A boolean to indicate the reset of a state
        """

        # Store summary
        if self.reset_state_machine:

            # Update the repeated states
            self.repeated_states = np.unique(self.repeated_states).tolist()

            # Get the summary
            summary = {
                "class_occurrence_counter": self.class_occurrence_counter,
                "class_occurrence_time": self.class_occurrence_counter_normalized,
                "current_state": self.get_current_state(self.states),
                "sequence_break_list": self.sequence_break_list,
                "sequence_break_flag": self.sequence_break_flag,
                "reset_state_machine": self.reset_state_machine,
                "untouched_states": self.untouched_states,
                "repeated_states": self.repeated_states,
                "state_changes": self.state_changes,
                "state_ids": self.state_ids,
                "orderly_states_visited": self.states_visited,
                "time_states_visited": self.time_states_visited
            }

            # Store the summary
            self.summary.append(summary)

            # Reset all the state variables
            self.states = [True if x == 0 else False for x in range(len(self.state_dependencies))]
            self.past_state = 0
            self.past_inference = 0
            self.past_inference_elapsed_time = 0
            self.current_inference = 0
            # The class occurrences counter - update
            self.class_occurrence_counter = np.zeros(shape=(1, 7), dtype=np.uint16)
            self.class_occurrence_counter[0, 0] = self.cycle_reset_timer
            self.class_occurrence_counter_normalized = self.class_occurrence_counter / self.fps
            self.class_occurrence_counter_no_other = np.zeros(shape=(1, 7), dtype=np.uint16)
            self.class_occurrence_counter_normalized_no_other = np.zeros(shape=(1, 7), dtype=np.float16)

            # Managing errors and other
            self.untouched_states = list(range(len(self.states)))
            self.state_changes = []
            self.sequence_break_flag = True
            self.sequence_break_list = []
            self.repeated_states = []
            # Flag to reset the state machine
            self.reset_state_machine = False

            # State transition variables
            self.state_transition_timer = np.zeros(shape=(1, len(self.state_dependencies)), dtype=np.uint16)
            self.cycle_reset_timer = 0

            # Reset state-wise time counter items
            self.states_visited = []
            self.time_states_visited = []
            self.current_state_counter = self.cycle_reset_timer

        # Return if there is a need to update state
        return self.reset_state_machine

    def update_state(self, majority_vote):

        """
        Main function of the class. Keeps track of the properties and parameters of the state machine for each cycle
        of inference from the deep learning model.

        :param: majority_vote: The class that has been identified by the majority voting
        :return: A boolean indicating if a state reset is required.
        """

        # Determine the current state of the machine
        current_state = self.get_current_state(self.states)
        # Determine the current inference
        self.current_inference = majority_vote
        # Check if the inference matches the state
        state_inference_match = (current_state == majority_vote)

        if state_inference_match:
            # If they match
            # Update the counters based on inference
            self.__update_class_occurrences(majority_vote=majority_vote)

            # Increment the time for current state
            self.current_state_counter += 1

            # Check to see if cycle ends
            if current_state == 0:
                # Timer for cycle reset
                self.cycle_reset_timer += 1
            else:
                self.cycle_reset_timer = 0

            self.reset_state_machine = self.__check_cycle_completion(current_state)
            return self.status_update()

        else:
            # Increment the state transition time
            self.state_transition_timer[0, majority_vote] += 1

            # Time check for the counter
            if self.state_transition_timer[0, majority_vote]/self.fps < self.d1:
                # Update the counters based on the current state and not the majority voting
                self.__update_class_occurrences(majority_vote=current_state)

                # Increase the current state time
                self.current_state_counter += 1

                return self.status_update()

            # Check if the state change is already complete
            # TODO: Remove repeated states tracker
            if self.__repeated_state_inference_check(majority_vote):

                # Force them to the Other state
                majority_vote = self.state_ids[-1]
                self.current_inference = self.state_ids[-1]

                # Check for the state inference match
                state_inference_match = (current_state == majority_vote)
                if state_inference_match:
                    # Update the counters based on inference
                    self.state_transition_timer = np.zeros(shape=(1, len(self.state_dependencies)), dtype=np.uint16)
                    self.__update_class_occurrences(majority_vote=majority_vote)
                    return self.status_update()

                else:
                    # Update the state transition timer counters
                    self.state_transition_timer[0, majority_vote] += int(self.d1 * self.fps)

            # Update current and past states
            self.states = [False] * len(self.states)
            self.states[self.current_inference] = True
            self.past_state = current_state
            # Update the current running state
            current_state = self.get_current_state(self.states)

            # Track all possible state changes
            self.state_changes.append(current_state)

            # Do the computations associated with states visited
            self.states_visited.append(self.past_state)
            self.current_state_counter -= self.state_transition_timer[0, majority_vote]
            self.current_state_counter /= self.fps
            self.time_states_visited.append(self.current_state_counter)
            # Reset the current state counters
            self.current_state_counter = self.state_transition_timer[0, majority_vote]

            # Correct class occurrences - only after state update
            self.__correct_class_occurrences(majority_vote=majority_vote)
            # Start updating the new state - only after state update
            self.__update_class_occurrences(majority_vote=majority_vote)
            # Reset the timer for state transition
            self.state_transition_timer = np.zeros(shape=(1, len(self.state_dependencies)), dtype=np.uint16)

            # Update the untouched items
            if current_state in self.untouched_states:
                self.untouched_states.remove(current_state)
            else:
                # TODO: What to do when we have repeated states
                self.repeated_states.append(current_state)

            # Check for sequence
            # Get all the previous states
            input_states = self.state_dependencies[current_state][0]
            # Get the subsequent states
            output_states = self.state_dependencies[current_state][1]

            # Remove items from sequence break
            self.__update_sequence_break_status(current_state)

            # Conditions to add to the sequence break
            if self.past_state not in input_states and self.past_state is not self.state_ids[-1] and not self.first_inference:
                # Add to sequence break list if the detected state has occurred only within 2.0s
                if int(self.class_occurrence_counter_normalized[0, current_state]) <= self.max_timer + 0.5:
                    self.__determine_sequence_break_states(current_state)
                    # Set the flag only when the sequence break list has elements
                    if self.sequence_break_list:
                        self.sequence_break_flag = True

            elif self.past_state is self.state_ids[-1] and not self.first_inference:
                # Don't check the "Other" state but check one before
                state_one_before = self.state_changes[-3]
                if state_one_before not in input_states and state_one_before is not current_state:
                    # The detected state has occurred only within 2.0s
                    if int(self.class_occurrence_counter_normalized[0, current_state]) <= self.max_timer + 0.5:
                        self.__determine_sequence_break_states(current_state)
                        # Set the flag only when the sequence break list has elements
                        if self.sequence_break_list:
                            self.sequence_break_flag = True
                        
            # Reset the first time run flag
            self.first_inference = False

            return self.status_update()
