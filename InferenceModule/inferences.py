import cv2
import numpy as np
from tensorflow import keras
import os
import sys
from collections import deque


class C3DOpticalFlowRealTime:

    def __init__(self, model, inference_length=10, resize=112, temporal_length=16):

        """
        The initialization for the real-time optical flow inference
        :param model: The model used in the inference process
        :param inference_length: The length for majority voting. Default to 10
        :param resize: The resize for the image input. Depends on the model used
        :param temporal_length: The length in frames to consider to make the inference
        """

        # Initialize
        self.inferences = deque(maxlen=inference_length)
        self.inference_probabilities = deque(maxlen=inference_length)
        self.resize = resize
        self.temporal_length = temporal_length
        self.augmented_frames = np.zeros(shape=(self.temporal_length, self.resize, self.resize, 5), dtype=np.float16)
        # A single past frame
        self.gray_past_frame = None
        self.start_inference = False
        # Model for inference
        self.model = model
        # Counter for frames
        self.processed_frame_counter = 0

    def resize_image(self, img):
        """
        Resizing the image input
        :param img: Image frame to resize
        :return: The resized frame
        """
        return cv2.resize(img, (self.resize, self.resize))

    def preprocess_image(self, img):
        """

        :param img: The image frame that needs to be preprocessed
        :return: The preprocessed image frame
        """
        img = self.resize_image(img)
        img = img / 255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean)/std
        return img

    def make_inference(self, current_frame):

        """

        :param current_frame: The current frame in time for which the inference needs to be made
        :return: A list of inferences depending on the inference length variable
        """

        # Process the current frame
        gray_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_current_frame = self.resize_image(gray_current_frame)

        if self.gray_past_frame is None:
            self.gray_past_frame = gray_current_frame
            return

        mask = np.zeros(shape=(112, 112, 2))
        # Keep the optical flow computation going
        flow_past = cv2.calcOpticalFlowFarneback(
            self.gray_past_frame,
            gray_current_frame,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        # Magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow_past[..., 0], flow_past[..., 1])
        # update mask
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Augmented frame
        preprocessed_frame = self.preprocess_image(current_frame)
        augmented_frame = np.concatenate([preprocessed_frame, mask], axis=-1)
        # Keep record
        self.augmented_frames[:-1] = self.augmented_frames[1:]
        self.augmented_frames[-1] = augmented_frame

        # Increment
        self.processed_frame_counter += 1
        self.start_inference = True if self.processed_frame_counter >= self.temporal_length - 1 else False

        # Update the past gray frame - for optical flow
        self.gray_past_frame = gray_current_frame

        if self.start_inference:

            # Predict on individual instance
            inference_probs = self.model(np.expand_dims(self.augmented_frames, axis=0))
            inference = np.argmax(inference_probs, axis=1)
            # Add to the queue and return queue
            self.inferences.appendleft(inference[0])
            self.inference_probabilities.appendleft(list(inference_probs))

            return self.inferences, self.inference_probabilities

        else:
            return None


# TODO: Check to see if the below works
class C3DHydra:

    def __init__(self, model, inference_length=10, rgb_resize=112, of_resize=112*2, temporal_length=16):

        """
        The initialization for the real-time optical flow inference
        :param model: The model used in the inference process
        :param inference_length: The length for majority voting. Default to 10
        :param resize: The resize for the image input. Depends on the model used
        :param temporal_length: The length in frames to consider to make the inference
        """

        # Initialize
        self.inferences = deque(maxlen=inference_length)
        self.inference_probabilities = deque(maxlen=inference_length)
        self.rgb_resize = rgb_resize
        self.of_resize = of_resize
        self.temporal_length = temporal_length

        # Temporal data store
        self.rgb_augmented_frames_inp1 = np.zeros(shape=(self.rgb_resize, self.rgb_resize, 3, self.temporal_length),
                                                  dtype=np.float16)
        self.of_augmented_frames_inp0 = np.zeros(shape=(self.of_resize, self.of_resize, 2, self.temporal_length),
                                                 dtype=np.float16)

        # A single past frame
        self.gray_past_frame = None
        self.start_inference = False
        # Model for inference
        self.model = model
        # Counter for frames
        self.processed_frame_counter = 0

    def resize_rgb_frames(self, frame):

        return cv2.resize(frame, (self.rgb_resize, self.rgb_resize))

    def resize_of_frames(self, frame):

        return cv2.resize(frame, (self.of_resize, self.of_resize))

    def preprocess_image(self, img):

        """
        :param img: The image frame that needs to be preprocessed
        :return: The preprocessed image frame
        """
        img = self.resize_rgb_frames(img)
        img = img / 255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean)/std
        return img

    def make_inference(self, current_frame):

        """

        :param current_frame: The current frame in time for which the inference needs to be made
        :return: A list of inferences depending on the inference length variable
        """

        # Process the current frame
        gray_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_current_frame = self.resize_of_frames(gray_current_frame)

        if self.gray_past_frame is None:
            self.gray_past_frame = gray_current_frame
            return

        # Get the optical flow required shape and size
        mask = np.zeros(shape=(self.of_resize, self.of_resize, 2))
        # Keep the optical flow computation going
        flow_past = cv2.calcOpticalFlowFarneback(
            self.gray_past_frame,
            gray_current_frame,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        # Magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow_past[..., 0], flow_past[..., 1])
        # update mask
        mask[..., 0] = angle * 180 / np.pi / 2
        mask[..., 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)

        # Get the normal processed RGB frames
        preprocessed_frame = self.preprocess_image(current_frame)

        # Create the model inputs
        self.of_augmented_frames_inp0[..., :-1] = self.of_augmented_frames_inp0[..., 1:]
        self.of_augmented_frames_inp0[..., -1] = mask
        self.rgb_augmented_frames_inp1[..., :-1] = self.rgb_augmented_frames_inp1[..., 1:]
        self.rgb_augmented_frames_inp1[..., -1] = preprocessed_frame

        # Increment
        self.processed_frame_counter += 1
        self.start_inference = True if self.processed_frame_counter >= self.temporal_length - 1 else False

        # Update the past gray frame - for optical flow
        self.gray_past_frame = gray_current_frame

        if self.start_inference:
            # Construct the inputs for model
            inference_inputs = {
                "input_1": np.expand_dims(self.of_augmented_frames_inp0, axis=0),
                "input_2": np.expand_dims(self.rgb_augmented_frames_inp1, axis=0)
            }
            # Predict on a single instance
            inference_probs = self.model(inference_inputs)
            inference = np.argmax(inference_probs, axis=1)
            # Add to the queue and return queue
            self.inferences.appendleft(inference[0])
            self.inference_probabilities.appendleft(list(inference_probs))

            return self.inferences, self.inference_probabilities

        else:
            return None
