import os
import cv2
import numpy as np
import torch
from collections import deque
from intergrated_model import IntegratedInference
from utils.general import non_max_suppression


class InferenceMachine:

    def __init__(self, obd_model_weight_path, al_model_weight_path, inference_length=10, resize=112, temporal_length=16,
                gpu_ids=(0,), obd_img_size=640, obd_stride=32):

        """

        :param obd_model_weight_path:
        :param al_model_weight_path:
        :param inference_length:
        :param resize:
        :param temporal_length:
        :param gpu_ids:
        :return:
        """

        # Select the GPU based on ID
        assert isinstance(gpu_ids, tuple), "gpu_ids should be a tuple"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)
        self.device = torch.device("cuda")
        print(f"Training on device {self.device}.")

        # Init Parameters
        self.resize = resize
        self.inference_length = inference_length
        self.temporal_length = temporal_length
        self.obd_img_size = obd_img_size
        self.obd_stride = obd_stride
        self.start_inference = False
        # Counter for frames
        self.processed_frame_counter = 0
        # Action-Localization
        self.gray_past_frame = None
        self.augmented_frames = np.zeros(shape=(self.temporal_length, self.resize, self.resize, 5), dtype=np.float16)
        self.inferences = deque(maxlen=inference_length)
        self.inference_logits = deque(maxlen=inference_length)

        # Load the model
        self.model = IntegratedInference(obd_model_weight_path, al_model_weight_path, gpu_ids)

        # Class Information
        # Object-Detection model
        self.obd_classes = {}
        with open(os.path.join(os.path.dirname(obd_model_weight_path), "classes.txt")) as file_handle:
            for index, line in enumerate(file_handle.readlines()):
                self.obd_classes[index] = line

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

    @staticmethod
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True,
                  stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def obd_process_image(self, img0):

        # Preprocess
        img = self.letterbox(img0, self.obd_img_size, stride=self.obd_stride)[0]

        # Convert to PyTorch yolov7 requirement
        img = img[:, :, ::-1].transpose(2, 0, 1) # BGR to RGB and channel-first
        img = np.ascontiguousarray(img)

        return img, img0

    def make_inference(self, current_frame):

        """

        :param current_frame: The current frame in time for which the inference needs to be made
        :return:
        """
        '''
        Action Localization
        '''
        # Process the current frame
        gray_current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        gray_current_frame = self.resize_image(gray_current_frame)

        if self.gray_past_frame is None:
            self.gray_past_frame = gray_current_frame
            return

        mask = np.zeros(shape=(self.resize, self.resize, 2))
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
        self.start_inference = True if self.processed_frame_counter >= self.temporal_length else False

        # Update the past gray frame - for optical flow
        self.gray_past_frame = gray_current_frame

        # Start inference after a certain number of frames
        if self.start_inference:
            '''
            Object Detection Data
            '''
            img, img0 = self.obd_process_image(current_frame)
            img = torch.from_numpy(img).to(self.device)
            img = img.half()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            '''
            Action-Localization Data
            '''
            # Reshape as required
            gpu_input = torch.permute(torch.from_numpy(self.augmented_frames), dims=(0, 3, 1, 2)).float().\
                to(device=self.device).unsqueeze(0)
            # gpu_input = self.augmented_frames.float().to(device=self.device).unsqueeze(0)

            # Infer
            with torch.no_grad():
                inferences = self.model([gpu_input, img])
                # Separate the outputs
                al_output, obd_output = inferences
                # AL-Inference
                _, al_inference = torch.max(al_output, dim=1)
            # Add to queue
            self.inferences.appendleft(al_inference.item())
            self.inference_logits.appendleft(al_output.cpu().numpy())

            # Object Detection Inference
            obd_output = non_max_suppression(obd_output)

            return self.inferences, self.inference_logits, (obd_output[0], img, img0)

        else:
            return None






