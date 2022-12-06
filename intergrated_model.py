import os
import torch
from torch import nn
from models.experimental import Ensemble
from models.common import Conv
from action_localization.models.c3d_model import C3DOpticalFlow


class IntegratedInference(nn.Module):

    def __init__(self, obd_model_weight_path, al_model_weight_path, gpu_ids=(0,)):

        """

        :param obd_model_weight_path:
        :param al_model_weight_path:
        """

        super(IntegratedInference, self).__init__()
        self.obd_model_weight_path = obd_model_weight_path
        self.al_model_weight_path = al_model_weight_path

        # Select the GPU based on ID
        assert isinstance(gpu_ids, tuple), "gpu_ids should be a tuple"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)
        self.device = torch.device("cuda")
        print(f"Training on device {self.device}.")

        # Loading the models
        self.obd_model = None
        self.obd_model_init()
        self.al_model = None
        self.al_model_init()

    def obd_model_init(self):

        """
        Initialize and load the object detection model
        :return: None
        """

        self.obd_model = self.load_obd_model(self.obd_model_weight_path, self.device)
        # If using GPU
        self.obd_model.half()

    def al_model_init(self):

        """
        Initialize and load the action-localization model
        :return: None
        """

        # Load the model
        al_model_params = torch.load(self.al_model_weight_path, map_location=self.device)
        self.al_model = C3DOpticalFlow(num_classes=al_model_params["num_classes"]).to(device=self.device)
        self.al_model.load_state_dict(al_model_params["model_state_dict"])
        self.al_model.eval()

    @staticmethod
    def load_obd_model(obd_model_weight_path, device):

        # Create an Ensemble model
        obd_model = Ensemble()

        # Load the weights
        ckpt = torch.load(obd_model_weight_path, map_location=device)
        obd_model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().fuse().eval())

        # Compatibility resolution
        for m in obd_model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

        if len(obd_model) == 1:
            return obd_model[-1]
        else:
            print("Ensemble created with %s\n" % obd_model_weight_path)
            for k in ['names', 'stride']:
                setattr(obd_model, k, getattr(obd_model[-1], k))
            return obd_model

    def forward(self, inputs):

        """
        The forward run for the model

        :param inputs: The model inputs
        :return:
        """

        out0 = self.al_model(inputs[0])
        out1 = self.obd_model(inputs[1])[0]

        return out0, out1
