from torch import nn


class C3DOpticalFlow(nn.Module):
    """
    The C3D-OpticalFlow model
    """

    def __init__(self, num_classes):
        """
        Initialization

        :param num_classes: The number of classes for the model
        """
        super(C3DOpticalFlow, self).__init__()

        # Define the layers
        # Block-1
        self.conv1 = nn.Conv3d(16, 32, 3, padding=1)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv3d(32, 32, 3, padding=1)
        self.act2 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(2)
        self.dropout1 = nn.Dropout(0.5)

        # Block-2
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1)
        self.act3 = nn.ReLU()
        self.conv4 = nn.Conv3d(64, 64, 3, padding=1)
        self.act4 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.dropout2 = nn.Dropout(0.5)

        # Flattening
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(100352, 128)
        self.act5 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)

        # The output layer
        self.dense2 = nn.Linear(128, num_classes)

    def forward(self, x):

        """

        :param x: The forward pass
        :return: The model with logits for output
        """

        # C3D-OpticalFLow network
        # Block-1
        out = self.dropout1(self.pool1(self.act2(self.conv2(self.act1(self.conv1(x))))))
        # Block-2
        out = self.dropout2(self.pool2(self.act4(self.conv4(self.act3(self.conv3(out))))))
        # Block-3
        out = self.dropout3(self.act5(self.dense1(self.flatten(out))))
        out = self.dense2(out)

        return out

