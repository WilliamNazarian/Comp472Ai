import torch.nn as nn
import torch.nn.functional as F


class OB_05Model(nn.Module):
    def __init__(self):
        super(OB_05Model, self).__init__()

        # Convolution Layer 1
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)  # 28 x 28 x 1 -> 24 x 24 x 20
        self.relu1 = nn.ReLU()  # Activation function

        # Convolution Layer 2
        self.conv2 = nn.Conv2d(20, 30, kernel_size=5)  # 24 x 24 x 20 -> 20 x 20 x 30
        self.conv2_drop = nn.Dropout2d(p=0.5)  # Dropout
        self.maxpool2 = nn.MaxPool2d(2)  # Pooling layer 20 x 20 x 30 -> 10 x 10 x 30
        self.relu2 = nn.ReLU()  # Activation function

        # Fully connected layers
        self.fc1 = nn.Linear(3000, 500)  # 10 x 10 x 30 -> 3000 -> 500
        self.fc2 = nn.Linear(500, 4)  # 500 -> 4


        # Had to change the number of nodes in the output layer

    def forward(self, x):
        # Convolution Layer 1
        x = self.conv1(x)
        x = self.relu1(x)

        # Convolution Layer 2
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Fully connected layer 1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        # Fully connected layer 2
        x = self.fc2(x)

        return x
