# Neural networks for facial expression recognition
import torch.nn as nn

from neural_net import NeuralNet


class CNNFER1(NeuralNet):
    def __init__(self):
        super().__init__()

        # Input size: 1x40x40
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(4096, 7),
        )

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.convolutional(x)
        x = self.linear(x)
        return x


class CNNFER2(NeuralNet):
    def __init__(self):
        super().__init__()

        # Input size: 1x40x40
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=0),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=0),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=5, padding=0),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 7),
        )

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.convolutional(x)
        x = self.linear(x)
        return x


class CNNFER3(NeuralNet):
    def __init__(self):
        super().__init__()

        # Input size: 1x40x40
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4608, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 7)
        )

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.convolutional(x)
        x = self.linear(x)
        return x
