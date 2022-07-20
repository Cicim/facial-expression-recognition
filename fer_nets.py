# Neural networks for facial expression recognition
import torch.nn as nn

from neural_net import NeuralNet


class CNN_FER_1(NeuralNet):
    def __init__(self):
        super().__init__()

        # Input size: 1x48x48
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
            nn.Linear(128, 7),
        )

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.convolutional(x)
        x = self.linear(x)
        return x

class Face_Emotion_CNN(NeuralNet):
    def __init__(self):
        super().__init__()

        self.loss_function = nn.MSELoss()

        self.convolutional = nn.Sequential(
            #1
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 1),
            nn.ReLU(),
            #2
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.Dropout(0.3),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 1),
            nn.ReLU(),
            #3
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 1),
            nn.ReLU(),
            #4
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.Dropout(0.3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 1),
            nn.ReLU(),
            #5
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            #6
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.Dropout(0.3),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            #7
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3),
            nn.Dropout(0.3),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )

        self.linear = nn.Sequential(
            nn.Flatten(),

            nn.Linear(1024, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(256, 7),
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = self.linear(x)
        return x



# List of neural network classes to choose from
neural_nets: dict[str, NeuralNet] = {
    'cnn1': CNN_FER_1,
    'cnnfer': Face_Emotion_CNN,
}
