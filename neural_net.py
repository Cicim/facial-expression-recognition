# This is the file containing the neural network structure
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights

from data_loader import load_samples, EMOTION

# Make sure the gpu is connected
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_name = device.type == 'cuda' and f'GPU ({torch.cuda.get_device_name(0)})' or 'CPU'
torch.cuda.empty_cache()
print(f"Running on {device_name}")

class FacialRecognitionNetwork(nn.Module):
    def __init__(self):
        super().__init__()


        # Input size: 1x48x48
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.linear = nn.Sequential(
            nn.Linear(9216, 3600),
            nn.ReLU(),
            nn.Linear(3600, 3600),
            nn.ReLU(),
            nn.Linear(3600, 440),
            nn.ReLU(),
            nn.Linear(440, 7)
        )

    def forward(self, x):
        x = self.convolutional(x)
        x = self.linear(x)
        return x

    def train(self, training_data_with_desc, save_dest: str = "model.pt", 
              epochs: int = 10, learning_rate: float = 0.001, test=False):
        network = self.to(device)

        # Create the optimizer
        optimizer = optim.Adam(network.parameters(), lr=learning_rate)

        # Create the loss function
        loss_function = nn.CrossEntropyLoss()

        training_data, num_samples, batch_size = training_data_with_desc

        if test:
            testing_data = load_samples('datasets/fer2013_valid.samples', limit=100)

        # Train the model
        for epoch in range(epochs):
            print(f"Epoch {epoch}/{epochs}")
            for i, (data, target) in enumerate(training_data):
                data = data.to(device)
                target = target.to(device)
                # Triple the channels
                optimizer.zero_grad()
                output = network(data)
                loss = loss_function(output, target)
                loss.backward()
                optimizer.step()
                print(f"\r\33[2K\r[{i * batch_size:5}/{num_samples}] Loss: {loss:.3f}", end='')
            # Validate with the validation set
            if test:
                accuracy = self.test(testing_data, print_to_screen=False)
                print(f"\r\33[2K\r[{num_samples:5}/{num_samples}] Loss: {loss:.3f} Validation Accuracy: {accuracy:.2f}")

            torch.save(network.state_dict(), save_dest)



    def test(self, validation_data, print_to_screen: bool = True):
        network = self.to(device)
        # Test the model for validation
        correct = 0
        total = 0

        if print_to_screen:
            confusion_matrix = np.zeros((7, 7), dtype=np.int32)

            print("Validating model...")
            for data, target in tqdm(zip(*validation_data), total=len(validation_data[0])):
                data = data.view(-1, 1, 48, 48)
                data = data.to(device)
                output = network(data)
                # Get the highest index in the output
                _, predicted = torch.max(output.data, 1)

                total += 1
                correct += int(predicted == target)

                confusion_matrix[predicted][target] += 1
            
            print(f"Accuracy: {correct/total*100:.2f}%")
            # Print the confusion matrix
            print_confusion_matrix(confusion_matrix)

        else:
            for data, target in zip(*validation_data):
                data = data.view(-1, 1, 48, 48)
                data = data.to(device)
                output = network(data)
                # Get the highest index in the output
                _, predicted = torch.max(output.data, 1)

                total += 1
                correct += int(predicted == target)

            return correct / total

def prepare_training_data(samples_file: str, batch_size: int = 2000, sample_limit = None):
    # Load the training data
    tensor, int_emotion = load_samples(samples_file, limit=sample_limit)

    num_samples = len(tensor)
    num_batches = int(np.ceil(num_samples / batch_size))

    # Convert all emotion to a one-hot vector
    emotion = torch.zeros(len(int_emotion), 7)
    for i, emotion_index in enumerate(int_emotion):
        emotion[i][emotion_index] = 1

    # Shuffle the data
    shuffled_indices = torch.randperm(len(tensor))
    tensor = tensor[shuffled_indices]
    emotion = emotion[shuffled_indices]

    batches = []
    for i in range(num_batches):
        batches.append((tensor[i*batch_size:(i+1)*batch_size], emotion[i*batch_size:(i+1)*batch_size]))
    return batches, num_samples, batch_size

def print_confusion_matrix(cm: np.ndarray):
    SHORTER = list(map(lambda x: x.lower()[0:3], EMOTION))
    header = ' ' + '  '.join(SHORTER) + '  ptot '
    print("     " + 'Actual'.center(len(header)))
    print("     " + header)
    for i, head in enumerate(SHORTER):
        print(head + ' ', end='')
        
        # total = np.sum(cm[i])
        for el in cm[i]:
            # if total == 0:
            #     intensity = 0
            # else:
            #     ratio = np.sqrt(el / total)
            #     intensity = int(ratio * 255)

            # rgb_bg = (intensity, 0, intensity // 3)
            # rgb_fg = (255 - intensity, 255, 255)

            # bg_ansi = f'\x1b[48;2;{rgb_bg[0]};{rgb_bg[1]};{rgb_bg[2]}m'
            # fg_ansi = f'\x1b[38;2;{rgb_fg[0]};{rgb_fg[1]};{rgb_fg[2]}m'

            # print(bg_ansi + fg_ansi + str(el).rjust(5), end='\033[0m')
            print(str(el).rjust(5), end='')
            
        print(str(np.sum(cm[i])).rjust(5))
    print('atot', end='')
    for i, head in enumerate(SHORTER):
        print(str(np.sum(cm[:, i])).rjust(5), end='')
    print()
