# This is the file containing the neural network structure
from doctest import script_from_examples
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader import load_samples

# Make sure the gpu is connected
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_name = device.type == 'cuda' and f'GPU ({torch.cuda.get_device_name(0)})' or 'CPU'
torch.cuda.empty_cache()
print(f"Running on {device_name}")

class FacialRecognitionNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # Input size: 1x48x48
        # Convolutional layers
        self.convolutional = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(num_features=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(),
            nn.Flatten(),
        )

        # Linear layers
        self.linear = nn.Sequential(
            nn.Linear(30976//4, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 7)
        )


    def forward(self, x):
        x = self.convolutional(x)
        x = self.linear(x)
        return x

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
    return batches
    
        



def train_model():
    # Load the prepared data
    training_data = prepare_training_data('datasets/fer2013_train.samples', batch_size=1000)
    
    # Load the model from file
    network = FacialRecognitionNetwork()
    network.load_state_dict(torch.load('models/model.pt'), strict=False)
    network.to(device)

    # Create the optimizer
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    # Create the loss function
    loss_function = nn.CrossEntropyLoss()



    # Train the model
    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs}")
        for i, (data, target) in enumerate(training_data):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = network(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
            print(f"Batch {i} -> Loss: {loss}")

    # Save the model
    torch.save(network.state_dict(), 'models/model.pt')
    return network

def test_model(network: FacialRecognitionNetwork):
    network = network.to(device)
    # Test the model for validation
    validation_data = load_samples('datasets/fer2013_valid.samples')
    correct = 0
    total = 0

    confusion_matrix = np.zeros((7, 7), dtype=np.int32)

    print("Validating model...")
    for data, target in tqdm(zip(*validation_data), total=len(validation_data[0])):
        data = data.view(-1, 1, 48, 48).to(device)
        output = network(data)
        # Get the highest index in the output
        _, predicted = torch.max(output.data, 1)

        total += 1
        correct += int(predicted == target)

        confusion_matrix[predicted][target] += 1
    
    print(f"Validation Accuracy: {correct/total}")
    # Show the confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix)


    # Test on the training dataset
    training_data = load_samples('datasets/fer2013_train.samples')
    correct = 0
    total = 0

    confusion_matrix = np.zeros((7, 7), dtype=np.int32)

    print("Validating model...")
    for data, target in tqdm(zip(*training_data), total=len(training_data[0])):
        data = data.view(-1, 1, 48, 48).to(device)
        output = network(data)
        # Get the highest index in the output
        _, predicted = torch.max(output.data, 1)

        total += 1
        correct += int(predicted == target)

        confusion_matrix[predicted][target] += 1

    print(f"Training Accuracy: {correct/total}")
    # Show the confusion matrix
    print("Confusion matrix:")
    print(confusion_matrix)


network = train_model()
# network = FacialRecognitionNetwork()
# network.load_state_dict(torch.load('models/model.pt'), strict=False)
test_model(network)