# This is the file containing the neural network structure
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from helpers import EMOTIONS, clear_line

## Classes
class NeuralNet(nn.Module):
    """
    Class for everything needed in every model of the neural network
    """
    loss_function: nn.CrossEntropyLoss

class EpochStats():
    """
    Collection of statistics for an epoch of training.
    """
    def __init__(self, epoch: int):
        self.epoch = epoch + 1
        self.time = 0.0

        self.training_loss_per_batch: list[float] = []
        self.training_accuracy_per_batch: list[float] = []
        self.validation_loss = 0.0
        self.validation_accuracy = 0.0

    @property
    def training_loss(self):
        return sum(self.training_loss_per_batch) / len(self.training_loss_per_batch)
    
    @property
    def training_accuracy(self):
        return sum(self.training_accuracy_per_batch) / len(self.training_accuracy_per_batch)

    def __str__(self):
        val_acc = self.validation_accuracy * 100
        tr_acc = self.training_accuracy * 100
        return f"Epoch {self.epoch:3} [{self.time:5.0f}s]: train_loss: {self.training_loss:.4f}, train_acc: {tr_acc:.1f}%, val_loss: {self.validation_loss:.4f}, val_acc: {val_acc:.1f}%"


## Training methods
def compute_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """
    Returns the accuracy of the model on the given outputs.
    """
    max_output = torch.argmax(output, dim=1)
    max_target = torch.argmax(target, dim=1)
    correct = torch.sum(max_output == max_target).item()
    return correct / len(target)

def train(network: NeuralNet, training_data: TensorDataset, validation_data: TensorDataset, 
          model_save_path: str = None, epochs: int = 50, batch_size: int = 256, learning_rate: float = 0.001,
          print_to_screen: bool = True):
    """
    Trains the `network` on the `training_data` and validates on the `validation_data`.
    Saves the results to `model_save_path` for each epoch if given.
    """
    # Get the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training network on {device}")

    training_start_time = perf_counter()

    # Copy the network to the GPU
    network = network.to(device)

    # Get the optimizer
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    # TODO: add a scheduler

    # Get the loss function
    loss_fn = network.loss_function

    # Create a list of epoch stats
    epoch_stats: list[EpochStats] = []

    # Training loop
    for epoch in range(epochs):
        # Start the timer
        start = perf_counter()

        # Create the data loader for the training data
        training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)

        # Create an epoch stat object
        epoch_stat = EpochStats(epoch)
        epoch_stats.append(epoch_stat)

        # Train the model
        for batch, (x, d) in enumerate(training_loader):
            # Move the data to the GPU
            x = x.to(device)
            d = d.to(device)

            optimizer.zero_grad()
            # Forward pass
            y = network(x)
            # Calculate the loss
            loss = loss_fn(y, d)
            # Backward pass
            loss.backward()
            # Update the weights
            optimizer.step()

            # Save the loss for plotting
            loss_avg = loss.item()
            # Calculate the training accuracy for plotting
            training_accuracy = compute_accuracy(y, d)

            # Save the loss and accuracy for this batch
            epoch_stat.training_loss_per_batch.append(loss_avg)
            epoch_stat.training_accuracy_per_batch.append(training_accuracy)

            # Show the loss and accuracy for this batch
            if print_to_screen:
                clear_line()
                print(f"Epoch {epoch+1:3}/{epochs} | Batch {batch:4}/{len(training_loader)} | Loss: {loss_avg:.4f} | Accuracy: {training_accuracy*100:.1f}%", end='')

        # Create the data loader for the validation data
        validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, drop_last=True)
        # Calculate the validation loss and accuracy
        val_loss = 0
        val_acc = 0
        with torch.no_grad():
            for x, d in validation_loader:
                x = x.to(device)
                d = d.to(device)
                y = network(x)
                loss = loss_fn(y, d)
                val_loss += loss.item()
                val_acc += compute_accuracy(y, d)
        epoch_stat.validation_loss = val_loss / len(validation_loader)
        epoch_stat.validation_accuracy = val_acc / len(validation_loader)

        # Save the model to file
        if model_save_path is not None:
            torch.save(network.state_dict(), model_save_path)

        # Stop the timer
        epoch_stat.time = perf_counter() - start

        # Show the epochs stats
        if print_to_screen:
            clear_line()
            print(epoch_stat)

    print(f"Network trained in {perf_counter() - training_start_time} seconds")


def print_confusion_matrix(cm: np.ndarray):
    """
    Prints the confusion matrix of the model.
    """
    SHORTER = list(map(lambda x: x.lower()[0:3], EMOTIONS))
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

def test(network: NeuralNet, validation_data: TensorDataset, print_to_screen: bool = True):
    """
    Tests the `network` on the `validation_data`.
    """
    # Get the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing network on {device}")
    network = network.to(device)

    # Create the data loader for the validation data
    validation_loader = DataLoader(validation_data, batch_size=1, shuffle=True)

    # Create the confusion matrix
    cm = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=np.int64)

    correct_sum = 0
    loss_sum = 0

    with torch.no_grad():
        # For each element in the validation data
        for x, d in validation_loader:
            # Move the data to the GPU
            x = x.to(device)
            d = d.to(device)
            # Forward pass
            y = network(x)
            # Calculate the loss
            loss = network.loss_function(y, d)
            
            # Get the predicted emotions of y and d
            y_emotion = torch.argmax(y[0], dim=0).item()
            d_emotion = torch.argmax(d[0], dim=0).item()
            # Check if the prediction is correct
            correct_sum += int(y_emotion == d_emotion)

            # Add the loss to the loss sum
            loss_sum += loss.item()

            # Add 1 to the confusion matrix at the correct indices
            cm[d_emotion, y_emotion] += 1

    accuracy = correct_sum / len(validation_data)
    loss = loss_sum / len(validation_data)

    print(f"Accuracy: {accuracy*100:.1f}%")
    print(f"Loss: {loss:.4f}")
    print_confusion_matrix(cm)
        