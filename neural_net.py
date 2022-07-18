# This is the file containing the neural network structure
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from PIL import Image

from helpers import EMOTIONS, clear_line

## MTCNN face finder
mtcnn = None
def find_faces(image: Image.Image):
    """
    Load the MTCNN network and extract the faces from the image,
    with the relative position boxes. Return None if you don't find faces.
    """
    global mtcnn

    if mtcnn is None:
        from facenet_pytorch import MTCNN
        # Get the GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize the MTCNN network
        mtcnn = MTCNN(
            thresholds=[0.6, 0.7, 0.7],
            min_face_size=48,
            device=device,
        )
    
    # Detect faces
    boxes, probabilities = mtcnn.detect(image)
    if boxes is None:
        return None, None

    # Get the faces
    faces = []
    new_boxes = []
    for box, probability in zip(boxes, probabilities):
        if probability < 0.95:
            continue
        x1, y1, x2, y2 = list(map(int, list(box)))

        w = x2 - x1
        h = y2 - y1

        if w < 48 or h < 48:
            continue

        # Make sure the image is a square
        if w > h:
            y1 -= (w - h) // 2
            y2 += (w - h) // 2
        elif h > w:
            x1 -= (h - w) // 2
            x2 += (h - w) // 2
        # Print the square ratio
        
        # Get the face
        face = image.crop((x1, y1, x2, y2))
        # Resize the image to 48x48
        face = face.resize((48, 48), Image.BICUBIC)
        # Convert it to grayscale
        face = face.convert('L')

        faces.append(face)
        new_boxes.append((x1, y1, x2, y2))

    if len(faces) == 0:
        return None, None
    
    return faces, new_boxes


## Classes
class NeuralNet(nn.Module):
    """
    Class for everything needed in every model of the neural network
    """
    loss_function: nn.CrossEntropyLoss

    @classmethod
    def load(cls, path: str):
        """
        Loads a model from a file
        """
        model = cls()
        model.load_state_dict(torch.load(path))
        return model

    def predict_from_image(self, image) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
        """
        Use this network to predict the emotion of people given a photo
        """
        # Convert the image to a PIL Image
        if type(image) == str:
            image = Image.open(image)
        elif type(image) == np.ndarray:
            image = Image.fromarray(image)
        elif type(image) == torch.Tensor:
            image = Image.fromarray(image.numpy())

        # Find the image to RGB
        image = image.convert('RGB')

        # Find the faces in the image
        faces, boxes = find_faces(image)
        if faces is None:
            return None, None
        
        # Convert all faces to tensors
        faces = [torch.from_numpy(np.array(face)).float().unsqueeze(0) for face in faces]
        # Then to a batched tensor
        faces = torch.stack([face for face in faces])
        # Bring the range to 0,1
        faces = faces / 255
        # Move everything to the gpu
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        faces_gpu = faces.to(device)
        network_gpu = self.to(device)

        # Apply the network on the dataset
        with torch.no_grad():
            output = network_gpu(faces_gpu)

        return output.cpu(), boxes


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
        return np.mean(self.training_loss_per_batch)
    
    @property
    def training_accuracy(self):
        return np.mean(self.training_accuracy_per_batch)

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
    correct = sum(max_output == max_target).item()
    return correct / target.shape[0]

def train(network: NeuralNet, training_data: TensorDataset, validation_data: TensorDataset, 
          model_save_path: str = None, epochs: int = 50, batch_size: int = 256, learning_rate: float = 0.001,
          print_to_screen: bool = True) -> list[EpochStats]:
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
    optimizer = optim.Adam(network.parameters(), lr=learning_rate, weight_decay=1e-3)

    # Get the loss function
    loss_fn = network.loss_function

    # Create a list of epoch stats
    epoch_stats: list[EpochStats] = []

    try:
        # Training loop
        for epoch in range(epochs):
            # Start the timer
            start = perf_counter()

            # Create the data loader for the training data
            training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, drop_last=True)

            # Create an epoch stat object
            epoch_stat = EpochStats(epoch)

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
            validation_loader = DataLoader(validation_data, batch_size=batch_size)
            # Calculate the validation loss and accuracy
            val_losses = []
            val_accuracies = []
            with torch.no_grad():
                for x, d in validation_loader:
                    x = x.to(device)
                    d = d.to(device)
                    y = network(x)
                    loss = loss_fn(y, d)
                    val_losses.append(loss.item())
                    val_accuracies.append(compute_accuracy(y, d))
            epoch_stat.validation_loss = np.mean(val_losses)
            epoch_stat.validation_accuracy = np.mean(val_accuracies)

            # Save the model to file
            if model_save_path is not None:
                # Split the path into the filename and extension
                filename, extension = model_save_path.split(".")
                # Add the validation accuracy as an integer to the filename
                filename += f"_{int(epoch_stat.validation_accuracy * 100):03}"
                # Add the extension back to the filename
                filename += f".{extension}"
                
                torch.save(network.state_dict(), filename)

            # Stop the timer
            epoch_stat.time = perf_counter() - start
            epoch_stats.append(epoch_stat)

            # Show the epochs stats
            if print_to_screen:
                clear_line()
                print(epoch_stat)

        print(f"Network trained in {perf_counter() - training_start_time:.2f} seconds")

        return epoch_stats
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        return epoch_stats

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

def test(network: NeuralNet, validation_data: TensorDataset, batch_size: int = 1):
    """
    Tests the `network` on the `validation_data`.
    """
    # Get the GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Testing network on {device}")
    network = network.to(device)

    # Create the data loader for the validation data
    validation_loader = DataLoader(validation_data, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=True)

    # Create the confusion matrix
    cm = np.zeros((len(EMOTIONS), len(EMOTIONS)), dtype=np.int64)

    correct_sum = 0
    loss_sum = 0
    total = 0
    with torch.no_grad():
        # For each element in the validation data
        for x, d in tqdm(validation_loader):
            # Move the data to the GPU
            x = x.to(device)
            d = d.to(device)
            # Forward pass
            y = network(x)
            # Calculate the loss
            loss = network.loss_function(y, d)
            
            # Get the predicted emotions of y and d
            y_emotions = torch.argmax(y, dim=1)
            d_emotions = torch.argmax(d, dim=1)

            # For each emotion in the predicted emotions
            for ye, de in zip(y_emotions, d_emotions):
                # Add 1 to the confusion matrix
                cm[ye][de] += 1
                # If the predicted emotion is the same as the actual emotion
                if ye == de:
                    # Add 1 to the correct sum
                    correct_sum += 1
                # Add the loss to the loss sum
                loss_sum += loss.item()
                total += 1

    print(correct_sum, total)
    accuracy = correct_sum / total
    loss = loss_sum / total

    print(f"Accuracy: {accuracy*100:.1f}%")
    print(f"Loss: {loss:.4f}")
    print_confusion_matrix(cm)
