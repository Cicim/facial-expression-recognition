# Data loaded contains a collection of utilties for transforming samples
# from different sources into the same format as the FER2013 dataset.
# All samples will be stored in a custom .samples file.
import random

from helpers import TimeIt, EMOTIONS

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

from torch.utils.data import TensorDataset, DataLoader

## Dataset creation
def convert_fer2013_dataset(filename='datasets/fer2013.csv'):
    """
    Convert the FER2013 dataset to a pandas dataframe.
    """
    with TimeIt('Reading fer2013.csv as dataframe'):
        df = pd.read_csv(filename)

    with TimeIt('Converting images to binary'):
        df['pixels'] = df['pixels'].apply(
            lambda x: bytes([int(i) for i in x.split()]))

    with TimeIt('Converting labels to bytes'):
        df['emotion'] = df['emotion'].apply(lambda x: x.to_bytes(1, 'big'))

    with TimeIt('Splitting samples'):
        train_df = df[df['Usage'] == 'Training']
        valid_df = df[df['Usage'] == 'PublicTest']
        test_df = df[df['Usage'] == 'PrivateTest']

    with TimeIt('Removing the "Usage" column'):
        train_df = train_df.drop(columns=['Usage'])
        valid_df = valid_df.drop(columns=['Usage'])
        test_df = test_df.drop(columns=['Usage'])

    # Save the dataframes to a file
    with TimeIt('Creating fer2013_train.samples'):
        with open('datasets/fer2013_train.samples', 'wb') as f:
            f.write(len(train_df).to_bytes(4, 'little'))
            for row in train_df.itertuples():
                f.write(row.pixels + row.emotion)

    with TimeIt('Creating fer2013_valid.samples'):
        with open('datasets/fer2013_valid.samples', 'wb') as f:
            f.write(len(valid_df).to_bytes(4, 'little'))
            for row in valid_df.itertuples():
                f.write(row.pixels + row.emotion)

    with TimeIt('Creating fer2013_test.samples'):
        with open('datasets/fer2013_test.samples', 'wb') as f:
            f.write(len(test_df).to_bytes(4, 'little'))
            for row in test_df.itertuples():
                f.write(row.pixels + row.emotion)


## Dataset loading
NORMALIZATION_MEAN = 0.5077
NORMALIZATION_STD = 0.255

def transform_sample(image: bytes, label: bytes) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transforms the image and label bytearrays into Tensors.
    """

    # Convert the bytes to a 48x48 float32 numpy array
    nparray = np.frombuffer(image, dtype=np.uint8, count=48*48)
    nparray = np.reshape(nparray, (48, 48))

    # # Normalize the exposure of the image
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(1, 1))
    # nparray = clahe.apply(nparray)

    # # Detect the edges of the image
    # nparray = cv2.Canny(nparray, 100, 200)

    # If the image is all black
    if np.all(nparray == 0):
        return None, None

    nparray = nparray.astype(np.float32) / 255
    # Then to a tensor
    tensor = torch.from_numpy(nparray).unsqueeze(0)

    # Read the label as an integer
    label = int.from_bytes(label, 'big')
    # Convert the label to a one-hot tensor of size len(EMOTIONS)
    target = torch.zeros(len(EMOTIONS))
    target[label] = 1

    # NOTE These are additional steps to make the dataset compatible with a
    #      specific neural network, and may change in the future.
    # Delete the external 4 pixel wide border
    # tensor = tensor[:, 4:-4, 4:-4]
    # # Normalize the image
    # tensor = tensor.sub(NORMALIZATION_MEAN).div(NORMALIZATION_STD)

    return tensor, target
    
def load_dataset(samples_file: str, limit: int = None):
    """
    Create a samples dataset from a .samples file.
    `limit` limits the numer of samples to load.
    """
    if limit is None:
        limit = np.inf

    try:
        inputs = []
        targets = []

        with open(samples_file, 'rb') as f:
            # Read the first 4 bytes to get the number of samples
            num_samples = int.from_bytes(f.read(4), 'little')

            # Read each image
            for _ in tqdm(range(min(num_samples, limit))):
                # Read the image
                image = f.read(48 * 48)
                # If the image is empty, break
                if not image:
                    break
                label = f.read(1)

                # Transform the sample
                input, target = transform_sample(image, label)
                if input is not None and target is not None:
                    inputs.append(input)
                    targets.append(target)

    except FileNotFoundError:
        print(f'Samples file `{samples_file}` not found.')
        exit(1)

    return TensorDataset(torch.stack(inputs), torch.stack(targets))

def random_samples_plot(dataset: TensorDataset, num_samples: int = 128, cols: int = 16):
    """
    Plots a random subset of the given tensor
    """
    rows = np.ceil(num_samples / cols).astype(int)

    # Create a random sample of the dataset
    samples = DataLoader(dataset, batch_size=None, shuffle=True)

    fig, axs = plt.subplots(rows, cols, figsize=(20, 12))
    # Plot the samples
    for i, (image, target) in tqdm(enumerate(samples)):
        if i == num_samples:
            break
        ax = axs[i // cols, i % cols]
        # Show the image
        ax.imshow(image.squeeze().numpy(), cmap='gray')
        # Add a title with the emotion
        emotion = torch.argmax(target).item()
        ax.set_title(EMOTIONS[emotion])
        ax.axis('off')


## Dataset combining
def split_samples(src_path: str, weights: list[float], dest_paths: list[str]):
    """
    Splits the given file into multiple files (with the given path).
    Each file will get a weights percentage of the original file's samples.
    """
    # Load the samples in bytes
    samples = []
    with open(src_path, 'rb') as f:
        size = int.from_bytes(f.read(4), 'little')
        for _ in range(size):
            samples.append(f.read(48 * 48 + 1))
    
    # Create a random permutation
    indices = torch.randperm(size)
    # Randomize the sample's order
    samples = [samples[i] for i in indices]

    # Split the samples into the given files
    for dest_path, weight in zip(dest_paths, weights):
        # Get the first % of the samples
        last_index = int(size * weight)
        file_samples = samples[:last_index]
        samples = samples[last_index:]

        # Write the samples to the file
        with open(dest_path, 'wb') as f:
            f.write(len(file_samples).to_bytes(4, 'little'))
            for sample in file_samples:
                f.write(sample)

def merge_samples(dst_path: str, src_paths: list[str]):
    """
    Merge multiple samples files into a single file.
    """
    # Open all the source files in binary read mode
    files = [open(src_path, 'rb') for src_path in src_paths]
    # Read the first 4 bytes to get the number of samples
    sizes = [int.from_bytes(f.read(4), 'little') for f in files]
    # Create a file to write the merged samples
    with open(dst_path, 'wb') as fw:
        # Write the number of samples
        fw.write(sum(sizes).to_bytes(4, 'little'))
        # Write the samples
        for size, fr in zip(sizes, files):
            for _ in range(size):
                fw.write(fr.read(48*48+1))

    # Close all the files
    for f in files:
        f.close()

## Dataset statistics
def show_dataset_stats(dataset: TensorDataset):
    """
    Get the statistics of the given dataset.
    """
    # Get the statistics of the dataset
    inputs, targets = dataset[:]
    # Get the mean and std of the inputs
    mean = inputs.mean()
    std = inputs.std()
    # Get the number of samples
    num_samples = inputs.shape[0]
    # Get the number of classes
    num_classes = targets.shape[1]
    # Get the number of samples per class
    num_samples_per_class = torch.sum(targets, 0).tolist()

    print(f'Number of samples: {num_samples}')
    print(f'Number of classes: {num_classes}')
    print(f'Number of samples per class: {num_samples_per_class}')
    print(f'Mean: {mean}')
    print(f'Std: {std}')


## Dataset transformations
def equalize_dataset(dst_path: str, src_path: str, samples_per_class: int, discarded_path: str = None, repeat_samples: bool = True):
    """
    Make sure the number of samples for each class is the same
    """
    # Read the samples into an array
    samples: list[tuple[bytes, int]] = []
    with open(src_path, 'rb') as f:
        size = int.from_bytes(f.read(4), 'little')
        for _ in range(size):
            samples.append((f.read(48 * 48), int.from_bytes(f.read(1), 'little')))
    
    # Divide them into targets
    samples_by_targets: list[list[tuple[bytes, int]]] = [[] for i in range(len(EMOTIONS))]
    for image, target in samples:
        samples_by_targets[target].append((image, target))

    # Statistics
    print("From:")
    for emotion, samples in enumerate(samples_by_targets):
        print(f'{EMOTIONS[emotion]:11}: {len(samples):5}')

    discarded_samples: list[tuple[bytes, int]] = []

    # For each emotion, correct the number of samples to reach samples_per_class
    for emotion, samples in enumerate(samples_by_targets):
        # Get the number of samples for this emotion
        num_samples = len(samples)
        
        if num_samples > samples_per_class:
            # Remove some samples
            new_samples = random.sample(samples, samples_per_class)

            if discarded_path is not None:
                # Get the samples that are in samples but not in new_samples
                discarded_samples.extend([s for s in samples if s not in new_samples])

            samples_by_targets[emotion] = new_samples
        elif repeat_samples:
            new_samples: list[tuple[bytes, int]] = []

            # Repeat the samples until the number of samples is equal to samples_per_class
            while len(new_samples) < samples_per_class:
                # Extract a random sample from samples
                sample = random.choice(samples)
                # Add it to the new samples
                new_samples.append(sample)
            # Set the new samples
            samples_by_targets[emotion] = new_samples

    if discarded_path is not None:
        # Write the discarded samples to the discarded_path
        with open(discarded_path, 'wb') as f:
            f.write(len(discarded_samples).to_bytes(4, 'little'))
            for image, target in discarded_samples:
                f.write(image)
                f.write(target.to_bytes(1, 'little'))
    
    # Write the new samples to the dst_path
    with open(dst_path, 'wb') as f:
        f.write(sum(map(len, samples_by_targets)).to_bytes(4, 'little'))
        for samples in samples_by_targets:
            for image, target in samples:
                f.write(image)
                f.write(target.to_bytes(1, 'little'))

    
    # Statistics
    print("To:")
    for emotion, samples in enumerate(samples_by_targets):
        print(f'{EMOTIONS[emotion]:11}: {len(samples):5}')
        

# equalize_dataset('datasets/equalized_fer.samples', 'datasets/fer2013plus_train.samples', 3000, discarded_path='datasets/equalization_losses.samples', repeat_samples=False)
