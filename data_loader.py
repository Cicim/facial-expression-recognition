# Data loaded contains a collection of utilties for transforming samples
# from different sources into the same format as the FER2013 dataset.
# All samples will be stored in a custom .samples file.

from time import perf_counter
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

EMOTION = ['Anger', 'Disgust', 'Fear', 'Happiness',
           'Sadness', 'Surprise', 'Neutrality']


class TimeIt:
    """
    Context manager for timing code.
    """

    def __init__(self, message: str):
        self.message = message
        self.start_time = perf_counter()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        print(f'[{perf_counter() - self.start_time:8.2f}s] {self.message}')
        return False


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


def load_samples_generator(samples_file: str):
    """
    Loads samples from a file.
    """
    try:
        with open(samples_file, 'rb') as f:
            # Read the first 4 bytes to get the number of samples
            num_samples = int.from_bytes(f.read(4), 'little')
            yield num_samples

            # Read each image
            while True:
                # Read the image
                image = f.read(48 * 48)
                # Read the label
                label = int.from_bytes(f.read(1), 'big')
                # If the image is empty, break
                if not image:
                    break

                # Convert the bytes to a 48x48 float32 numpy array
                nparray = np.frombuffer(image, dtype=np.uint8, count=48*48)
                nparray = np.reshape(nparray, (48, 48))
                nparray = nparray.astype(np.float32) / 255

                # Then to a tensor
                tensor = torch.from_numpy(nparray).view(1, 48, 48)

                # Convert the image to a numpy array
                yield tensor, label

    except FileNotFoundError:
        print(f'{samples_file} not found.')
        return None

def load_samples(samples_file: str, limit: int = None):
    # Load the samples
    samples = load_samples_generator(samples_file)
    # Get the number of samples
    num_samples = next(samples) if limit is None else min(next(samples), limit)
    # Create a tensor to store the samples
    tensor = torch.zeros(num_samples, 1, 48, 48)
    # Create a tensor to store the labels
    labels = torch.zeros(num_samples, dtype=torch.long)
    # Load the samples
    print(f"Loading samples from {samples_file}")
    for i, (sample, label) in tqdm(enumerate(samples), total=num_samples):
        if i == limit:
            break
        tensor[i] = sample
        labels[i] = label

    return tensor, labels

def random_samples_plot(tensor: torch.Tensor, labels: torch.Tensor = None):
    """
    Randomly sample a subset of the dataset.
    """
    # Randomly sample a subset of the dataset
    indices = torch.randperm(tensor.shape[0])[:128]

    # Each subplot should be at least 192x192 pixels
    fig, axs = plt.subplots(8, 16, figsize=(20, 12))
    # Plot the samples
    for i in tqdm(range(128)):
        ax = axs[i // 16, i % 16]
        # Get the sample
        sample = tensor[indices[i]]
        # Get the label
        if labels is not None:
            label = labels[indices[i]]
        ax.imshow(sample.squeeze().numpy(), cmap='gray')
        ax.set_title(EMOTION[label])
        ax.axis('off')
