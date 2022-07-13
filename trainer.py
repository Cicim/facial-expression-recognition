import os
import glob
import datetime
from typing import Any

import torch

from data_loader import load_samples
from neural_net import CNNFER1, prepare_training_data


## Utilities
RED_TEXT = "\033[31m"
RESET = "\033[0m"


def try_int(x: str, default: int = 0):
    try:
        return int(x)
    except:
        return default

def choose_option(options: dict[Any, str], allow_invalid: bool = False):
    """
    Prompts the user to choose an option from a list of options.
    """
    option_keys = []
    for i, key in enumerate(options):
        print(f" {i+1:3}) {options[key]}")
        option_keys.append(key)

    choice = input(">>> ")
    try:
        choice_index = int(choice) - 1
    except:
        choice_index = -1

    if choice_index in range(len(option_keys)):
        return option_keys[choice_index]
    elif allow_invalid:
        return choice
    else:
        print(RED_TEXT + "Invalid choice." + RESET)
        return choose_option(options)

def list_dir(dir: str, glob_str: str = "*"):
    """
    Lists the contents of a directory sorted by last modified.
    """
    listed = sorted(os.listdir(dir), key=lambda x: -os.path.getmtime(f"{dir}/{x}"))
    # Filter out files that don't match the glob
    if glob_str:
        listed = [x for x in listed if glob.fnmatch.fnmatch(x, glob_str)]
    return listed

def get_samples_file(default: str = "fer2013_train.samples"):
    """
    Prompts the user to choose a samples file.
    """
    options = { f"{x}": x for x in list_dir("datasets", "*.samples") }
    options[default] = f"{default} (default)"

    print("Enter the path to the samples:")
    return 'datasets/' + (choose_option(options, allow_invalid=True) or default)

def try_get_model_time(model: str):
    try:
        last_modified_time = os.path.getmtime(f'models/{model}')
        # Make it human readable
        return str(datetime.datetime.fromtimestamp(last_modified_time))
    except:
        return "???"

def get_model_path(save: bool, default: str = "model.pt"):
    """
    Prompts the user to choose a save destination.
    """
    options = { f"{x}": f"[{try_get_model_time(x)}] {x}" for x in list_dir("models", "*.pt") }
    options[default] = f"[{try_get_model_time(default)}] {default} (default)"

    if save:
        print("Enter the path to the save destination (or select one of the following):")
    else:
        print("Enter the path to the model (or select one of the following):")

    return 'models/' + (choose_option(options, allow_invalid=True) or default)

def train_sequence(network):
    # Ask for the training data
    training_data_file = get_samples_file()

    # Choose the save destination
    save_dest = get_model_path(save=True)

    # Ask for the batch size
    batch_size = try_int(input("Enter the batch size (default 1000): "), 1000)
    # Ask for the epochs
    epochs = try_int(input("Enter the number of epochs (default 10): "), 10)
    # Ask for the learning rate
    learning_rate = try_int(input("Enter the learning rate (default 0.001): "), 0.001)


    # Load the training data
    training_data = prepare_training_data(training_data_file, batch_size)
            
    # Train the network
    network.train(training_data, save_dest, epochs, learning_rate)
    print("Training complete.")

def test_sequence(network, default: str = "fer2013_valid.samples"):
    # Ask for the validation set
    test_data_file = get_samples_file(default)
    # Load the validation data
    test_data = load_samples(test_data_file)

    # Validate the network
    network.test(test_data)
    print("Testing complete.")




def main():
    print("Choose an option:")
    chosen = choose_option({
        "new_train": "Train a new network",
        "load_train": "Load a network and train it",
        "load_test": "Load a network and test it",
        "exit": "Exit"
    })

    if chosen == "new_train":
        # Create an empty network
        network = CNNFER1()
        train_sequence(network)

        print("Do you want to validate the network? (y/n)")
        if input(">>> ") == "y":
            test_sequence(network)

    elif chosen == "load_train":
        # Load an existing network
        network = CNNFER1()
        network.load_state_dict(torch.load(get_model_path(save=False)), strict=False)

        train_sequence(network)
        print("Do you want to validate the network? (y/n)")
        if input(">>> ") == "y":
            test_sequence(network)

    elif chosen == "load_test":
        # Load a network
        network = CNNFER1()
        network.load_state_dict(torch.load(get_model_path(save=False)), strict=False)

        # Test the network
        test_sequence(network)


    elif chosen == "exit":
        return



if __name__ == '__main__':
    main()