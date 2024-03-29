#!/usr/bin/env python3


# Get parameters through arguments
import argparse
import json
import os

import matplotlib.pyplot as plt

from data_loader import load_dataset, load_multiple_datasets
from fer_nets import neural_nets
from helpers import print_error
from neural_net import EpochStats, train, test, ModelSaveStrategy

def transform_dataset_path(path: str):
    # If the path is just a file name, with no / or .
    if "/" not in path and "." not in path:
        return f"datasets/{path}.samples"
    return path

def get_datasets(paths: str, limit=None):
    # Split the paths by comma
    paths = paths.split(",")
    # Transform the paths to absolute paths
    paths = [transform_dataset_path(path) for path in paths]
    # If there is only one
    if len(paths) == 1:
        return load_dataset(paths[0], limit)

    if limit is not None:
        print_error("Cannot apply limit to multiple datasets!")
    # If there are more than one, merge them
    return load_multiple_datasets(paths)

def get_model_path(name: str):
    # If it is already a path
    if "/" in name or "." in name:
        return name

    # Check if the model name file exists
    if os.path.exists(f"models/{name}.pt"):
        return f"models/{name}.pt"

    try:
        # Else, look for all the models starting with
        # the name in the models folder
        candidates = [f for f in os.listdir(f"models") if f.startswith(name) and f.endswith('.pt')]

        max_number = 0
        best_candidate = None
        # Sort them by number in their name
        for candidate in candidates:
            # Get only the digits in the name
            filename = candidate.split("_")[-1]
            filename = filename.split(".")[0]
            number = int(''.join(c for c in filename if c.isdigit()))
            if number > max_number:
                max_number = number
                best_candidate = candidate

        return 'models/' + best_candidate
    except:
        print_error("Could not find model with the name only. Try using the full path")
        exit(1)

def save_graphs_and_stats(stats: EpochStats, name: str, show_plots: bool = False):
    # Create a graph with the loss stats
    epochs = [stat.epoch for stat in stats]
    train_loss = [stat.training_loss for stat in stats]
    val_loss = [stat.validation_loss for stat in stats]
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, train_loss, label="Training loss")
    plt.plot(epochs, val_loss, label="Validation loss")
    plt.legend()
    plt.savefig(f"models/{name}_loss_graph.png")
    if show_plots: plt.show()

    plt.clf()
    # Create a graph with the accuracy stats
    train_acc = [stat.training_accuracy for stat in stats]
    val_acc = [stat.validation_accuracy for stat in stats]
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(epochs, train_acc, label="Training accuracy")
    plt.plot(epochs, val_acc, label="Validation accuracy")
    plt.legend()
    plt.savefig(f"models/{name}_accuracy_graph.png")
    if show_plots: plt.show()

    # Save the stats to a json file
    with open(f"models/{name}_stats.json", "w") as f:
        f.write(json.dumps([stat.__dict__ for stat in stats]))



DEFAULT_TRAINING_DATA_PATH = "datasets/complete.samples"
DEFAULT_VALIDATION_DATA_PATH = "datasets/fer2013plus_valid.samples"

def main():
    argparser = argparse.ArgumentParser(description="Train a neural network")

    # Add the train or test commands
    argparser.add_argument("command", choices=["train", "test"], help="Train or test the network")

    # Arguments
    argparser.add_argument("--epochs", "-e", type=int, default=10, 
        help="Number of epochs to train")
    argparser.add_argument("--batch-size", "-b", type=int, default=256, 
        help="Batch size")
    argparser.add_argument("--learning-rate", "-l", type=float, default=0.001, 
        help="Learning rate")
    argparser.add_argument("--model-save-name", "-m", type=str, 
        help="Name of the saved model")
    argparser.add_argument("--validation-data", "-v", type=str, default=DEFAULT_VALIDATION_DATA_PATH, 
        help="Path to the validation data")
    argparser.add_argument("--training-data", "-t", type=str, default=DEFAULT_TRAINING_DATA_PATH, 
        help="Path to the training data")
    argparser.add_argument("--load-model", "-lm", type=str, default=None, 
        help="Path to a model to load")
    argparser.add_argument("--network", "-n", type=str, default="cnn1", choices=neural_nets.keys(), 
        help="Network to use")
    argparser.add_argument("--show-plots", "-p", action="store_true", 
        help="Show plots")
    argparser.add_argument("--limit-training-samples", "-lt", type=int, default=None, 
        help="Limit the number of training samples")
    argparser.add_argument("--limit-validation-samples", "-lv", type=int, default=None, 
        help="Limit the number of validation samples")
    argparser.add_argument("--save-strategy", "-s", type=str, default="BEST", choices=ModelSaveStrategy.__members__.keys(),
        help="What to do when saving the model" )
    argparser.add_argument("--flip-rate", "-f", type=float, default=None,
        help="How often should the samples in a batch be flipped")
    args = argparser.parse_args()

    # Get the network class
    network_class = neural_nets.get(args.network)
    if network_class is None:
        print_error(f"Network type `{args.network}` not found!")
        exit(1)

    # Expect a model save path if we are training
    if args.command == "train" and args.model_save_name is None:
        print_error("You must specify a model save path if you are training!")
        print_error("   Do it with the -m flag")
        exit(1)
    # Expect a model load path if we are testing
    elif args.command == "test" and args.load_model is None:
        print_error("You must specify a model load path if you are testing!")
        print_error("   Do it with the -lm flag")
        exit(1)


    if args.load_model is not None:
        args.load_model = get_model_path(args.load_model)
        print(f"Loading model from `{args.load_model}`")
        
        # Load the model
        try:
            network = network_class.load(args.load_model)
        except Exception as e:
            print_error(f"Could not load `{args.network}` model from `{args.load_model}`")
            print_error(*e.args)
            exit(1)
    # Else, create it
    else:
        network = network_class()
        

    print("Loading validation data...")
    validation_data = get_datasets(args.validation_data, limit=args.limit_validation_samples)

    # Train the network
    if args.command == "train":
        # Convert the save strategy to an enum
        args.save_strategy = ModelSaveStrategy[args.save_strategy]

        # Load the training and validation data
        print("Loading training data...")
        training_data = get_datasets(args.training_data, limit=args.limit_training_samples)

        # Train the network
        stats = train(network, training_data, validation_data, args.model_save_name, 
                      epochs=args.epochs, batch_size=args.batch_size, 
                      learning_rate=args.learning_rate, print_to_screen=True,
                      save_strategy=args.save_strategy)

        # Save and show the results
        save_graphs_and_stats(stats, args.model_save_name, args.show_plots)

    try:
        # In any case, test the network (print the confusion matrix)
        test(network, validation_data, batch_size=64)
    except KeyboardInterrupt:
        print("Testing stopped. Exiting...")
        exit(0)
    


if __name__ == "__main__":
    main()
