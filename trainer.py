#!/usr/bin/env python3


# Get parameters through arguments
import argparse
import json

import matplotlib.pyplot as plt

from data_loader import load_dataset
from fer_nets import neural_nets
from helpers import print_error
from neural_net import train, test


DEFAULT_TRAINING_DATA_PATH = "datasets/complete.samples"
DEFAULT_VALIDATION_DATA_PATH = "datasets/fer2013plus_valid.samples"

def main():
    argparser = argparse.ArgumentParser(description="Train a neural network")

    # Add the train or test commands
    argparser.add_argument("command", choices=["train", "test"], help="Train or test the network")

    # Arguments
    argparser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs to train")
    argparser.add_argument("--batch-size", "-b", type=int, default=256, help="Batch size")
    argparser.add_argument("--learning-rate", "-l", type=float, default=0.001, help="Learning rate")
    argparser.add_argument("--model-save-path", "-m", type=str, help="Path to save the model")
    argparser.add_argument("--validation-data", "-v", type=str, default=DEFAULT_VALIDATION_DATA_PATH, help="Path to the validation data")
    argparser.add_argument("--training-data", "-t", type=str, default=DEFAULT_TRAINING_DATA_PATH, help="Path to the training data")
    argparser.add_argument("--load-model", "-lm", type=str, default=None, help="Path to a model to load")
    argparser.add_argument("--network", "-n", type=str, default="cnn1", help="Network to use")
    argparser.add_argument("--show-plots", "-s", action="store_true", help="Show plots")
    argparser.add_argument("--limit-training-samples", "-lt", type=int, default=None, help="Limit the number of training samples")
    argparser.add_argument("--limit-validation-samples", "-lv", type=int, default=None, help="Limit the number of validation samples")
    args = argparser.parse_args()

    # Get the network class
    network_class = neural_nets.get(args.network)
    if network_class is None:
        print_error(f"Network type `{args.network}` not found!")
        exit(1)

    # Load or create the model
    if args.load_model is not None:
        try:
            network = network_class.load(args.load_model)
        except Exception as e:
            print_error(f"Could not load `{args.network}` model from `{args.load_model}`")
            print_error(*e.args)
            exit(1)
    else:
        network = network_class()

    print("Loading validation data...")
    validation_data = load_dataset(args.validation_data, limit=args.limit_validation_samples)

    # Train the network
    if args.command == "train":
        # Load the training and validation data
        print("Loading training data...")
        training_data = load_dataset(args.training_data, limit=args.limit_training_samples)

        # Train the network
        stats = train(network, training_data, validation_data, args.model_save_path, 
                    epochs=args.epochs, batch_size=args.batch_size, 
                    learning_rate=args.learning_rate, print_to_screen=True)

        # Create a graph with the loss stats
        epochs = [stat.epoch for stat in stats]
        train_loss = [stat.training_loss for stat in stats]
        val_loss = [stat.validation_loss for stat in stats]
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epochs, train_loss, label="Training loss")
        plt.plot(epochs, val_loss, label="Validation loss")
        plt.legend()
        plt.savefig("models/loss_stats.png")
        if args.show_plots: plt.show()

        plt.clf()
        # Create a graph with the accuracy stats
        train_acc = [stat.training_accuracy for stat in stats]
        val_acc = [stat.validation_accuracy for stat in stats]
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(epochs, train_acc, label="Training accuracy")
        plt.plot(epochs, val_acc, label="Validation accuracy")
        plt.legend()
        plt.savefig("models/accuracy_stats.png")
        if args.show_plots: plt.show()

        # Save the stats to a json file
        with open("models/stats.json", "w") as f:
            f.write(json.dumps([stat.__dict__ for stat in stats]))

    try:
        # In any case, test the network (print the confusion matrix)
        test(network, validation_data, batch_size=64)
    except KeyboardInterrupt:
        print("Testing stopped. Exiting...")
        exit(0)
    


if __name__ == "__main__":
    main()
