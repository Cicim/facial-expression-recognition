import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from PIL import Image, ImageGrab

from fer_nets import neural_nets
from helpers import print_error, EMOTIONS, TimeIt

argparser = argparse.ArgumentParser(description='Use a trained CNN model to recognize human emotions in photos')
argparser.add_argument('--network', '-n', type=str, default='cnn1', help='Network to use')
argparser.add_argument('--image', '-i', type=str, required=False, help='Image to use')
argparser.add_argument('--clipboard', '-c', action='store_true', help='Use the clipboard as input')
argparser.add_argument('--model', '-m', type=str, required=True, help='Model to use')
args = argparser.parse_args()

# Get the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the network class
network_class = neural_nets.get(args.network)
if network_class is None:
    print_error(f"Network type `{args.network}` not found!")
    exit(1)

# Load the model
network = network_class.load(args.model)
network = network.to(device)

# Load the image
if args.image is not None:
    try:
        image = Image.open(args.image)
    except FileNotFoundError:
        print_error(f"Could not find image `{args.image}`")
        exit(1)
elif args.clipboard:
    image = ImageGrab.grabclipboard()
    if image is None:
        print_error("Could not get image from clipboard")
        exit(1)
    else:
        image = image.convert('RGB')
else:
    print_error("No image specified!")
    exit(1)


# Create a plot with the image
plt.imshow(image)

# Ask the network to predict the emotions of the faces in the image
with TimeIt("Performed Facial Expression Recognitions"):
    predictions, boxes = network.predict_from_image(image)
    if predictions is None:
        print_error("No faces found!")
        exit(1)
    
    # Draw the bounding boxes
    for i, (prediction, box) in enumerate(zip(predictions, boxes)):
        x1, y1, x2, y2 = box
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r')

        # Get the top three predictions
        top_predictions = prediction.topk(3).indices
        # Convert them to a string
        top_predictions = ' '.join([EMOTIONS[i][0:3] for i in top_predictions])

        # Add the prediction to the plot
        plt.text(x1, y1, top_predictions, color='yellow', fontsize=12)


plt.show()