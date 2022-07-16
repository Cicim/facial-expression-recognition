import argparse

import torch
import numpy as np
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
from PIL import Image, ImageGrab

from fer_nets import neural_nets
from helpers import print_error, EMOTIONS

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


# Create the MTCNN detector
mtcnn = MTCNN(
    thresholds=[0.6, 0.7, 0.7],
    min_face_size=48,
    device=device,
)

# Detect faces
boxes, probabilities = mtcnn.detect(image)
if boxes is None:
    print_error("No faces detected!")
    exit(0)

# Create a plot with the image
plt.imshow(image)

# Get the faces
for box, probability in zip(boxes, probabilities):
    box = list(map(int, list(box)))
    x1, y1, x2, y2 = box
    x1old, y1old, x2old, y2old = box

    # Plot the old rectangle in the plot in blue
    plt.plot([x1old, x2old, x2old, x1old, x1old], [y1old, y1old, y2old, y2old, y1old], 'b-')

    w = x2 - x1
    h = y2 - y1
    if w > h:
        y1 -= (w - h) // 2
        y2 += (w - h) // 2
    elif h > w:
        x1 -= (h - w) // 2
        x2 += (h - w) // 2

    # Draw the rectangle in the plot in red
    plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-')

    # Resize the image to 48x48
    cropped = image.crop((x1, y1, x2, y2))
    cropped = cropped.resize((48, 48), Image.BICUBIC)
    cropped = cropped.convert('L')

    # Predict the emotion
    with torch.no_grad():
        # Convert the output to a numpy array
        cropped = np.array(cropped)
        # Convert it to a tensor
        cropped = torch.from_numpy(cropped).float().unsqueeze(0).unsqueeze(0)
        # Move to GPU
        cropped = cropped.to(device)

        # Predict the class
        output = network(cropped)
        output = output.cpu().numpy()
        # Get the three indices of the largest values
        indices = np.argsort(output)[0][4:]

    third, second, first = map(lambda x: EMOTIONS[x], indices)

    # Print the emotion
    plt.text(x1, y1, f"{first}\n{second}\n{third}", color='r')

plt.show()