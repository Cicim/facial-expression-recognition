from helpers import *

import argparse
import os
import json

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from facenet_pytorch import MTCNN
import cv2

# Accept inputs through the command line
parser = argparse.ArgumentParser()
parser.add_argument('--samples_file', type=str, default='datasets/CalTech_WebFaces.samples',
                    help='Path to the file to save the samples.')
parser.add_argument('--faces_images', type=str, default='datasets/faces',
                    help='Path to the output faces images.')
parser.add_argument('--correct_labels', type=str, required=True,
                    help='Path to the JSON file containing the correct labels.')
args = parser.parse_args()

# If the samples_file already exists, delete it
if os.path.exists(args.samples_file):
    os.remove(args.samples_file)
# If the images_path doesn't exist, exit
if not os.path.exists(args.faces_images):
    print(f'The directory {args.faces_images} does not exist.')
    exit(1)
# If the correct_labels file doesn't exist, exit
if not os.path.exists(args.correct_labels):
    print(f'The file {args.correct_labels} does not exist.')
    exit(1)

# -------------------------------------
# 5. Read the images from the directory
# -------------------------------------
faces = []
with TimeIt('Read images from the directory'):
    # 5.1 Get the files in the path
    image_files = os.listdir(args.faces_images)
    # 5.2 Sort the files by name
    image_files.sort()
    # 5.3 Read the images
    print("Reading images...")
    for image_file in tqdm(image_files):
        face = Image.open(os.path.join(args.faces_images, image_file))
        # Convert the face to an array
        face = np.array(face)
        # Add the face to the list
        faces.append(face)

# -------------------------------------
# 6. Read the correct labels from the file
# -------------------------------------
correct_labels = []
with TimeIt('Read the correct labels from the file'):
    # 6.1 Read the JSON file
    with open(args.correct_labels, 'r') as f:
        json_file: dict = json.load(f)

    # 6.2 Convert the labels to a list of integers
    for _, i in zip(range(len(faces)), json_file.keys()):
        correct_labels.append(json_file[i])

    # 6.3 Count the non-null labels
    non_null_labels = 0
    for label in correct_labels:
        if label is not None:
            non_null_labels += 1

print(f"Predicted emotions distribution over {non_null_labels} faces:")
for i, emotion in enumerate(EMOTIONS):
    c = correct_labels.count(i)
    print(f"{emotion:11}: {c:10} ({c/non_null_labels*100:5.2f}%)")


# -------------------------------------
# 7. Create the samples file
# -------------------------------------
with TimeIt('Created the samples file'):
    # 7.1 Open the file for writing
    with open(args.samples_file, 'wb') as f:
        # 7.2 Write the number of samples
        f.write(non_null_labels.to_bytes(4, 'little'))

        # 7.3 Write the samples
        for i, label in tqdm(enumerate(correct_labels), total=non_null_labels):
            if label is None:
                continue
        
            # Convert the original grayscale image to a numpy array
            array = np.array(faces[i])
            # Convert the array to bytes
            bytes = array.tobytes()
            # Write the bytes to the file
            f.write(bytes)

            # Write the label as a little-endian byte
            f.write(label.to_bytes(1, 'big'))

print("Finished!")