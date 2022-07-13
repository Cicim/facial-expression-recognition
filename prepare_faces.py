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
parser.add_argument('--images_path', type=str, default='datasets/CalTech_WebFaces',
                    help='Path to the directory containing the images.')
parser.add_argument('--samples_file', type=str, default='datasets/CalTech_WebFaces.samples',
                    help='Path to the file to save the samples.')
parser.add_argument('--output_path', type=str, default='datasets/faces',
                    help='Path to the output faces images.')
parser.add_argument('--model_weights', type=str, default='models/convo.pt',
                    help='Path to the FER model\'s weights.')
parser.add_argument('--predictions_file', type=str, default='datasets/labels.js',
                    help='Path to the JS file containing the network\'s predictions')
args = parser.parse_args()

# If the samples_file already exists, delete it
if os.path.exists(args.samples_file):
    os.remove(args.samples_file)
# If the images_path doesn't exist, exit
if not os.path.exists(args.images_path):
    print(f'The directory {args.images_path} does not exist.')
    exit(1)
# If the output_path doesn't exist, create it
if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)
# If the model_weights file doesn't exist, exit
if not os.path.exists(args.model_weights):
    print(f'The file {args.model_weights} does not exist.')
    exit(1)





# -------------------------------------
# 1. Read the images from the directory
# -------------------------------------
images = []
with TimeIt('Read images from the directory'):
    # 1.1 Get the files in the path
    image_files = os.listdir(args.images_path)
    # 1.2 Sort the files by name
    image_files.sort()
    # 1.3 Read the images
    print("Reading images...")
    for image_file in tqdm(image_files):
        # Read the image
        image = Image.open(os.path.join(args.images_path, image_file))
        # Convert the image to RGB
        image = image.convert('RGB')
        # Add the image to the list
        images.append(image)
print("Size in memory of the images:", sizeof_fmt(getsize(images)))

# --------------------------------------
# 2. Pass them through the face detector
# --------------------------------------
faces = []
with TimeIt('Face detection complete!'):
    # 2.1 Get the best device for training
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 2.2 Create the face detector
    face_detector = mtcnn = MTCNN(
        image_size=100, margin=0, min_face_size=48,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device
    )
    # 2.3 Detect the faces
    print("Detecting faces...")
    for img in tqdm(images):
        if img.size < (48, 48):
            continue

        boxes, precision = mtcnn.detect(img)
        if boxes is not None:
            for box, prec in zip(boxes, precision):
                if prec > 0.9:
                    x1, y1, x2, y2 = box
                    # Make sure the box is square
                    w = x2 - x1
                    h = y2 - y1
                    if w > h:
                        y1 -= (w - h) // 2
                        y2 += (w - h) // 2
                    elif h > w:
                        x1 -= (h - w) // 2
                        x2 += (h - w) // 2
                    
                    faces.append(img.crop((x1, y1, x2, y2)))
print(f"Detected {len(faces)} faces in {len(images)} images.")
print("Size in memory of the faces:", sizeof_fmt(getsize(faces)))

# -----------------------------------
# 3. Preparing images for the labeler
# -----------------------------------
good_faces = []
with TimeIt('Prepared images for the labeler'):
    faces_48x48 = []
    # 3.1 Resize the faces to 48x48 and convert them to grayscale
    print("Converting faces to 48x48 grayscale...")
    for face in tqdm(faces):
        # Convert the face to grayscale
        face = face.convert('L')
        # Resize the face to 48x48
        face = face.resize((48, 48))
        # Add the face to the list
        faces_48x48.append(face)
    
    # 3.2 Remove duplicate images
    print("Removing duplicate images...")
    faces_hashes = set()
    unique_faces = []
    for img in tqdm(faces_48x48):
        img_hash = hash(img.tobytes())
        if img_hash not in faces_hashes:
            unique_faces.append(img)
            faces_hashes.add(img_hash)
    print(f"Removed {len(faces_48x48) - len(unique_faces)} duplicate faces.")

    # 3.3 Remove blurred images
    print("Removing blurred images...")
    for img in tqdm(unique_faces):
        # Compute the variance of the Laplacian filter of img
        var = cv2.Laplacian(np.array(img), cv2.CV_64F).var()
        if var > 500:
            good_faces.append(img)

    # 3.4 Clear the output folder
    print("Clearing the output folder...")
    for file in tqdm(sorted(os.listdir(args.output_path))):
        os.remove(os.path.join(args.output_path, file))

    # 3.5 Save the images
    print("Saving the images...")
    for i, img in enumerate(tqdm(good_faces)):
        img.save(os.path.join(args.output_path, f'{i}.png'))

print(f"Got {len(good_faces)} usable faces out of the starting {len(faces)}.")
print("Size in memory of the good faces:", sizeof_fmt(getsize(good_faces)))


# -----------------------------------
# 4. Preclassify the faces
# -----------------------------------
predictions = []
with TimeIt('Faces classified'):
    # 4.1 Load the model
    from neural_net import CNNFER1
    model = CNNFER1()
    model.load_state_dict(torch.load('models/convo.pt'))
    model = model.to(device)

    # 4.2 Convert the faces to tensors
    faces_tensors = []
    print("Converting faces to tensors...")
    for face in tqdm(good_faces):
        array = np.array(face)
        tensor = torch.from_numpy(array).float().unsqueeze(0)
        # Divide by 255 to convert to a range of 0-1
        tensor = tensor / 255.0
        # Normalize the tensor
        tensor = (tensor - 0.5) / 0.255
        faces_tensors.append(tensor)

    # 4.3 Predict the faces
    print("Predicting faces' emotions...")
    for tensor in tqdm(faces_tensors):
        # Add the batch dimension
        tensor = tensor.unsqueeze(0)
        # Move the tensor to the GPU
        tensor = tensor.to(device)
        # Run the model
        output = model(tensor)

        # Get the index of the maximum value in the output
        _, index = torch.max(output, 1)
        # Add the result to the list
        predictions.append(index.item())

    # 4.4 Save the predictions to the predictions file
    with open(args.predictions_file, 'w') as f:
        f.write('export const data = ')
        f.write(json.dumps(predictions, separators=(',', ':')))
print(f"Predicted emotions distribution over {len(predictions)} faces:")
for i, emotion in enumerate(EMOTIONS):
    c = predictions.count(i)
    print(f"{emotion:11}: {c:10} ({c/len(predictions)*100:5.2f}%)")

