import argparse

import cv2
import torch
from helpers import TimeIt, EMOTIONS, print_error
from fer_nets import neural_nets
from PIL import Image

# Parse the arguments
argparser = argparse.ArgumentParser(description='Use a trained CNN model to recognize human emotions from your webcam')
argparser.add_argument('--network', '-n', type=str, default='cnn1', help='Network to use')
argparser.add_argument('--model', '-m', type=str, required=True, help='Model to use')
args = argparser.parse_args()


# Get the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Get the network class
network_class = neural_nets.get(args.network)
if network_class is None:
    print_error(f"Network type `{args.network}` not found!")
    exit(1)

# Load the model
model = network_class.load(args.model)
model = model.to(device)

# Capture the video from the webcam
cap = cv2.VideoCapture()
cap.open(0, cv2.CAP_DSHOW)

def render_emotions_graph(frame, predicted: torch.Tensor, box: tuple[int, int, int, int]):
    x1, y1, x2, _ = box

    # Set the negatives to 0
    predicted[predicted < 0] = 0
    # Normalize the values
    predsum = torch.sum(predicted)
    if predsum > 0:
        predicted = predicted / predsum

    bar_width = (x2 - x1) // len(EMOTIONS)
    bar_height = 40
    # Draw each emotion
    for i, emotion in enumerate(EMOTIONS):
        # Get the new pixel position
        x = int(x1 + i * bar_width)
        y = int(y1 - (predicted[i] * bar_height))
        # Draw a bar graph with the emotion
        cv2.rectangle(frame, (x, y), (x + bar_width, int(y1)), (0, 0, 255), cv2.FILLED)
        # Draw the emotion name
        cv2.putText(frame, emotion[0:3].upper(), (x, y + 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)



while(True):
    ret, frame = cap.read()

    # Use the network method for detecting faces and predicting emotions
    with TimeIt("Performed Facial Expression Recognitions"):
        # Convert the frame to a PIL image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)

        predictions, boxes = model.predict_from_image(image)
        if predictions is None:
            continue
        for i, (prediction, box) in enumerate(zip(predictions, boxes)):
            x1, y1, x2, y2 = box
            # Draw the squarer rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            render_emotions_graph(frame, prediction, box)

    cv2.imshow('Webcam Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
