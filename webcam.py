import cv2
import torch
from facenet_pytorch import MTCNN
from helpers import TimeIt, EMOTIONS
from fer_nets import CNNFER3

# Get the GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Initialize the MTCNN network
mtcnn = MTCNN(
    thresholds=[0.6, 0.7, 0.7],
    min_face_size=48,
    device=device,
)

# Load the model
model = CNNFER3.load('models/cnn3_new_071.pt')
model = model.to(device)


# Capture the video from the webcam
cap: 'cv2.VideoCapture' = cv2.VideoCapture()
cap.open(0, cv2.CAP_DSHOW)


while(True):
    ret, frame = cap.read()

    with TimeIt("Detected faces"):
        # Run the MTCNN network on the frame
        boxes, probabilities = mtcnn.detect(frame)

    if boxes is not None:
        for box, probability in zip(boxes, probabilities):
            box = list(map(int, list(box)))
            x1, y1, x2, y2 = box
            x1old, y1old, x2old, y2old = box

            w = x2 - x1
            h = y2 - y1
            if w > h:
                y1 -= (w - h) // 2
                y2 += (w - h) // 2
            elif h > w:
                x1 -= (h - w) // 2
                x2 += (h - w) // 2

            with TimeIt("Predicted class"):
                # Crop the image
                cropped = frame[y1:y2, x1:x2]
                # Convert to grayscale
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                # Resize it to 48x48
                cropped = cv2.resize(cropped, (48, 48), interpolation=cv2.INTER_AREA)
                # Correct the exposure
                cropped = cv2.equalizeHist(cropped)

                # Save the image
                cv2.imwrite('test.jpg', cropped)

                # Convert to tensor
                cropped = torch.from_numpy(cropped).float().unsqueeze(0).unsqueeze(0)
                # Move to GPU
                cropped = cropped.to(device)

                # Predict the class
                predicted = model(cropped)
                # Rank the top three predictions
                predicted = predicted.cpu().detach().numpy()
                predicted = predicted.squeeze()
                predicted = predicted.argsort()[-3:][::-1]
                # Create a string with all three
                predicted = ' '.join(EMOTIONS[i][0:3].upper() for i in predicted)

                # Get the index of the max value
                # predicted = predicted.argmax()
                # Get the emotion
                # emotion = EMOTIONS[predicted]
                # Draw the emotion on top
                cv2.putText(frame, predicted, (0, 24), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            cv2.rectangle(frame, (x1old, y1old), (x2old, y2old), (0, 255, 0), 2)
            # Draw the probability on top
            cv2.putText(frame, f'{probability:.2f}', (x1old, y1old - 2), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            # Draw the squarer rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)


    cv2.imshow('Webcam Feed', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
