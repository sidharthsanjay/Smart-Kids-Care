# Import necessary libraries
import numpy as np
import argparse
import imutils
import sys
import cv2
# Set up argument parser to pass arguments to the script
argv = argparse.ArgumentParser()
argv.add_argument("-m", "--model", required=True, help="path to pre-trained model")
argv.add_argument("-c", "--classes", required=True, help="path to class labels file")
argv.add_argument("-i", "--input", type=str, default="", help="path to video file")
argv.add_argument("-o", "--output", type=str, default="", help="path to output video file")
argv.add_argument("-d", "--display", type=int, default=1, help="whether to display output frame")
argv.add_argument("-g", "--gpu", type=int, default=0, he1p="whether to use GPU")
args = vars(argv.parse_args())

# Load class labels and set sample duration and size
ACT = open(args["classes"]).read().strip().split("\n")
SAMPLE_DURATION = 16
SAMPLE_SIZE = 112

# Load the neural network model
print("Loading the model for Human Activity Recognition")
gp = cv2.dnn.readNet(args["model"])
                          
# Set up GPU usage if specified
if args["gpu"] > 0:
    print("Setting backend and target to CUDA...")
    gp.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    gp.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Access the video stream
print("Accessing the video stream...")
vs = cv2.VideoCapture(args["input"] if args["input"] else 0)
writer = None
fps = vs.get(cv2.CAP_PROP_FPS)
print("Original FPS:", fps)

# Process video frames until the stream ends
while True:
    frames = [] # List to store processed frames
    originals = [] # List to store original frames

    # Capture sample frames for processing
    for i in range(0, SAMPLE_DURATION):
        (grabbed, frame) = vs.read()
        if not grabbed:
            print("No frame read from the stream Exiting. .")
            sys.exit(0)
        originals.append(frame)
        frame = imutils.resize(frame, width=400)
        frames.append(frame)

    # Createa blob from the captured frames
    blob = cv2.dnn.blobFromImages(frames, 1.0, (SAMPLE_SIZE, SAMPLE_SIZE), (114.7748, 107.7354, 99.4750), swapRB=True, crop=True)
    blob = np.transpose(blob, (1, 0, 2, 3))
    blob = np.expand_dims(blob, axis=0)

    # Perform activity prediction using the blob
    gp.setInput(blob)
    outputs = gp.forward()
    label = ACT[np.argmax(outputs)]

    # Add labels to the original frames
    for frame in originals:
        cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), -1)
        cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Display the frame if specified
        if args["display"] > 0:
            cv2.imshow("Activity Recognition", frame)
            key = cv2.waitKey(1)& 0xFF
            if key == ord("q"):
                break

        # Initialize video writer if output path is provided
        if args["output"] != "" and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args["output"], fourcc, fps, (frame.shape[1], frame.shape[0]), True)
                                     
        # Write the frame to the output video file
        if writer is not None:
            writer.write(frame)