import cv2  # Importing OpenCV library for image and video processing
import time  # Importing time module for time-related tasks
import winsound  # Importing winsound module for playing system sounds
from datetime import datetime  # Importing datetime class from datetime module
from ultralytics import YOLO  # Importing YOLO class from ultralytics module for object detection
from config import *  # Importing all variables from config module
from utils import rects_overlap, calc_distance, ensure_folder_exists  # Importing utility functions
from database import init_db, insert_alert, update_notification_status  # Importing database functions
from alert import send_twilio_alert, can_send_alert  # Importing alert functions

# Define the variables
MODEL_NAME = 'yolov3'  # Example model name
VIDEO_PATH = 'path/to/video.mp4'  # Path to the video file
ALERT_LOG_FILE = 'alert_log.txt'  # Path to the alert log file
DETECTION_INTERVAL = 30  # Example detection interval
ALERT_SNAPSHOT_FOLDER = 'snapshots'  # Folder for saving snapshots
DISTANCE_THRESHOLD = 50  # Distance threshold for high severity

# Main processing function with MIL tracking
def process_video_with_tracking():
    ensure_folder_exists(ALERT_SNAPSHOT_FOLDER)  # Ensure snapshot folder exists
    conn, cursor = init_db()  # Initialize the database
    last_alert_time = 0  # Track the last alert time for cooldown

    # Load the YOLO model
    model = YOLO(MODEL_NAME)
    cap = cv2.VideoCapture(VIDEO_PATH)

    trackers = []
    tracked_objects = []
    frame_count = 0  # Frame counter

    with open(ALERT_LOG_FILE, "a") as log_file:  # Open in append mode
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            frame_count += 1

            if frame_count % DETECTION_INTERVAL == 1:
                # Detect objects
                results = model(frame)[0]
                trackers = []
                tracked_objects = []

                # Initialize trackers for detected objects
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = box.cls
                    label = results.names[int(cls)]
                    if label in ["person", "knife", "scissors", "cell_phone"]:
                        tracker = cv2.TrackerMIL_create()
                        tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
                        trackers.append(tracker)
                        tracked_objects.append((x1, y1, x2, y2, label))
            else:
                # Use tracking to update object positions
                updated_tracked_objects = []
                for tracker in trackers:
                    success, bbox = tracker.update(frame)
                    if success:
                        x, y, w, h = map(int, bbox)
                        updated_tracked_objects.append((x, y, x + w, y + h))

                # Ensure the object labels are carried over
                for i in range(len(tracked_objects)):
                    updated_tracked_objects[i] = (
                        *updated_tracked_objects[i], tracked_objects[i][4]
                    )
                tracked_objects = updated_tracked_objects

            # Detect overlapping children and dangerous objects
            children_boxes = [obj[:4] for obj in tracked_objects if obj[4] == "person"]
            dangerous_objects_boxes = [obj[:4] for obj in tracked_objects if obj[4] in ["knife", "scissors", "cell_phone"]]
            for child_box in children_boxes:
                for dangerous_object_box in dangerous_objects_boxes:
                    if rects_overlap(child_box, dangerous_object_box):

                        # Log the alert
                        alert_message = f"Frame {frame_count}: Alert! Child is handling a dangerous object (overlap detected).\n"
                        print(alert_message)
                        log_file.write(alert_message)
                        log_file.flush()

                        # Calculate distance
                        distance = calc_distance(child_box, dangerous_object_box)
                        screenshot_filename = None
                        # Determine severity
                        severity = "High" if distance < DISTANCE_THRESHOLD else "Low"

                        # Save snapshot
                        if severity == "High":
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            screenshot_filename = f"{ALERT_SNAPSHOT_FOLDER}/frame_{frame_count}_{timestamp}.png"
                            cv2.imwrite(screenshot_filename, frame)
