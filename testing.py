import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import pytesseract
from datetime import datetime
import torch
import re
import csv
import os

# Configure pytesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Load the trained YOLO model
model = YOLO('best.pt')

# Define a callback function to show RGB values on mouse movement
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(f"Mouse at {point}")

# Set up a named window and mouse callback
cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Open video file or webcam
cap = cv2.VideoCapture('mycarplate.mp4')
# cap = cv2.VideoCapture(0)  # Uncomment this to use webcam

# Load class labels for the model
my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Define area of interest for vehicle detection
area = [(27, 417), (16, 456), (1015, 451), (992, 417)]

# Initialize variables
count = 0
processed_numbers = set()

# Create directory for storing cropped license plates
if not os.path.exists('plates'):
    os.makedirs('plates')

# Open files for writing car plate data
with open("car_plate_data.txt", "a") as file_txt, open("car_plate_data.csv", "a", newline='') as file_csv:
    # Writing column headers if files are empty
    if os.path.getsize("car_plate_data.txt") == 0:
        file_txt.write("NumberPlate\tDate\tTime\n")
    if os.path.getsize("car_plate_data.csv") == 0:
        csv_writer = csv.writer(file_csv)
        csv_writer.writerow(["NumberPlate", "Date", "Time"])

# Start video processing loop
while True:
    ret, frame = cap.read()
    count += 1
    if count % 5 != 0:  # Process every 5th frame to reduce computational load
        continue
    if not ret:
       break

    # Resize frame for consistent processing
    frame = cv2.resize(frame, (1020, 500))

    # Perform vehicle detection using YOLO model
    results = model.predict(frame)

    # Extract bounding box data
    bboxes = results[0].boxes.data
    if bboxes is not None:
        px = pd.DataFrame(bboxes).astype("float")

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            confidence = row[4]
            class_id = int(row[5])
            class_name = class_list[class_id] if class_id < len(class_list) else "Unknown"

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # Crop the detected vehicle region and process it with pytesseract
            crop = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 10, 20, 20)

            # Extract text from the detected license plate
            text = pytesseract.image_to_string(gray).strip()
            text = text.replace('(', '').replace(')', '').replace(',', '').replace(']', '')

            # Define a pattern for valid license plates (adjust pattern based on your use case)
            license_plate_pattern = r'^[A-Z0-9]{1,7}$'

            # Check if detected text matches the pattern and is not already processed
            if re.match(license_plate_pattern, text) and text not in processed_numbers:
                processed_numbers.add(text)
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                # Write the detected license plate data to the text and CSV files
                with open("car_plate_data.txt", "a") as file_txt, open("car_plate_data.csv", "a", newline='') as file_csv:
                    file_txt.write(f"{text}\t{current_datetime}\n")
                    csv_writer = csv.writer(file_csv)
                    csv_writer.writerow([text, current_datetime])

                # Save the cropped license plate image
                license_plate_filename = f"plates/{text}_{current_datetime}.jpg"
                cv2.imwrite(license_plate_filename, crop)

                # Draw bounding box with label and confidence score
                cvzone.cornerRect(frame, (x1, y1, x2-x1, y2-y1), l=10, rt=2, colorR=(0, 255, 0))
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow('License Plate', crop)

    # Draw the area of interest polygon on the frame
    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)

    # Add timestamp overlay to the frame
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the frame with detections
    cv2.imshow("RGB", frame)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
