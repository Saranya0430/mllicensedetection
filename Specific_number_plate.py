import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
import pytesseract
from datetime import datetime
import torch
import winsound
import math

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

model = YOLO('best.pt')

# Target number plates to monitor
target_plates = ["DL2CAT4762", "HR26CO6869"]  # Example number plates

# Conversion factor for pixel distance to real-world distance (e.g., meters per pixel)
# This value is camera-specific and should be calibrated for accuracy.
conversion_factor = 0.05  # Example: 0.05 meters per pixel

# Previous positions and time for speed calculation
prev_positions = {}

# Time variable to calculate elapsed time between frames
prev_time = datetime.now()

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('mycarplate.mp4')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")

area = [(27, 417), (16, 456), (1015, 451), (992, 417)]

count = 0
list1 = []
processed_numbers = set()

# Open file for writing car plate data
with open("car_plate_data.txt", "a") as file:
    file.write("NumberPlate\tDate\tTime\tSpeed (km/h)\n")  # Adding Speed column header

while True:    
    ret, frame = cap.read()
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
       break
   
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
   
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        
        d = int(row[5])
        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        result = cv2.pointPolygonTest(np.array(area, np.int32), ((cx, cy)), False)
        if result >= 0:
           crop = frame[y1:y2, x1:x2]
           gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
           gray = cv2.bilateralFilter(gray, 10, 20, 20)

           text = pytesseract.image_to_string(gray).strip()
           text = text.replace('(', '').replace(')', '').replace(',', '').replace(']','')
           
           if text:
               current_time = datetime.now()

               # Speed Detection Logic
               if text in prev_positions:
                   prev_x, prev_y, prev_time = prev_positions[text]
                   time_elapsed = (current_time - prev_time).total_seconds()
                   if time_elapsed > 0:
                       distance_traveled = math.sqrt((cx - prev_x) ** 2 + (cy - prev_y) ** 2) * conversion_factor
                       speed = (distance_traveled / time_elapsed) * 3.6  # Convert from m/s to km/h

                       # Print and log speed information
                       print(f"Vehicle {text} Speed: {speed:.2f} km/h")
                       with open("car_plate_data.txt", "a") as file:
                           file.write(f"{text}\t{current_time.strftime('%Y-%m-%d')}\t{current_time.strftime('%H:%M:%S')}\t{speed:.2f}\n")
                       
                       # If speed exceeds a threshold, trigger alert
                       if speed > 60:  # Example speed threshold: 60 km/h
                           print(f"ALERT! {text} is overspeeding at {speed:.2f} km/h!")
                           winsound.PlaySound("alert.wav", winsound.SND_FILENAME)
               else:
                   # Store initial position and time for new vehicle
                   prev_positions[text] = (cx, cy, current_time)

               # Log the number plate even if speed is not calculated
               if text not in processed_numbers:
                  processed_numbers.add(text) 
                  list1.append(text)
                  current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                  # Check if the detected number plate is in the target list
                  if text in target_plates:
                      print(f"ALERT! Target number plate {text} detected!")
                      winsound.PlaySound("alert.wav", winsound.SND_FILENAME)
                  
                  with open("car_plate_data.txt", "a") as file:
                      file.write(f"{text}\t{current_datetime.split()[0]}\t{current_datetime.split()[1]}\tN/A\n")
                  
                  cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                  cv2.imshow('crop', crop)

    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()    
cv2.destroyAllWindows()

