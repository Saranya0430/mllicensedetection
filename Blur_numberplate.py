import cv2
import pandas as pd
from ultralytics import YOLO
import numpy as np
import pytesseract
from datetime import datetime
import torch

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

model = YOLO('best.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('mycarplate.mp4')

# Read class labels
with open("coco1.txt", "r") as my_file:
    data = my_file.read()
class_list = data.split("\n") 

# Define area for detection
area = [(27, 417), (16, 456), (1015, 451), (992, 417)]

count = 0
processed_numbers = set()

# Open file for writing car plate data
with open("car_plate_data.txt", "w") as file:
    file.write("NumberPlate\tDate\tTime\n")  # Writing column headers

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
            
            if text and text not in processed_numbers:
                # Add detected license plate number to the set
                processed_numbers.add(text)
                
                # Get current date and time
                current_datetime = datetime.now()
                date_str = current_datetime.strftime("%Y-%m-%d")
                time_str = current_datetime.strftime("%H:%M:%S")
                
                # Log detected number plate data to the file
                with open("car_plate_data.txt", "a") as file:
                    file.write(f"{text}\t{date_str}\t{time_str}\n")
                
                # Print log info
                print(f"Detected Number Plate: {text} at {date_str} {time_str}")
                
                # Blur the detected license plate area
                blurred_plate = cv2.GaussianBlur(crop, (51, 51), 30)  # Increase the kernel size to blur more
                frame[y1:y2, x1:x2] = blurred_plate  # Replace original frame region with blurred version
                
                # Display blurred image in the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow('crop', crop)

    cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 0), 2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()    
cv2.destroyAllWindows()
