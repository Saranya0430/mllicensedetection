# Automated License Plate Recognition (ALPR) System

## Overview

This project is a comprehensive Automated License Plate Recognition (ALPR) system designed for real-time detection and recognition of vehicle license plates in video feeds. It combines advanced object detection and Optical Character Recognition (OCR) methods to provide a robust, scalable, and privacy-conscious solution for various applications, including traffic monitoring, law enforcement, and secure access control.

Key features include:
- **Real-time License Plate Detection** using YOLO (You Only Look Once) for high-speed and accurate plate localization.
- **Optical Character Recognition (OCR)** using Pytesseract to extract alphanumeric characters from license plates.
- **Privacy Protection** through Gaussian blur on detected license plates to safeguard sensitive data.
- **Data Logging** with timestamps for historical tracking and analysis.

## Table of Contents
- [Architecture](#architecture)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Results](#results)
- [Future Enhancements](#future-enhancements)

## Architecture

The ALPR system is organized in modules:
1. **Data Acquisition**: Captures video feeds and processes frames individually.
2. **Pre-processing**: Converts frames to grayscale and applies bilateral filtering to enhance clarity.
3. **License Plate Detection**: YOLO model locates plates, refined with Region of Interest (ROI) filtering.
4. **OCR Processing**: Pytesseract extracts characters, validated against standard license plate formats.
5. **Privacy Protection**: Gaussian blur is selectively applied to protect license plate information.
6. **Data Logging**: Logs detected plates with timestamps in a `.csv` file for analysis.

## Features

- **Scalability**: Operates on both standard CPUs and higher-end GPUs.
- **Real-time Processing**: Efficient for high-traffic applications.
- **Privacy Control**: Selective blurring maintains compliance with privacy standards.
- **Cross-platform Support**: Runs on Windows, macOS, and Linux.

## Installation

To run this ALPR system, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PutluruAravindaReddy/License_plate_recognition.git
   cd ALPR-System

2. **Install dependencies**:
- Ensure Python 3.8+ is installed.
- Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt

3. **Download YOLO Model:**
- Place the YOLO model file (best.pt) in the root project directory. You can train your custom model or use a pre-trained model compatible with the YOLO architecture.

4. **Set up Tesseract:**
- Download and install Tesseract OCR (installation guide).
- Set the pytesseract command path in main.py:
    ```bash
    pytesseract.pytesseract.tesseract_cmd = r'path/to/tesseract.exe'

5. **Run the System:**
- To initiate real-time video processing via the webcam:
    ```bash
    python main.py
- Alternatively, to analyze a pre-recorded video:
    ```bash
    cap = cv2.VideoCapture('path/to/video.mp4')

## Usage

1. **Video Feed Processing:**
- The system captures frames from a video feed or webcam.
- Each frame undergoes pre-processing, license plate detection, OCR, and privacy protection.

2. **Data Logging:**
- Detected license plates are saved in a .csv file (car_plate_data.txt) with corresponding timestamps.

## Web Deployment On Render

This repository now includes a deployable Flask web app for Render.

### Files added for deployment
- `app.py` - Flask web app for image/video upload and ALPR processing
- `templates/index.html` - upload UI and result page
- `requirements.txt` - Python dependencies
- `Dockerfile` - installs Python dependencies and Tesseract OCR
- `render.yaml` - Render service definition

### Local run

```bash
python app.py
```

Then open `http://localhost:5000`.

### Deploy on Render

1. Push this repository to GitHub.
2. Sign in to Render and choose **New +** -> **Blueprint**.
3. Select your GitHub repo.
4. Render will detect `render.yaml` and create the web service automatically.
5. After deploy completes, open the generated Render URL and upload an image or video.

### Important deployment notes

- The web deployment uses Docker so Tesseract OCR is installed on the server.
- The training dataset and large local videos are excluded from the Docker build using `.dockerignore`.
- Keep `best.pt` in the repository root because the web app loads that model at runtime.

## File Structure

- **main.py** : Core script for running the ALPR system.
- **requirements.txt** : Lists all Python dependencies.
- **best.pt** : YOLO model file (download separately).
- **car_plate_data.txt** : Logs detected license plate data.
- **coco1.txt** : Contains YOLO class labels.


## Results

1. **Detection Accuracy**:
   - Achieves ~92% accuracy in well-lit environments.
   - Maintains ~85% accuracy under challenging conditions.

2. **OCR Accuracy**:
   - Reaches ~90% accuracy in standard conditions, with character validation to minimize errors.

3. **Real-time Performance**:
   - Processes frames at 20 FPS on a mid-range GPU, suitable for real-time applications.

4. **Example Detected Plate Log**:
   - Saved in `car_plate_data.txt` with format:
     ```plaintext
     NumberPlate     Date        Time
     ABC1234         2024-11-01   14:30:45
     ```

## Future Enhancements

1. **Enhanced Privacy**:
   - Further refine Gaussian blur application to adjust based on context, enhancing privacy protection.

2. **Improved Accuracy**:
   - Explore adaptive thresholding techniques for improved performance in low-light conditions.

3. **Model Efficiency**:
   - Optimize the YOLO model to support edge devices and low-power applications, increasing deployment versatility.

## Contributing

This project was developed by a collaborative team under the guidance of our faculty mentor, [Dr. Hari Haran](https://www.srmist.edu.in/faculty/dr-hariharan-r/), ML Faculty at SRM University. 

### Team

The team consists of **ARAVINDA REDDY PUTLURU** (Team Head, [aravindareddy60@gmail.com](mailto:aravindareddy60@gmail.com)), **Vivek Vanga** ([vivekvanga@example.com](mailto:vivekvanga@example.com)), **Krishna Samhitha K** ([krishnasamhithachillara@gmail.com](mailto:krishnasamhithachillara@gmail.com)), and **Sairam T** ([sairam.temmanaboyina@gmail.com](mailto:sairam.temmanaboyina@gmail.com)).

For any inquiries or contributions, feel free to contact any team member via email. Contributions are welcome! Please fork the repository and submit a pull request for any improvements or features.
