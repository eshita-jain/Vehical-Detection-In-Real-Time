# Vehicle Number Plate Detection In Real Time

![image](https://github.com/eshita-jain/xy/assets/80577092/82be4762-2225-472b-826a-c16b5a50aaad)

## Introduction

Vehicle Number Plate Detection is a project that utilizes computer vision and deep learning techniques to detect and recognize license plates on vehicles. This project is designed to assist in automating tasks related to vehicle identification, such as parking management, toll collection, and security monitoring. By accurately detecting and recognizing license plates, it can help streamline various processes and enhance security measures.

## Features

- Detect and recognize license plates in images and videos.
- Extract license plate information, including the plate number and, if available, the owner's name.
- Real-time processing of video streams for continuous monitoring.
- User-friendly graphical user interface (GUI) for easy interaction.

## Problem Statement

The manual recognition of license plates can be a time-consuming and error-prone task, especially in situations where there is a need to process a large volume of vehicle data. Traditional methods rely on human operators to read and record license plate information, which can lead to inaccuracies and inefficiencies. The Vehicle Number Plate Detection project aims to address this problem by automating the process, making it faster and more reliable.

## Installation

To run this project, follow these installation steps:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/vehicle-number-plate-detection.git
   ```

2. Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. Download and install Tesseract OCR from the official website: [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

4. Specify the path to the Tesseract executable in the code by modifying the `pytesseract.pytesseract.tesseract_cmd` variable:

   ```python
   pytesseract.pytesseract.tesseract_cmd = r'path/to/your/tesseract/executable'
   ```

5. Download the YOLOv5 custom model checkpoint file (best.pt) and place it in the project directory.

6. Prepare a CSV file containing vehicle data with columns 'Number Plate' and 'Owner' and save it as 'Vehicle-data.csv' in the project directory.

## How It Works

The Vehicle Number Plate Detection project consists of several key components:

1. **Object Detection:** The project uses the YOLOv5 deep learning model to perform object detection on input images or video frames. It detects vehicles and their bounding boxes.

2. **License Plate Recognition:** For each detected vehicle, the system extracts the region of interest (ROI) containing the license plate. It then uses Tesseract OCR to recognize the text on the license plate.

3. **Data Lookup:** The recognized license plate number is used to look up the corresponding owner's name in the vehicle data CSV file.

4. **Visualization:** The system visualizes the results by drawing bounding boxes around the detected vehicles and displaying the license plate number and owner's name on the image or video frame.

5. **Output:** The results, including the license plate number and owner's name, are saved to a CSV file for future reference.

The GUI allows users to browse and process both images and videos, providing a user-friendly interface for utilizing the license plate recognition system.

## Results

After running the Vehicle Number Plate Detection project, you will obtain the following results:

![WhatsApp Image 2023-09-03 at 01 39 40](https://github.com/eshita-jain/xy/assets/80577092/0be1f9cb-51ea-4093-beff-e59bcc233eb7)

- Detected license plates with bounding boxes drawn around them in the processed images or video frames.

**![WhatsApp Image 2023-09-03 at 01 39 59](https://github.com/eshita-jain/xy/assets/80577092/468e563a-023c-4fb2-b0bb-a694d49f978a)
**
- Recognized license plate numbers and, if available, the corresponding owner's names.

![WhatsApp Image 2023-09-03 at 01 40 49](https://github.com/eshita-jain/xy/assets/80577092/a2b35675-a4a5-4fe9-9689-7c9a7b80d0a4)

- Results saved to a CSV file named 'results.csv' for further analysis and reference.

Feel free to explore and use the project to enhance your applications related to vehicle identification and management.

