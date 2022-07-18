import os
import torch
import cv2
import csv
import numpy as np
import pytesseract
from PIL import ImageTk, Image
from tkinter import Tk, Label, Button, filedialog

# Specify the path to the Tesseract executable if it's not in your system's PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# DEFINING GLOBAL VARIABLE
OCR_TH = 0.2

# -------------------------------------- function to run detection ---------------------------------------------------------
def detectx(frame, model):
    frame = [frame]
    print("[INFO] Detecting...")
    results = model(frame)
    labels, coordinates = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, coordinates

### ------------------------------------ to plot the BBox and results --------------------------------------------------------
def plot_boxes(results, frame, classes, vehicle_data, csv_writer):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]

    print(f"[INFO] Total {n} detections...")
    print("[INFO] Looping through all detections...")

    owner_name = "Owner not found"
    plate_num = ""

    for i in range(n):
        row = cord[i]
        if row[4] >= 0.55:
            print("[INFO] Extracting BBox coordinates...")
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            text_d = classes[int(labels[i])]
            coords = [x1, y1, x2, y2]
            plate_num = recognize_plate_tesseract(frame, coords, OCR_TH)

            owner_name = vehicle_data.get(plate_num, "Owner not found")

            text = f"{plate_num} - {owner_name}"

            # Draw green box around license plate and owner text
            text_width, text_height = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1-text_height-10), (x1+text_width+10, y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Draw red box around license plate only
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            print("Result :")
            print(f"Number Plate = {plate_num}")
            print(f"The Owner of the Vehicle = {owner_name}")

            # Write the results to the CSV file
            csv_writer.writerow([plate_num, owner_name])

    return frame

#### ---------------------------- function to recognize license plate using Tesseract --------------------------------------
def recognize_plate_tesseract(img, coords, region_threshold):
    xmin, ymin, xmax, ymax = coords
    nplate = img[int(ymin):int(ymax), int(xmin):int(xmax)]
    # Convert the cropped image to grayscale (required for Tesseract)
    nplate_gray = cv2.cvtColor(nplate, cv2.COLOR_BGR2GRAY)
    # Perform OCR using Tesseract
    plate_num = pytesseract.image_to_string(nplate_gray, config='--psm 6')
    # Filter and clean the recognized text
    plate_num = filter_text_tesseract(plate_num, region_threshold)
    return plate_num

def filter_text_tesseract(text, region_threshold):
    text = text.strip()
    if len(text) == 1:
        text = text.upper()
    return text

### ---------------------------------------------- Main function -----------------------------------------------------

def load_vehicle_data(file_path):
    vehicle_data = {}
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            plate = row['Number Plate']
            owner = row['Owner']
            vehicle_data[plate.strip()] = owner.strip()
    return vehicle_data

def process_image(img_path, image_label, text_label, browse_button):
    print("[INFO] Loading model...")
    model = torch.hub.load('./yolov5-master', 'custom', source='local', path='best.pt', force_reload=True)
    classes = model.names

    # Load vehicle data from file
    vehicle_data = load_vehicle_data('Vehicle-data.csv')
    frame = cv2.imread(img_path)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detectx(frame, model=model)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Create a CSV file to save the results
    csv_filename = 'results.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Number Plate', 'Owner'])

        frame = plot_boxes(results, frame, classes=classes, vehicle_data=vehicle_data, csv_writer=csv_writer)

    # Convert output image to PIL format
    image = Image.fromarray(frame)
    image = image.resize((500, 350), Image.ANTIALIAS)
    img_tk = ImageTk.PhotoImage(image)

    # Display the output image on the GUI
    image_label.configure(image=img_tk)
    image_label.image = img_tk

    # Display the output text on the GUI
    text_label.configure(text="Results saved to 'results.csv'")

    # Enable Browse Image button
    browse_button.configure(state="normal")

def process_video(video_path, image_label, text_label, browse_button):
    print("[INFO] Loading model...")
    model = torch.hub.load('./yolov5-master', 'custom', source='local', path='best.pt', force_reload=True)
    classes = model.names

    # Load vehicle data from file
    vehicle_data = load_vehicle_data('Vehicle-data.csv')

    cap = cv2.VideoCapture(video_path)

    # Create a CSV file to save the results
    csv_filename = 'results.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Number Plate', 'Owner'])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detectx(frame, model=model)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = plot_boxes(results, frame, classes=classes, vehicle_data=vehicle_data, csv_writer=csv_writer)

            # Convert output image to PIL format
            image = Image.fromarray(frame)
            image = image.resize((500, 350), Image.ANTIALIAS)
            img_tk = ImageTk.PhotoImage(image)

            # Display the output image on the GUI
            image_label.configure(image=img_tk)
            image_label.image = img_tk

            # Update the GUI
            root.update()

    cap.release()

    # Display the output text on the GUI
    text_label.configure(text="Results saved to 'results.csv'")

    # Enable Browse Video button
    browse_button.configure(state="normal")

def open_media(image_label, text_label, browse_button):
    root = Tk()
    root.withdraw()
    media_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png"), ("Video Files", "*.mp4 *.avi *.mkv")])
    if media_path:
        # Disable Browse Media button
        browse_button.configure(state="disabled")
        media_type = get_media_type(media_path)
        if media_type == "image":
            process_image(media_path, image_label, text_label, browse_button)
        elif media_type == "video":
            process_video(media_path, image_label, text_label, browse_button)
    root.destroy()

def get_media_type(media_path):
    if os.path.isfile(media_path):
        _, file_extension = os.path.splitext(media_path)
        if file_extension.lower() in ('.jpg', '.jpeg', '.png'):
            return "image"
        elif file_extension.lower() in ('.mp4', '.avi', '.mkv'):
            return "video"
    return None

def main():
    root = Tk()
    root.title("License Plate Recognition")
    root.geometry("800x600")

    label = Label(root, text="License Plate Recognition", font=("Arial", 16))
    label.pack(pady=20)

    image_label = Label(root)
    image_label.pack()

    text_label = Label(root, text="", font=("Arial", 12))
    text_label.pack(pady=10)

    browse_button = Button(root, text="Browse Media", command=lambda: open_media(image_label, text_label, browse_button))
    browse_button.pack(pady=10)

    root.mainloop()

main()