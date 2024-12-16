import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

# Load YOLOv8 models and PaddleOCR
date_model = YOLO("Best_For_Date.pt")
dmy_classifier = YOLO("Best_For_DMY.pt")
ocr = PaddleOCR(use_angle_cls=True, lang="en")

# Function to compute Intersection over Union (IoU)
def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2

    inter_x1 = max(x1, x1_)
    inter_y1 = max(y1, y1_)
    inter_x2 = min(x2, x2_)
    inter_y2 = min(y2, y2_)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0
    return iou

def filter_overlapping_boxes(boxes):
    filtered_boxes = []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)

    while boxes:
        best_box = boxes.pop(0)
        filtered_boxes.append(best_box)
        boxes = [box for box in boxes if compute_iou(best_box[:4], box[:4]) <= 0.5]

    return filtered_boxes

# Function to extract date regions and process the image
def extract_date_regions(image_path):
    image = cv2.imread(image_path)
    results = date_model(image)

    if len(results[0].boxes) == 0:
        return "No regions detected"

    detected_boxes = []
    for box in results[0].boxes.data:
        x1, y1, x2, y2, confidence, class_id = box.cpu().numpy()
        if confidence < 0.5:
            continue
        detected_boxes.append([x1, y1, x2, y2, confidence])

    filtered_boxes = filter_overlapping_boxes(detected_boxes)

    detected_dates = []
    for x1, y1, x2, y2, confidence in filtered_boxes:
        region = image[int(y1):int(y2), int(x1):int(x2)]
        dmy_results = dmy_classifier(region)
        if len(dmy_results[0].boxes) < 1:
            continue

        dmy_boxes = dmy_results[0].boxes.data
        dmy_dict = {"day": None, "month": None, "year": None}
        for boxo in dmy_boxes:
            x1_, y1_, x2_, y2_, confidence_, class_id_ = boxo.cpu().numpy()
            class_name = {0: "day", 1: "month", 2: "year"}[int(class_id_)]
            cropped_region = region[int(y1_):int(y2_), int(x1_):int(x2_)]

            if class_name == "day":
                dmy_dict["day"] = cropped_region
            elif class_name == "month":
                dmy_dict["month"] = cropped_region
            elif class_name == "year":
                dmy_dict["year"] = cropped_region

        def ocr_extract(region):
            result = ocr.ocr(region, cls=True)
            text = result[0][1][0] if result and result[0] else ""
            return text.strip().upper()

        day = ocr_extract(dmy_dict["day"]) if dmy_dict["day"] is not None else "1"
        month = ocr_extract(dmy_dict["month"]) if dmy_dict["month"] is not None else "1"
        year = ocr_extract(dmy_dict["year"]) if dmy_dict["year"] is not None else "1"


        formatted_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
        detected_dates.append({
            "formatted_date": formatted_date,
            "bounding_box": [int(x1), int(y1), int(x2), int(y2)]
        })

    return detected_dates

# Streamlit UI
st.title("Expiry Date Prediction")

# Image uploader
uploaded_image = st.file_uploader("Upload an image with expiry date", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Save uploaded image to a temporary file
    image_path = "uploaded_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Process image and extract dates
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write("Processing...")

    # Get output
    output = extract_date_regions(image_path)

    # Display results
    if output == "No regions detected":
        st.write("No expiry dates detected.")
    else:
        for item in output:
            st.write(f"Detected Expiry Date: {item['formatted_date']}")
            st.write(f"Bounding Box: {item['bounding_box']}")

            # Optionally, display the cropped region
            cropped_image = cv2.imread(image_path)
            x1, y1, x2, y2 = item['bounding_box']
            cropped_region = cropped_image[int(y1):int(y2), int(x1):int(x2)]
            st.image(cropped_region, caption=f"Cropped Date Region - {item['formatted_date']}", use_column_width=True)
