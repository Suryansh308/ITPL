# pip install streamlit easyocr pdf2image opencv-python pillow
# Main Streamlit App

import streamlit as st
import easyocr
import numpy as np
import cv2
from PIL import Image


from pdf2image import convert_from_bytes
import os

# Set poppler path manually
POPPLER_PATH = r"C:\Poppler\poppler-24.08.0\Library\bin"  # adjust this to your exact path



# Preprocessing
def preprocess_for_handwriting(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 15
    )
    return thresh

#App UI
st.set_page_config(page_title="Handwritten PDF OCR", layout="wide")
st.title("📝 Handwritten Text Recognition from PDF (EasyOCR)")
st.markdown("Upload a PDF containing handwritten pages and view OCR results with bounding boxes.")

#Upload PDF
uploaded_file = st.file_uploader("📄 Upload a PDF", type=["pdf"])

if uploaded_file:
    images = convert_from_bytes(uploaded_file.read(), dpi=400, poppler_path=POPPLER_PATH)
    total_pages = len(images)
    all_page_numbers = list(range(1, total_pages + 1))
    
    col1, col2 = st.columns([3, 1])
    with col1:
        selected_pages = st.multiselect(
            "📃 Select page(s) to process", all_page_numbers, default=[1]
        )
    with col2:
        if st.checkbox("Select all"):
            selected_pages = all_page_numbers

    # Initialize OCR model
    reader = easyocr.Reader(['en'], gpu=False)

    # Process selected pages
    for page_idx in selected_pages:
        st.markdown(f"---\n### 📄 Page {page_idx}")
        pil_image = images[page_idx - 1]
        img_rgb = np.array(pil_image)

        # Preprocess and OCR
        processed_img = preprocess_for_handwriting(img_rgb)
        results = reader.readtext(processed_img)

        # Draw boxes
        img_out = img_rgb.copy()
        for bbox, text, conf in results:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(img_out, [pts], True, (0, 255, 0), 2)
            cv2.putText(img_out, text, (int(pts[0][0]), int(pts[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Show image with boxes
        st.image(img_out, caption=f"Detected Text - Page {page_idx}", use_column_width=True)

        # Show text results
        with st.expander(f"📋 Detected Text on Page {page_idx}"):
            for _, text, conf in results:
                st.markdown(f"- **{text}**  _(confidence: {conf:.2f})_")
