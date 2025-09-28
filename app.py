import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load your trained model
MODEL_PATH = "runs/detect/train2/weights/best.pt"  # adjust path if needed
model = YOLO(MODEL_PATH)

st.title("🔍 IC Defect Detection")
st.write("Upload an image of an IC, and the model will detect if it has defects.")

# File uploader
uploaded_file = st.file_uploader("Upload an IC image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO prediction (save=False keeps everything in memory)
    with st.spinner("Detecting..."):
        results = model.predict(image, conf=0.25, save=False)

    # Show results directly without saving
    result_image = results[0].plot()  # numpy array with boxes/labels drawn
    st.image(result_image, caption="Detection Result", use_column_width=True)

    # Optional: show raw results
    st.write("Detections:", results[0].boxes.data.cpu().numpy())
