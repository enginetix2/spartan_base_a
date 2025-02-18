import random
import streamlit as st

import torch
import cv2
import os
import yaml
import time
from ultralytics import YOLO

# --------------------------------------------------
# 0. Ensure Directory & Data File Structure
# --------------------------------------------------

DATA_YAML_PATH = "datasets/custom_dataset/data.yaml"

REQUIRED_DIRS = [
    "datasets/custom_dataset/images/train",
    "datasets/custom_dataset/images/val",
    "datasets/custom_dataset/labels/train",
    "datasets/custom_dataset/labels/val"
]

for d in REQUIRED_DIRS:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(DATA_YAML_PATH):
    default_data = {
        "train": "datasets/custom_dataset/images/train",
        "val": "datasets/custom_dataset/images/val",
        "names": []  # We'll add classes dynamically
    }
    with open(DATA_YAML_PATH, "w") as f:
        yaml.dump(default_data, f)

# --------------------------------------------------
# 1. Pick Device (GPU or CPU)
# --------------------------------------------------
device_type = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_type}")

# --------------------------------------------------
# 2. Global Model (start as None)
# --------------------------------------------------
model = None

# --------------------------------------------------
# Utility: Load & Save data.yaml
# --------------------------------------------------
def load_data_yaml(path=DATA_YAML_PATH):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data

def save_data_yaml(data, path=DATA_YAML_PATH):
    with open(path, "w") as f:
        yaml.dump(data, f)

# --------------------------------------------------
# Utility: Save YOLO label file
# --------------------------------------------------
def save_yolo_label(image_path, class_id, bbox, img_shape):
    h, w, _ = img_shape
    x1, y1, x2, y2 = bbox

    x_center = (x1 + x2) / 2.0 / w
    y_center = (y1 + y2) / 2.0 / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h

    normalized_path = image_path.replace("\\", "/")

    if "images/train" in normalized_path:
        label_path = normalized_path.replace("images/train", "labels/train")
    elif "images/val" in normalized_path:
        label_path = normalized_path.replace("images/val", "labels/val")
    else:
        raise ValueError(f"image_path not recognized: {image_path}")

    label_path = label_path.replace(".jpg", ".txt")
    with open(label_path, "w") as f:
        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# --------------------------------------------------
# Mode 1: Data Collection
# --------------------------------------------------
def data_collection():
    st.header("Data Collection")
    st.markdown("""
    **Instructions**:
    1. Enter a class name.
    2. Point your webcam at the object.
    3. We'll capture one bounding box around the center of the frame.
    4. Press "Capture Frame" to save an image + label.
    5. A certain percentage go to train, others go to val.
    """)

    data_yaml = load_data_yaml()

    # Choose ratio for train images
    train_ratio = st.slider("Train ratio", 0.0, 1.0, 0.8, 0.05)
    """
    If train_ratio=0.8, then 80% of captures go to train, 20% go to val.
    """

    class_name = st.text_input("Class Name", value="my_object")

    if class_name not in data_yaml["names"]:
        st.info(f"'{class_name}' will be added to data.yaml once you capture a frame.")

    if "capture_requested" not in st.session_state:
        st.session_state.capture_requested = False

    if st.button("Capture Frame"):
        st.session_state.capture_requested = True

    capture_mode = st.checkbox("Start Webcam Preview")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    box_size = 200

    while capture_mode:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not read from webcam.")
            break

        h, w, _ = frame.shape
        x_center = w // 2
        y_center = h // 2
        x1 = x_center - box_size // 2
        y1 = y_center - box_size // 2
        x2 = x_center + box_size // 2
        y2 = y_center + box_size // 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        if st.session_state.capture_requested:
            # Decide train or val by random
            if random.random() < train_ratio:
                subset = "train"
            else:
                subset = "val"

            timestamp = int(time.time() * 1000)
            img_filename = f"img_{class_name}_{timestamp}.jpg"
            img_path = os.path.join("datasets/custom_dataset/images", subset, img_filename)
            cv2.imwrite(img_path, frame)

            if class_name not in data_yaml["names"]:
                data_yaml["names"].append(class_name)
                save_data_yaml(data_yaml)

            class_id = data_yaml["names"].index(class_name)
            save_yolo_label(
                image_path=img_path,
                class_id=class_id,
                bbox=(x1, y1, x2, y2),
                img_shape=frame.shape
            )

            st.success(f"Captured image & label for class '{class_name}' in '{subset}' subset.")
            st.session_state.capture_requested = False

        time.sleep(0.05)

    cap.release()

# --------------------------------------------------
# Mode 2: Training
# --------------------------------------------------
def train_mode():
    st.header("Train New Model")
    st.markdown("""
    **Instructions**:
    1. Ensure you have images + labels in `datasets/custom_dataset/images/train` & `datasets/custom_dataset/labels/train`.
    2. Confirm your classes in data.yaml.
    3. Click "Start Training."
    """)
    data_yaml = load_data_yaml()

    st.write(f"Current Classes: {data_yaml['names']}")
    epochs = st.slider("Number of epochs", 1, 100, 50)

    if st.button("Start Training"):
        with st.spinner("Training in progress..."):
            # Initialize a fresh YOLO model
            new_model = YOLO("yolov8n.pt")
            results = new_model.train(
                data=DATA_YAML_PATH,
                epochs=epochs,
                imgsz=640,
                project="runs/train",
                name="custom_yolo",
                exist_ok=True,
                device=device_type
            )

        st.success("Training Complete!")

        best_model_path = f"{results.save_dir}/weights/best.pt"
        st.write(f"Best model saved at: {best_model_path}")

        # Update the global model with newly trained weights
        global model
        model = YOLO(best_model_path)
        st.success("Model updated with newly trained weights!")

        # For debugging, show the classes in the newly loaded model
        st.write("New model classes:", model.names)

# --------------------------------------------------
# Mode 3: Detection (Webcam)
# --------------------------------------------------
def detection_mode():
    st.header("Webcam Detection")
    conf_thres = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    run_detection = st.checkbox("Start Detection")

    placeholder = st.empty()
    cap = cv2.VideoCapture(0)

    while run_detection:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam failure.")
            break

        # If we have NOT loaded a model yet, default to "yolov8n.pt"
        global model
        if model is None:
            model = YOLO("yolov8n.pt")
            st.info("No custom model loaded yet; using default YOLOv8n.")

        results = model.predict(source=frame, conf=conf_thres, device=device_type)
        annotated_frame = frame.copy()

        for box in results[0].boxes:
            x1, y1, x2, y2, conf, cls_id = box.data[0]
            cls_id = int(cls_id.item())
            label = model.names[cls_id]
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        placeholder.image(annotated_frame)

        # For debugging, you could uncomment:
        # st.write("Current model classes:", model.names)
        time.sleep(0.03)

    cap.release()

# --------------------------------------------------
# Main Streamlit Interface
# --------------------------------------------------
def main():
    st.title("Simple YOLOv8 App (Data Collection, Training, Detection)")
    st.write(f"Using device: {device_type}")

    mode = st.sidebar.selectbox("Choose Mode", ["Data Collection", "Train", "Detection"])

    if mode == "Data Collection":
        data_collection()
    elif mode == "Train":
        train_mode()
    else:
        detection_mode()

if __name__ == "__main__":
    main()
