import random
import numpy as np
import streamlit as st
import torch
import cv2
import os
import yaml
import time
from ultralytics import YOLO

# --------------------------------------------------
# 0. Directory Setup
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
        "names": []
    }
    with open(DATA_YAML_PATH, "w") as f:
        yaml.dump(default_data, f)

# --------------------------------------------------
# 1. Pick Device
# --------------------------------------------------
device_type = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_type}")

# --------------------------------------------------
# 2. Global Model (optional)
# --------------------------------------------------
model = None

# --------------------------------------------------
# Utility: data.yaml loader/saver
# --------------------------------------------------
def load_data_yaml(path=DATA_YAML_PATH):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_data_yaml(data, path=DATA_YAML_PATH):
    with open(path, "w") as f:
        yaml.dump(data, f)

# --------------------------------------------------
# Utility: Save YOLO label file
# --------------------------------------------------
def save_yolo_label(image_path, class_id, bbox, img_shape):
    normalized_path = image_path.replace("\\", "/")
    h, w, _ = img_shape
    x1, y1, x2, y2 = bbox

    x_center = (x1 + x2) / 2.0 / w
    y_center = (y1 + y2) / 2.0 / h
    width = (x2 - x1) / w
    height = (y2 - y1) / h

    if "images/train" in normalized_path:
        label_path = normalized_path.replace("images/train", "labels/train")
    elif "images/val" in normalized_path:
        label_path = normalized_path.replace("images/val", "labels/val")
    else:
        raise ValueError(f"image_path not recognized: {image_path}")

    label_path = label_path.replace(".jpg", ".txt")
    label_path = label_path.replace("/", os.sep)

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
    train_ratio = st.slider("Train ratio", 0.0, 1.0, 0.8, 0.05)
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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB")

        if st.session_state.capture_requested:
            import random
            subset = "train" if random.random() < train_ratio else "val"
            timestamp = int(time.time() * 1000)
            img_filename = f"img_{class_name}_{timestamp}.jpg"
            img_path = os.path.join("datasets/custom_dataset/images", subset, img_filename)
            cv2.imwrite(img_path, frame)

            if class_name not in data_yaml["names"]:
                data_yaml["names"].append(class_name)
                save_data_yaml(data_yaml)

            class_id = data_yaml["names"].index(class_name)
            save_yolo_label(img_path, class_id, (x1, y1, x2, y2), frame.shape)

            st.success(f"Captured image & label for '{class_name}' in '{subset}' subset.")
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
            new_model = YOLO("yolo11n.pt")
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

        st.session_state["trained_model_path"] = best_model_path
        global model
        model = YOLO(best_model_path)
        st.success("Model updated with newly trained weights!")
        st.write("New model classes:", model.names)

# --------------------------------------------------
# Mode 3: Detection (Static)
# --------------------------------------------------
def detection_mode():
    st.header("Static Detection Mode")
    best_model_path = "./runs/train/custom_yolo/weights/best.pt"
    local_model = YOLO(best_model_path)
    st.write("Loaded classes:", local_model.names)

    conf_thres = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    uploaded_file = st.file_uploader("Upload an image (JPEG/PNG)", type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        results = local_model.predict(img_bgr, conf=conf_thres)

        for box in results[0].boxes:
            x1, y1, x2, y2, conf, cls_id = box.data[0]
            cls_id = int(cls_id.item())
            label = local_model.names[cls_id]
            cv2.rectangle(img_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
            cv2.putText(img_bgr, f"{label} {conf:.2f}",
                        (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.image(img_rgb, caption="Detection Result")
    else:
        st.info("Upload an image to run detection.")

# --------------------------------------------------
# Mode 4: Streaming - Runs EXACT known code in a separate window
# --------------------------------------------------
def streaming_mode():
    st.header("Live Streaming (OpenCV window)")

    st.markdown("""
    **Instructions**:
    - This will open a native OpenCV window with your detection code.
    - Adjust the slider below to set the confidence threshold.
    - Press **Start** to run. In the window, press **'q'** to quit streaming.
    """)

    # 1) Add a slider for confidence threshold
    conf_thres = st.slider("Confidence Threshold", 0.0, 1.0, 0.01, 0.01)

    if st.button("Start"):
        st.info("Launching the OpenCV stream. Look for the 'YOLO Live' window. Press 'q' there to quit.")
        run_opencv_snippet(conf_thres)

def run_opencv_snippet(conf_thres):
    """
    Your known-working detection code, but we overlay:
    'Press q to exit...'
    """
    model_path = "./runs/train/custom_yolo/weights/best.pt"
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf_thres, device=0)

        for box in results[0].boxes:
            x1, y1, x2, y2, conf, cls_id = box.data[0]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_id = int(cls_id.item())
            label = model.names[cls_id]
            confidence = float(conf.item())

            # Draw the bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            # Put the label text
            cv2.putText(frame,
                        f"{label} {confidence:.2f}",
                        (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

        # 2) Overlay "Press q to exit..." near the top-left or bottom-left corner
        cv2.putText(frame, 
                    "Press 'q' to exit...", 
                    (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, 
                    (255, 255, 255), 
                    2)

        cv2.imshow("YOLO Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# --------------------------------------------------
# Main Streamlit Interface
# --------------------------------------------------
def main():
    st.title("Simple YOLOv11 App")

    mode = st.sidebar.selectbox("Choose Mode", ["Data Collection", "Train", "Detection", "Streaming"])

    if mode == "Data Collection":
        data_collection()
    elif mode == "Train":
        train_mode()
    elif mode == "Detection":
        detection_mode()
    elif mode == "Streaming":
        streaming_mode()

if __name__ == "__main__":
    main()


