import cv2
import time
import threading
from queue import Queue
import os
import ollama
import base64
from ultralytics import YOLO

########################
# Vision Model (Ollama llama3.2-vision)
########################

def call_vision_model_ollama(frame):
    """
    Sends the frame to Ollama's llama3.2-vision model.
    Returns a short textual description of the image.
    
    Note: Must have 'llama3.2-vision' installed in Ollama.
    """
    # 1) Save frame to disk so Ollama can read it as an image
    temp_path = "temp_vision_input.jpg"
    cv2.imwrite(temp_path, frame)

    # 2) Call Ollama with 'images': [temp_path]
    #    The 'messages' structure might differ if your version needs something else.
    response = ollama.chat(
        model="moondream",
        messages=[
            {
                "role": "user",
                "content": "Very concise: describe this image in detail.",
                "images": [temp_path],
                "max_tokens": 64,
            }
        ],
        stream=False  # synchronous for the sake of clarity
    )

    # 3) Extract text from the response
    #    Typically it's in response["message"]["content"]
    message = response.get("message", {})
    vision_text = message.get("content", "").strip()
    return vision_text if vision_text else "No vision output."

########################
# Text LLM Model (Ollama llama3.2)
########################

def call_llm_model_ollama(vision_text, bounding_boxes):
    """
    Merges the vision text + YOLO bounding boxes into a final commentary.
    Calls 'llama3.2' model in Ollama for text-based summarization.
    """
    # bounding_boxes is [(label, conf, x1, y1, x2, y2), ...]
    if not bounding_boxes:
        detection_summary = "No objects detected"
    else:
        detection_summary = []
        for (lab, conf, x1, y1, x2, y2) in bounding_boxes:
            detection_summary.append(f"{lab} ({conf*100:.1f}% conf)")
        detection_summary = ", ".join(detection_summary)

    # A short prompt combining both
    prompt_text = (
        f"Vision model says: {vision_text}\n"
        f"YOLO sees: {detection_summary}.\n"
        "Please provide a combined commentary about this scene in one short sentence."
    )

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt_text,
                "max_tokens": 32,
            }
        ],
        stream=False  # blocking call
    )
    message = response.get("message", {})
    text_output = message.get("content", "").strip()
    return text_output if text_output else "No LLM output."

########################
# Worker Thread
########################

def llm_worker(task_queue, latest_comment_lock, latest_comment_dict):
    """
    Continuously fetch tasks from task_queue:
      - Each task is (frame, bounding_boxes).
      - Calls vision + text LLM, updates shared commentary in latest_comment_dict.
    """
    while True:
        task = task_queue.get()
        if task is None:
            # Signal to shutdown
            break
        
        frame, bounding_boxes = task
        
        # 1) Vision call
        vision_text = call_vision_model_ollama(frame)

        # 2) LLM call
        commentary = call_llm_model_ollama(vision_text, bounding_boxes)

        # 3) Update shared commentary
        with latest_comment_lock:
            latest_comment_dict["text"] = commentary

        task_queue.task_done()

########################
# Word-Wrapping for Overlay
########################

def wrap_text(img, text, start_x, start_y, line_height=25, color=(255,255,255), thickness=2):
    """
    Naive word-wrap for text overlay. Splits text on spaces, wraps ~60 chars.
    """
    words = text.split()
    line = []
    current_length = 0
    for w in words:
        if current_length + len(w) + 1 > 60:  # wrap at ~60
            cv2.putText(img, " ".join(line), (start_x, start_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
            start_y += line_height
            line = [w]
            current_length = len(w)
        else:
            line.append(w)
            current_length += len(w) + 1

    if line:
        cv2.putText(img, " ".join(line), (start_x, start_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

########################
# Main App
########################

def main():
    # 1) Load YOLO detection model
    model = YOLO("runs/train/custom_yolo/weights/best.pt")

    # 2) Start worker thread for LLM
    task_queue = Queue()
    latest_comment_lock = threading.Lock()
    latest_comment_dict = {"text": "Starting commentary..."}
    worker = threading.Thread(
        target=llm_worker,
        args=(task_queue, latest_comment_lock, latest_comment_dict),
        daemon=True
    )
    worker.start()

    # 3) Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open webcam.")
        return

    cv2.namedWindow("YOLO + LLM (Async)", cv2.WINDOW_NORMAL)

    frame_count = 0
    FRAME_INTERVAL = 50  # LLM calls every 50 frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection at conf=0.01
        results = model.predict(frame, conf=0.01, device=0)

        # Gather bounding boxes
        bounding_boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2, conf, cls_id = box.data[0]
            cls_id = int(cls_id.item())
            label = model.names[cls_id]
            bounding_boxes.append((label, float(conf.item()), int(x1), int(y1), int(x2), int(y2)))

        # Draw YOLO boxes
        for (lab, c, x1, y1, x2, y2) in bounding_boxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{lab} {c:.2f}", (x1, max(y1-10,10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Every 50 frames, enqueue a new LLM task
        if frame_count % FRAME_INTERVAL == 0:
            # Copy the frame so main loop isn't blocked
            frame_copy = frame.copy()
            task_queue.put((frame_copy, bounding_boxes))

        # Overlay latest commentary
        with latest_comment_lock:
            commentary = latest_comment_dict["text"]

        wrap_text(frame, commentary, start_x=20, start_y=30)

        cv2.imshow("YOLO + LLM (Async)", frame)
        frame_count += 1

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Signal worker to stop
    task_queue.put(None)
    worker.join()

if __name__ == "__main__":
    main()
