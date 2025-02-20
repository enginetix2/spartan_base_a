import cv2
import time
import threading
from queue import Queue
import os
import ollama
import numpy as np
from ultralytics import YOLO

################################
# 1) Word-Wrap + Timestampt
################################

def word_wrap(text, wrap_width=60):
    """
    Break 'text' into lines up to 'wrap_width' characters, splitting on spaces.
    Returns a list of line strings.
    """
    words = text.split()
    lines = []
    current_line = []
    current_len = 0

    for w in words:
        # If adding this word exceeds wrap_width, push the current line
        # else keep adding
        if current_len + len(w) + (1 if current_line else 0) > wrap_width:
            lines.append(" ".join(current_line))
            current_line = [w]
            current_len = len(w)
        else:
            current_line.append(w)
            current_len += len(w) + (1 if current_line else 0)

    if current_line:
        lines.append(" ".join(current_line))

    return lines

def render_text_scroll(comment_history, width=600, height=600, line_height=25, wrap_width=60):
    """
    Renders a scrolling text window onto a black image:
      - comment_history is a list of (timestamp_str, text).
      - We'll word-wrap each text and show from top to bottom.
      - If we have more lines than fit (height//line_height), older lines scroll off.
    """
    text_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Build all lines from the history
    all_lines = []
    for (tstamp, txt) in comment_history:
        # 1) Timestamp line
        all_lines.append(f"{tstamp} -")
        # 2) Word-wrap the text
        wrapped = word_wrap(txt, wrap_width=wrap_width)
        all_lines.extend(wrapped)
        # 3) Blank line after each commentary
        all_lines.append("")

    max_lines = height // line_height
    lines_to_show = all_lines[-max_lines:]  # show only last portion
    y = line_height
    for line in lines_to_show:
        cv2.putText(text_img, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        y += line_height

    return text_img

################################
# 2) Ollama calls (Vision + LLM)
################################

def call_vision_model_ollama(frame):
    """
    Saves frame to disk, calls llama3.2-vision with a short prompt,
    returns text describing the image.
    """
    temp_path = "temp_vision_input.jpg"
    cv2.imwrite(temp_path, frame)

    response = ollama.chat(
        model="moondream",
        messages=[
            {
                "role": "user",
                "content": "Briefly describe this image.",
                "images": [temp_path]
            }
        ],
        stream=False
    )
    msg = response.get("message", {})
    vision_text = msg.get("content", "").strip()
    return vision_text if vision_text else "No vision output."

def call_text_llm_ollama(vision_text, bounding_boxes):
    """
    Summarizes the combined result. bounding_boxes is a list of (label, conf, x1, y1, x2, y2).
    Returns a short text.
    """
    if not bounding_boxes:
        detection_summary = "No objects."
    else:
        detection_summary = []
        for (lab, conf, x1, y1, x2, y2) in bounding_boxes:
            detection_summary.append(f"{lab} ({conf*100:.1f}%)")
        detection_summary = ", ".join(detection_summary)

    prompt = (
        f"Vision says: {vision_text}\n"
        f"YOLO sees: {detection_summary}\n"
        "Give a very concise combined statement about this scene."
    )

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )
    msg = response.get("message", {})
    text_output = msg.get("content", "").strip()
    return text_output if text_output else "No LLM output."

################################
# 3) Worker Thread
################################

def llm_worker(task_queue, comment_lock, comment_history):
    """
    Worker thread loop:
      - Each task is (frame, bounding_boxes).
      - Calls vision + text LLM, appends new commentary with a timestamp to comment_history.
    """
    while True:
        task = task_queue.get()
        if task is None:
            print("[Worker] Stopping.")
            break

        frame, bounding_boxes = task
        # 1) Vision call
        vision_text = call_vision_model_ollama(frame)
        # 2) LLM call
        commentary = call_text_llm_ollama(vision_text, bounding_boxes)

        # 3) Store the commentary + timestamp
        now_str = time.strftime("%H:%M:%S")  # or add date if you prefer
        with comment_lock:
            comment_history.append((now_str, commentary))

        task_queue.task_done()

################################
# 4) Main
################################

def main():
    print("[Main] Loading YOLO model (best.pt)")
    model = YOLO("runs/train/custom_yolo/weights/best.pt")

    # Start the worker thread
    from queue import Queue
    task_queue = Queue()
    comment_lock = threading.Lock()
    comment_history = []

    worker = threading.Thread(
        target=llm_worker,
        args=(task_queue, comment_lock, comment_history),
        daemon=True
    )
    worker.start()

    print("[Main] Opening webcam")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Main] Error: cannot open webcam.")
        return

    cv2.namedWindow("YOLO Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("LLM Scroll", cv2.WINDOW_NORMAL)

    frame_count = 0
    FRAME_INTERVAL = 50  # do LLM call every 50 frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection at conf=0.01
        results = model.predict(frame, conf=0.01, device=0)
        bounding_boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2, conf, cls_id = box.data[0]
            cls_id = int(cls_id.item())
            label = model.names[cls_id]
            bounding_boxes.append((label, float(conf.item()), int(x1), int(y1), int(x2), int(y2)))

        # Draw boxes
        for (lab, c, x1, y1, x2, y2) in bounding_boxes:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(frame, f"{lab} {c:.2f}", (x1, max(y1-10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Every N frames, queue a task
        if frame_count % FRAME_INTERVAL == 0:
            frame_copy = frame.copy()
            task_queue.put((frame_copy, bounding_boxes))

        # Show YOLO detection
        cv2.imshow("YOLO Detection", frame)

        # Show the LLM scroll
        with comment_lock:
            text_img = render_text_scroll(comment_history, width=600, height=600,
                                          line_height=25, wrap_width=60)
        cv2.imshow("LLM Scroll", text_img)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Stop worker
    task_queue.put(None)
    worker.join()
    print("[Main] Exiting cleanly.")

if __name__ == "__main__":
    main()
