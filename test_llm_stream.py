import cv2
import time
import os
import ollama
from ultralytics import YOLO

def wrap_text_on_frame(frame, text, x, y, max_width_px=500, line_height_px=25, font_scale=0.6, thickness=2, color=(255,255,255)):
    """
    Word-wraps 'text' so it doesn't exceed 'max_width_px' in the given 'frame'.
    Draws each line at (x, y + line_number*line_height_px).
    
    We measure text width via cv2.getTextSize. 
    """
    words = text.split()
    if not words:
        return y  # no text to draw

    font_face = cv2.FONT_HERSHEY_SIMPLEX
    line = ""
    for word in words:
        test_line = line + (" " if line else "") + word
        text_size, _ = cv2.getTextSize(test_line, font_face, font_scale, thickness)
        text_width = text_size[0]
        # If this line would exceed max_width, we draw the current line and start a new one
        if text_width > max_width_px and line != "":
            cv2.putText(frame, line, (x, y), font_face, font_scale, color, thickness)
            y += line_height_px
            line = word  # start new line with current word
        else:
            line = test_line

    # Draw any leftover line
    if line:
        cv2.putText(frame, line, (x, y), font_face, font_scale, color, thickness)
        y += line_height_px

    return y  # return the final y offset so you can chain text if needed

def call_vision_model_ollama(frame):
    """
    Sends the frame to Ollama's llama3.2-vision model (with local disk approach).
    Asks for a short, concise description. 
    """
    temp_path = "temp_vision_input.jpg"
    cv2.imwrite(temp_path, frame)

    # Add "Be concise" in the prompt
    response = ollama.chat(
        model="llama3.2-vision",
        messages=[
            {
                "role": "user",
                "content": "Describe this image in detail, but be very concise.",
                "images": [temp_path]
            }
        ],
        stream=False
    )

    # Example structure: response["message"]["content"]
    vision_text = response.get("message", {}).get("content", "")
    return vision_text.strip()

def call_llm_model_ollama(vision_text, bounding_boxes):
    """
    Combines vision_text and YOLO bounding boxes. 
    Asks a text model for a short, one-line commentary. 
    """
    if not bounding_boxes:
        bounding_text = "No objects detected"
    else:
        bounding_text_list = []
        for (lab, conf, x1, y1, x2, y2) in bounding_boxes:
            bounding_text_list.append(f"{lab} ({conf*100:.1f}% conf)")
        bounding_text = ", ".join(bounding_text_list)

    # Add "very concise" 
    prompt = (
        f"Vision model says (concise): {vision_text}\n\n"
        f"YOLO sees: {bounding_text}.\n\n"
        "Please provide a very concise single-sentence commentary about this scene."
    )

    response = ollama.chat(
        model="llama3.2",  # text model
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=False
    )
    text_output = response.get("message", {}).get("content", "")
    return text_output.strip()

def main():
    # 1) Load YOLO detection model
    model_path = "runs/train/custom_yolo/weights/best.pt"
    model = YOLO(model_path)

    # 2) Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    cv2.namedWindow("YOLO + Vision + LLM", cv2.WINDOW_NORMAL)

    FRAME_INTERVAL = 50
    frame_count = 0
    latest_commentary = "Initializing..."

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        # YOLO detection with conf=0.01 to show more boxes
        results = model.predict(frame, conf=0.01, device=0)

        bounding_boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2, conf, cls_id = box.data[0]
            cls_id = int(cls_id.item())
            label = model.names[cls_id]
            bounding_boxes.append((label, float(conf.item()), int(x1), int(y1), int(x2), int(y2)))

        # Draw YOLO bounding boxes
        for (lab, c, x1, y1, x2, y2) in bounding_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{lab} {c:.2f}", (x1, max(y1 - 10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Every FRAME_INTERVAL frames, call vision + text LLM
        if frame_count % FRAME_INTERVAL == 0:
            vision_text = call_vision_model_ollama(frame)
            commentary = call_llm_model_ollama(vision_text, bounding_boxes)
            latest_commentary = commentary

        # Word-wrap 'latest_commentary' in the top-left corner
        wrap_text_on_frame(frame, latest_commentary, x=20, y=30,
                           max_width_px=500, line_height_px=25,
                           font_scale=0.6, thickness=2,
                           color=(255,255,255))

        cv2.imshow("YOLO + Vision + LLM", frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
