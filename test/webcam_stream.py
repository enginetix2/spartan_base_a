import cv2
from ultralytics import YOLO

# Test a streaming version of the YOLO model

def main():
    model = YOLO("./runs/train/custom_yolo/weights/best.pt")  # detection model
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Lower the conf threshold drastically
        results = model.predict(frame, conf=0.01, device=0)

        for box in results[0].boxes:
            x1, y1, x2, y2, conf, cls_id = box.data[0]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls_id = int(cls_id.item())
            label = model.names[cls_id]
            confidence = float(conf.item())

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}",
                        (x1, max(y1-10, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,255,0), 2)

        cv2.imshow("YOLO Live", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
