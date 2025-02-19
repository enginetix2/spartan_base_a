import cv2
from ultralytics import YOLO

# 1) Load your best model
model = YOLO("./runs/train/custom_yolo/weights/best.pt")

# 2) Choose a test image that should contain "j-dog"
TEST_IMAGE = "./datasets/custom_dataset/images/val/img_j-dog_1739920901747.jpg"

# 3) Inference
results = model.predict(TEST_IMAGE, conf=0.25)
print(results)

# 4) Draw bounding boxes on the image
img = cv2.imread(TEST_IMAGE)
for box in results[0].boxes:
    x1, y1, x2, y2, conf, cls_id = box.data[0]
    cls_id = int(cls_id.item())
    label = model.names[cls_id]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(img, f"{label} {conf:.2f}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 2)

# 5) Show the image in a pop-up window (requires a local desktop environment)
cv2.imshow("Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
