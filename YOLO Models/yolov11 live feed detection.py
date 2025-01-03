#yolov11 live feed test
    #Important: use 'q' to quit!

from ultralytics import YOLO
import cv2

# Load the YOLOv11 model
model = YOLO("yolo11n.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is accessible
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Loop to process frames from the webcam
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Perform object detection on the current frame
    results = model.predict(frame)

    # Visualize the detection results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv11 Webcam", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
