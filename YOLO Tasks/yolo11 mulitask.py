from ultralytics import YOLO
import cv2
import os

# This YOLOv11 script uses object detection, segmentation, and classification models to process a video file.
# This script is much more cpu/gpu intensive and is not reccommended to use on a live feed or webcam.

# Load the YOLOv11 models for detection and segmentation
detection_model = YOLO("yolo11n.pt")  # Object detection model
segmentation_model = YOLO("yolo11s-seg.pt")  # Segmentation model

# Path to the input video
video_path = r"C:\Users\rosej\Documents\Quickstart Project\yolo input\prison_pakour.mp4"  # !!Replace with source path to video and user name!!

# Path to save the output video
output_path = r"C:\Users\rosej\Documents\Quickstart Project\yolo output\video output\multi_output_video.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file is accessible
if not cap.isOpened():
    print(f"Error: Could not open video file {video_path}.")
    exit()

# Get the video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create the VideoWriter object
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the output directory exists
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process each frame from the video
while True:
    ret, frame = cap.read()
    if not ret:
        print("Processing complete.")
        break

    # Perform object detection
    detection_results = detection_model.predict(frame)
    detection_annotated_frame = detection_results[0].plot()

    # Perform segmentation
    segmentation_results = segmentation_model.predict(frame, task="segment")
    segmentation_annotated_frame = segmentation_results[0].plot()

    # Combine the detection and segmentation outputs
    # Start with the segmentation results and overlay detection results on top
    combined_frame = segmentation_annotated_frame.copy()
    combined_frame = cv2.addWeighted(combined_frame, 0.7, detection_annotated_frame, 0.3, 0)

    # Write the combined frame to the output video
    out.write(combined_frame)

    # Optionally display the frame (comment this section if not needed)
    cv2.imshow("YOLOv11 Combined Video Processing", combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video processing complete. Output saved at: {output_path}")

