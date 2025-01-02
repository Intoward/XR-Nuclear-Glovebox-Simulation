from ultralytics import YOLO
import cv2
import os

# Load the YOLOv11 segmentation model
model = YOLO("yolo11s-seg.pt")

# Path to the input video
video_path = r"C:\Users\rosej\Documents\Quickstart Project\yolo input\prison_pakour.mp4"  # !!Replace with source path to video and user name!!

# Path to save the output video
output_path = r"C:\Users\rosej\Documents\Quickstart Project\yolo output\video output\seg_output.mp4" # !!Replace with output path and user name!!

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

    # Perform segmentation on the current frame
    results = model.predict(frame, task="segment")  # Use `task="segment"` for segmentation

    # Annotate the frame with segmentation results
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Optionally display the frame (comment this section if not needed)
    cv2.imshow("YOLOv11 Segmentation Video Processing", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video processing complete. Output saved at: {output_path}")
