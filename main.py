import torch
import cv2
import json
import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO #YOLOv11

# Using Python 3.9 and PyTroch 2.5

#1. YoloV11 Object Detection and Segmentation

# This YOLOv11 script uses object detection and segmentation to process a video file.
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

#2.1 Train Custom Model on LabPics Dataset

model = YOLO("yolo11m-seg.pt")

results = model.train(data= r"C:\Users\rosej\Documents\Quickstart Project\yolov11_custom_segmentation\custom_dataset.yml", imgsz=640, device=0, batch = 8, epochs = 100, workers=0)

#Predict on a test image

model = YOLO(r"C:\Users\rosej\Documents\Quickstart Project\yolov11_custom_segmentation\yolo11m-seg-quickstart.pt") # Load custom model ("yolo11m-seg-quickstart.pt")

model.predict(source = r"C:\Users\rosej\Documents\Quickstart Project\Full LabPics Dataset\images\208.jpg", show = True, save = True, conf = 0.7, line_width = 2, save_crop=True, save_txt = True, show_labels = True, show_conf = True, classes = [0,1,2])

model.export(format="onnx") # Export model to ONNX format

#2.2 Point Cloud Generation

def depth_to_point_cloud(rgb_image, depth_image, fx, fy, cx, cy, depth_scale=1000.0):
    # Create RGBD image
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)),
        o3d.geometry.Image(depth_image.astype(np.float32) / depth_scale)
    )
    
    # Define camera intrinsics
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=depth_image.shape[1],
        height=depth_image.shape[0],
        fx=fx, fy=fy, cx=cx, cy=cy
    )
    
    # Convert to point cloud
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    return pcd

# Usage with YCB data:
rgb_path = "path/to/ycb/rgb/000001.jpg"
depth_path = "path/to/ycb/depth/000001.png"

rgb = cv2.imread(rgb_path)
depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

# YCB camera intrinsics
fx = 1066.778
fy = 1067.487
cx = 312.9869
cy = 241.3109

pcd = depth_to_point_cloud(rgb, depth, fx, fy, cx, cy)

#2.2 3D Model from Point Cloud

# Load and preprocess point cloud data
def load_point_cloud(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    pcd = o3d.io.read_point_cloud(file_path)
    if pcd.is_empty():
        raise ValueError(f"The point cloud file {file_path} is empty or invalid.")
    return pcd

def preprocess_point_cloud(pcd, voxel_size=0.01):
    # Downsample the point cloud for faster processing
    down_pcd = pcd.voxel_down_sample(voxel_size)
    # Estimate normals for Poisson reconstruction
    down_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=voxel_size * 2, max_nn=30))
    return down_pcd

def segment_objects(pcd, eps=0.02, min_points=10):
    # Perform DBSCAN clustering
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    unique_labels = set(labels)
    segmented_objects = []

    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        indices = np.where(labels == label)[0]
        object_points = pcd.select_by_index(indices)
        segmented_objects.append(object_points)
    
    print(f"Segmented into {len(segmented_objects)} objects.")
    return segmented_objects

def reconstruct_3d_model(segmented_objects):
    models = []
    for obj in segmented_objects:
        try:
            # Poisson surface reconstruction
            obj.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                obj, depth=8)
            
            # Filter low-density vertices
            if densities is not None and len(densities) > 0:
                densities = np.asarray(densities)
                vertices_to_remove = densities < np.quantile(densities, 0.01)
                mesh.remove_vertices_by_mask(vertices_to_remove)
            
            models.append(mesh)
        except Exception as e:
            print(f"Failed to reconstruct model for an object: {e}")
    
    print(f"Reconstructed {len(models)} 3D models.")
    return models

def visualize_point_cloud(pcd):
    o3d.visualization.draw_geometries([pcd], window_name="Point Cloud Visualization")

def visualize_models(models):
    # Assign random colors to each model for easier visualization
    for model in models:
        model.paint_uniform_color(np.random.rand(3))
    o3d.visualization.draw_geometries(models, window_name="3D Model Visualization")

def main():
    file_path = r"C:\Users\rosej\Documents\Quickstart Project\Test files\bunny.pcd"  # Replace with your file path
    try:
        # Load and preprocess the point cloud
        pcd = load_point_cloud(file_path)
        visualize_point_cloud(pcd)  # Optional step to visualize the original point cloud
        preprocessed_pcd = preprocess_point_cloud(pcd)
        objects = segment_objects(preprocessed_pcd, eps=0.02, min_points=10)
        models = reconstruct_3d_model(objects)
        if models:
            visualize_models(models)
        else:
            print("No 3D models generated.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()