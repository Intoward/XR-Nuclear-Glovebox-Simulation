# XR-Nuclear-Glovebox-Simulation
Project Overview
This repository demonstrates an innovative AI-enhanced VR simulation pipeline designed for nuclear research and training applications. Developed to align with the Oak Ridge Enhanced Technology and Training Center's (ORETTC) mission, this project focuses on advancing experimental setups, material interaction modeling, and complex data visualization within nuclear glovebox environments.

The project utilizes advanced computer vision techniques, including object detection and segmentation with YOLOv11, depth data integration for point cloud generation, and 3D modeling from segmented objects. The pipeline supports applications such as XR haptic glove training and theoretical research in radiation shielding and thermal analysis.

Key Features
Custom YOLOv11 Models:

Object detection with yolo11n.pt.
Segmentation with yolo11s-seg.pt.
Model trained using the LabPics dataset for laboratory settings: yolo11m-set-quickstart.pt.

Depth Data Integration:
Converts RGB-D images into detailed 3D point clouds using Open3D.

3D Model Generation:
Generates semi-accurate 3D models from point clouds via Poisson surface reconstruction.
Segments objects using clustering algorithms (e.g., DBSCAN).

Proof-of-Concept Metadata Integration:
Assigns attributes (e.g., weight, material properties) to 3D models for XR simulation.

Applications
- Simulating Experimental Setups.
- Modeling Material Interactions.
- Visualizing Complex Data (thermal gradients).

Repository Structure

├── yolov11_custom_segmentation/         # Custom YOLOv11 training scripts
│   ├── train.py                         # Script for training YOLOv11 models
│   ├── predict.py                       # Script for model inference
├── point_cloud_processing/              # Depth data and point cloud scripts
│   ├── depth_to_point_cloud.py          # Converts depth images to point clouds
│   ├── point_cloud_to_3d_model.py       # Generates 3D models from point clouds
├── data/                                # Sample datasets and test files
├── main.py                              # Integrated project script
└── README.md                            # Project documentation

Installation Instructions

Step 1: Create a Virtual Environment
Using Anaconda, create and activate the glovebox_pipeline environment:

conda create -n glovebox_pipeline python=3.9 -y
conda activate glovebox_pipeline

Step 2: Install Dependencies
Install PyTorch and additional libraries:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics open3d numpy opencv-python matplotlib

Step 3: Clone the Repository

git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
pip install ultralytics

Step 4: Verify GPU Availability

python
>>> import torch
>>> torch.cuda.is_available()  # Should return True for GPU support

How to Use
1. Training YOLOv11 Custom Models
Train YOLOv11 models on the LabPics dataset:
python yolov11_custom_segmentation/train.py

2. Running Object Detection and Segmentation
Perform object detection or segmentation on test images:
python yolov11_custom_segmentation/predict.py

3. Depth to Point Cloud Conversion
Convert RGB-D images into a point cloud:
python point_cloud_processing/depth_to_point_cloud.py

4. 3D Model Generation
Generate 3D models from point clouds:
python point_cloud_processing/point_cloud_to_3d_model.py


Acknowledgments
This project was part of an internship inspired by the DoE's and ORETTC's mission to advance nuclear security and training. Special thanks to the creators of the LabPics and YCB-Ev datasets, as well as the developers of YOLO and Open3D libraries.
