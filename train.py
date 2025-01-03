from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")

model.train(data= r"C:\Users\rosej\Documents\Quickstart Project\yolov11_custom_segmentation\custom_dataset.yml", imgsz=640, device=0, batch = 8, epochs = 100, workers=0)