from ultralytics import YOLO

model = YOLO(r"C:\Users\rosej\Documents\Quickstart Project\yolov11_custom_segmentation\yolo11m-seg-quickstart.pt")

model.predict(source = r"C:\Users\rosej\Documents\Quickstart Project\Full LabPics Dataset\images\208.jpg", show = True, save = True, conf = 0.7, line_width = 2, save_crop=True, save_txt = True, show_labels = True, show_conf = True, classes = [0,1])