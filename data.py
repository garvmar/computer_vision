import cv2
from ultralytics import YOLO

model = YOLO('obb3000.pt').to('cpu')

image_path = "test_image3.jpg"  

# Распознать объекты
results = model(image_path)

results[0].show()