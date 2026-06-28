import cv2
from ultralytics import YOLO

model = YOLO('best.pt').to('cpu')

image_path = "test_image2.jpg"  

# Распознать объекты
results = model(image_path)

results[0].show()