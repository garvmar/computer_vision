import cv2
from ultralytics import YOLO

model = YOLO('yolov12n-face.pt').to('cpu')
cap = cv2.VideoCapture(0) 

i = 0
while True:
    i += 1 
    
    ret, frame = cap.read()
    results = model.predict(frame)
    annotated_frame = results[0].plot()  
    #cv2.imwrite(f'{i}.png', annotated_frame)
    cv2.imshow("YOLOv12", annotated_frame)
    if cv2.waitKey(1) == 27:  
        break