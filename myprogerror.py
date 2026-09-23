import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import math  

model = YOLO('obb3000.pt').to('cpu')
print("Модель YOLO загружена")

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру")
    exit()

root = tk.Tk()
root.geometry("1200x800")
root.title("Detection")

current_frame = None
detection_active = False
detection_thread = None
stop_detection = False
detection_results = []
circle_detection_active = False  
frame_count = 0 
obb_angles_from_process = {}

def resize_for_display(img, max_size=700):
    height, width = img.shape[:2]
    if height > max_size or width > max_size:
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
        return cv2.resize(img, (new_width, new_height))
    return img

def create_mask_from_rect(img_shape, rect):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.fillPoly(mask, [box], 255)
    return mask

def detect_circles_in_rect_realtime(img, rect, min_radius=55, max_radius=60):
    if rect is None or img is None:
        return []

    mask = create_mask_from_rect(img.shape, rect)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )

    detected_circles = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        contour_rect = box.reshape((-1, 1, 2))
        
        for (x, y, r) in circles:
            point_inside = cv2.pointPolygonTest(contour_rect, (float(x), float(y)), False)
            if point_inside >= 0:
                detected_circles.append((x, y, r))
    
    return detected_circles

def calculate_angle(rect_center, circle_center):
    dx = circle_center[0] - rect_center[0]
    dy = -(circle_center[1] - rect_center[1])  
 
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    
    return angle_deg, angle_rad

def compute_obb_long_side_angle(box):
    try:
        coords = box.xyxyxyxy[0].cpu().numpy()
        points = coords.reshape(4, 2)

        side1 = np.linalg.norm(points[1] - points[0])
        side2 = np.linalg.norm(points[2] - points[1])

        sides = [side1, side2]
        max_side_index = np.argmax(sides)

        if max_side_index == 0:
            vector = points[1] - points[0]
        else:
            vector = points[2] - points[1]

        angle_rad = math.atan2(vector[1], vector[0])
        angle_deg = math.degrees(angle_rad)

        angle_deg = abs(angle_deg)
        if angle_deg > 90:
            angle_deg = 180 - angle_deg

        return angle_deg
    except Exception as e:
        print(f"Ошибка при вычислении угла OBB: {e}")
        return None

def show_error_angle():
    global detection_results, textError, obb_angles_from_process
    
    textError.delete("1.0", tk.END)
    
    if not detection_results:
        textError.insert(tk.END, "Нет результатов детекции. Запустите OBB detect сначала.\n")
        return
    
    try:
        if detection_results[0].obb is None or len(detection_results[0].obb) == 0:
            textError.insert(tk.END, "Объекты не обнаружены.\n")
            return
        
        obb = detection_results[0].obb
        textError.insert(tk.END, "="*40 + "\n")
        textError.insert(tk.END, "СРАВНЕНИЕ УГЛОВ OBB\n")
        textError.insert(tk.END, "="*40 + "\n\n")
        
        for i, box in enumerate(obb):

            angle_from_show = compute_obb_long_side_angle(box)
            angle_from_process = obb_angles_from_process.get(i, None)
            
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id] if cls_id in model.names else f"Class {cls_id}"
            
            msg = f"Объект {i+1} [{cls_name} {conf:.2%}]:\n"
            msg += f" Абсолютные значения углов от 0 до pi/2\n"

            if angle_from_show is not None:
                msg += f"  Угол OBB: {angle_from_show:.2f}°\n"
            else:
                msg += f"  Угол OBB: ошибка\n"
            
            if angle_from_process is not None:
                angle_from_process = abs(angle_from_process)
                if angle_from_process > 90:
                       angle_from_process = 180 - angle_from_process
                msg += f"  Угол вычисленный: {abs(angle_from_process):.2f}°\n"
            else:
                msg += f"  Угол вычисленный: не вычислен\n"
            
            # Абсолютная и относительная ошибк
            if angle_from_show is not None and angle_from_process is not None:
                abs_error = abs(angle_from_show - abs(angle_from_process))
                if angle_from_show != 0:
                    rel_error = (abs_error / angle_from_show) * 100.0
                else:
                    rel_error = 0.0 if abs_error == 0 else float('inf')
                
                msg += f"  Абсолютная ошибка: {abs_error:.2f}°\n"
                msg += f"  Относительная ошибка: {rel_error:.2f}%\n"
            else:
                msg += f"  Ошибка: недостаточно данных для сравнения\n"
            
            msg += "-"*40 + "\n"
            
            textError.insert(tk.END, msg)
            print(msg)
        
        textError.see(tk.END)
        
    except Exception as e:
        error_msg = f"Ошибка при вычислении угла: {e}\n"
        textError.insert(tk.END, error_msg)
        print(error_msg)

def run_detection():
    """Выполнение детекции в отдельном потоке"""
    global detection_active, stop_detection, detection_results, current_frame, frame_count
    
    print("\n=== ЗАПУЩЕНА OBB ДЕТЕКЦИЯ ===")
    detection_active = True
    stop_detection = False
    
    local_frame_count = 0
    
    while not stop_detection:
        if current_frame is not None:
            try:
                results = model(current_frame, verbose=False)
                
                if results and len(results) > 0:
                    detection_results = results
                    
                    if results[0].obb is not None and len(results[0].obb) > 0:
                        obb = results[0].obb
                        local_frame_count += 1
                        frame_count = local_frame_count
                        
                        if local_frame_count % 3 == 0:
                            if local_frame_count == 3:
                                textArea.delete("1.0", tk.END)
                                textArea.insert(tk.END, f"Кадр {local_frame_count} - Найдено объектов: {len(obb)}\n")
                                textArea.insert(tk.END, "="*40 + "\n")
                            else:
                                textArea.insert(tk.END, f"\nКадр {local_frame_count} - Найдено объектов: {len(obb)}\n")
                                textArea.insert(tk.END, "="*40 + "\n")
                            
                            textArea.see(tk.END)  
                            
                            print(f"\n{'='*40}")
                            print(f"Кадр {local_frame_count} - Найдено объектов: {len(obb)}")
                            print(f"{'-'*40}")
                            
                            if circle_detection_active:
                                for i, box in enumerate(obb):
                                    coords = box.xyxyxyxy[0].cpu().numpy()
                                    rect = cv2.minAreaRect(coords.astype(np.float32))
                                    rect_center = (int(rect[0][0]), int(rect[0][1]))
                                    
                                    cls_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    cls_name = model.names[cls_id] if cls_id in model.names else f"Class {cls_id}"
                                    
                                    msg_center = f"Объект {i+1} [{cls_name} {conf:.2%}]: Центр прямоугольника = ({rect_center[0]}, {rect_center[1]})"
                                    print(msg_center)
                                    textArea.insert(tk.END, msg_center + "\n")
                                    textArea.see(tk.END)
                                    
                                    circles = detect_circles_in_rect_realtime(current_frame, rect, min_radius=5, max_radius=50)
                                    
                                    if circles:
                                        for j, (x, y, r) in enumerate(circles):
                                            angle_deg, angle_rad = calculate_angle(rect_center, (x, y))
                                            msg_angle = f"  Окружность {j+1}: Центр = ({x}, {y}), Угол = {angle_deg:.2f}° ({angle_rad:.4f} рад)"
                                            print(msg_angle)
                                            textArea.insert(tk.END, msg_angle + "\n")
                                            textArea.see(tk.END)
                                    else:
                                        msg_no_circles = f"Окружности не найдены"
                                        print(msg_no_circles)
                                        textArea.insert(tk.END, msg_no_circles + "\n")
                                        textArea.see(tk.END)
                            else:
                                for i, box in enumerate(obb):
                                    coords = box.xyxyxyxy[0].cpu().numpy()
                                    center_x = int(np.mean(coords[:, 0]))
                                    center_y = int(np.mean(coords[:, 1]))
                                    
                                    cls_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    cls_name = model.names[cls_id] if cls_id in model.names else f"Class {cls_id}"
                                    
                                    msg = f"Объект {i+1} [{cls_name} {conf:.2%}]: Центр = ({center_x}, {center_y})"
                                    print(msg)
                                    textArea.insert(tk.END, msg + "\n")
                                    textArea.see(tk.END)
                                
                    else:
                        if local_frame_count % 10 == 0:
                            print("Объекты не обнаружены...")
                            if local_frame_count % 30 == 0:
                                textArea.insert(tk.END, "Объекты не обнаружены...\n")
                                textArea.see(tk.END) 
                    
            except Exception as e:
                textArea.insert(tk.END, "Ошибка при детекции \n")
                textArea.see(tk.END)  
        
        time.sleep(0.1)
    
    detection_active = False
    detection_results = []
    print("=== OBB ДЕТЕКЦИЯ ОСТАНОВЛЕНА ===\n")

def find_angle():
    """Запуск/остановка непрерывного поиска окружностей"""
    global circle_detection_active
    
    circle_detection_active = not circle_detection_active
    
    if circle_detection_active:
        print("Непрерывный поиск окружностей запущен")
        textArea.insert(tk.END, "Непрерывный поиск окружностей запущен\n")
        textArea.see(tk.END)
        buttonDetectCircle.config(text="Stop detect angle")
    else:
        print("Непрерывный поиск окружностей остановлен")
        textArea.insert(tk.END, "Непрерывный поиск окружностей остановлен\n")
        textArea.see(tk.END)
        buttonDetectCircle.config(text="Detect angle")

def start_detection():
    """Запуск детекции"""
    global detection_thread, stop_detection, detection_active, frame_count
    
    if detection_active:
        print("Детекция уже запущена")
        return
    
    if current_frame is None:
        print("Ошибка: Нет видео с камеры")
        return
    
    textArea.delete("1.0", tk.END)
    frame_count = 0
    
    detection_results = []
    stop_detection = False
    detection_thread = threading.Thread(target=run_detection, daemon=True)
    detection_thread.start()
    
    print("Детекция запущена...")
    textArea.insert(tk.END, "Детекция запущена...\n")
    textArea.see(tk.END)

def stop_detection_func():
    """Остановка детекции"""
    global stop_detection, detection_active, circle_detection_active
    
    if not detection_active:
        print("Детекция уже остановлена")
        return
    
    stop_detection = True
    detection_active = False
    circle_detection_active = False
    buttonDetectCircle.config(text="Detect angle")
    
    print("Остановка детекции...")
    textArea.insert(tk.END, "Остановка детекции...\n")
    textArea.see(tk.END)


def draw_detections(frame):
    """Отрисовка результатов OBB детекции на кадре"""
    if detection_results and len(detection_results) > 0:
        try:
            annotated_frame = detection_results[0].plot()
            
            if detection_results[0].obb is not None:
                obb = detection_results[0].obb
                for box in obb:
                    coords = box.xyxyxyxy[0].cpu().numpy()
                    center_x = int(np.mean(coords[:, 0]))
                    center_y = int(np.mean(coords[:, 1]))
                    
                    cv2.drawMarker(annotated_frame, (center_x, center_y), 
                                 (0, 0, 255), cv2.MARKER_CROSS, 20, 3)
                    
                    cv2.putText(annotated_frame, f"({center_x}, {center_y})", 
                              (center_x + 15, center_y - 15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            return annotated_frame
        except Exception as e:
            print(f"Ошибка при отрисовке: {e}")
            return frame
    return frame

def process_circles(frame):
    global detection_results, obb_angles_from_process
    
    if not circle_detection_active or not detection_results:
        return frame
    
    try:
        if detection_results[0].obb is None or len(detection_results[0].obb) == 0:
            return frame
        
        obb = detection_results[0].obb
        frame_copy = frame.copy()
        
        obb_angles_from_process = {}
        
        for i, box in enumerate(obb):
            coords = box.xyxyxyxy[0].cpu().numpy()
            rect = cv2.minAreaRect(coords.astype(np.float32))
            rect_center = (int(rect[0][0]), int(rect[0][1]))

            circles = detect_circles_in_rect_realtime(frame_copy, rect, min_radius=5, max_radius=50)
            
            if circles:
                for (x, y, r) in circles:
                    cv2.circle(frame_copy, (x, y), r, (0, 255, 0), 3)
                    cv2.circle(frame_copy, (x, y), 2, (0, 0, 255), 3)
                    
                    cv2.line(frame_copy, rect_center, (x, y), (255, 0, 0), 2)
                    
                    angle_deg, angle_rad = calculate_angle(rect_center, (x, y))
                    obb_angles_from_process[i] = angle_deg
                    
                    angle_text = f"{angle_rad:.1f}"
                    cv2.putText(frame_copy, angle_text, ((rect_center[0] + x)//2 - 20, (rect_center[1] + y)//2 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
        return frame_copy
        
    except Exception as e:
        print(f"Ошибка при поиске окружностей: {e}")
        return frame

def draw_camera_zero_and_horizon(frame):
    frame_copy = frame.copy()
    h, w = frame_copy.shape[:2]
    
    camera_zero = (w // 2, h // 2)
    
    cv2.circle(frame_copy, camera_zero, 6, (0, 255, 255), -1)
    cv2.circle(frame_copy, camera_zero, 8, (0, 0, 0), 2)
    
    cv2.line(frame_copy, (0, camera_zero[1]), (w, camera_zero[1]), (0, 255, 255), 2)
    cv2.line(frame_copy, (camera_zero[0], 0), (camera_zero[0], h), (0, 255, 255), 1)

    if detection_results and len(detection_results) > 0:
        try:
            if detection_results[0].obb is not None and len(detection_results[0].obb) > 0:
                obb = detection_results[0].obb
                for box in obb:
                    coords = box.xyxyxyxy[0].cpu().numpy()
                    rect = cv2.minAreaRect(coords.astype(np.float32))
                    cx, cy = int(rect[0][0]), int(rect[0][1])
                    
                    cv2.circle(frame_copy, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.line(frame_copy, (cx, cy), (w, cy), (0, 255, 0), 1)
                    cv2.line(frame_copy, (0, cy), (cx, cy), (0, 255, 0), 1)
                    
        except Exception as e:
            print(f"Ошибка при отрисовке мнимого нуля OBB: {e}")
    
    return frame_copy

def update_video():
    global current_frame
    
    ret, frame = cap.read()
    if ret:
        current_frame = frame.copy()
        
        if detection_active and detection_results:
            frame_display = draw_detections(frame)
        else:
            frame_display = frame
        
        if circle_detection_active:
            frame_display = process_circles(frame_display)

        frame_display = draw_camera_zero_and_horizon(frame_display) 
        
        frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        frame_resized = resize_for_display(frame_rgb)
        img_pil = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(img_pil)
        
        label_video.config(image=photo)
        label_video.image = photo
    
    root.after(30, update_video)

label_video = tk.Label(root)
label_video.pack(expand=True)
label_video.place(x=50, y=50)

labelTextArea = tk.Label(root, text="Detection Log")
labelTextArea.place(x=800, y=20)  
textArea = tk.Text(root, height=10, width=44)
textArea.place(x=800, y=50)

labelErrorArea = tk.Label(root, text="Углы наклона OBB")
labelErrorArea.place(x=800, y=300)  
textError = tk.Text(root, height=10, width=44)
textError.place(x=800, y=330)

buttonDetect = tk.Button(root, text="OBB detect", command=start_detection)
buttonDetect.place(x=50, y=600)

buttonStopDetect = tk.Button(root, text="Stop detect", command=stop_detection_func)
buttonStopDetect.place(x=50, y=680)

buttonDetectCircle = tk.Button(root, text="Detect angle", command=find_angle)
buttonDetectCircle.place(x=200, y=600)

buttonError = tk.Button(root, text="Show OBB angles", command=show_error_angle)
buttonError.place(x=200, y=680)

print("\n=== ПРОГРАММА ЗАПУЩЕНА ===")
print("Для выхода закройте окно")
print("============================\n")

update_video()

root.mainloop()

cap.release()
cv2.destroyAllWindows()