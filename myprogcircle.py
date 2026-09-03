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

def update_text_field(text):
    global textArea
    textArea.insert(tk.END, text + "\n")
    textArea.see(tk.END)

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

def detect_circles_in_rect_realtime(img, rect, min_radius=10, max_radius=60):
    if rect is None or img is None:
        return []

    mask = create_mask_from_rect(img.shape, rect)
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    
    gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Поиск окружностей методом Хафа
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
                    
                    # Обработка результатов для OBB
                    if results[0].obb is not None and len(results[0].obb) > 0:
                        obb = results[0].obb
                        local_frame_count += 1
                        frame_count = local_frame_count
                        
                        # Выводим информацию каждые 3 кадра
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
                            
                            # Если активен поиск окружностей, выводим центры и углы для каждого объекта
                            if circle_detection_active:
                                for i, box in enumerate(obb):
                                    coords = box.xyxyxyxy[0].cpu().numpy()
                                    rect = cv2.minAreaRect(coords.astype(np.float32))
                                    rect_center = (int(rect[0][0]), int(rect[0][1]))
                                    
                                    # Выводим центр прямоугольника
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
                                # Если поиск окружностей не активен, выводим только центры
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
            
            # Дополнительно рисуем центры объектов
            if detection_results[0].obb is not None:
                obb = detection_results[0].obb
                for box in obb:
                    coords = box.xyxyxyxy[0].cpu().numpy()
                    center_x = int(np.mean(coords[:, 0]))
                    center_y = int(np.mean(coords[:, 1]))
                    
                    cv2.drawMarker(annotated_frame, (center_x, center_y), 
                                 (0, 0, 255), cv2.MARKER_CROSS, 20, 3)
                    
                    # Добавляем текст с координатами центра
                    cv2.putText(annotated_frame, f"({center_x}, {center_y})", 
                              (center_x + 15, center_y - 15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            return annotated_frame
        except Exception as e:
            print(f"Ошибка при отрисовке: {e}")
            return frame
    return frame

def process_circles(frame):
    """Обработка кадра для поиска окружностей"""
    global detection_results
    
    if not circle_detection_active or not detection_results:
        return frame
    
    try:
        if detection_results[0].obb is None or len(detection_results[0].obb) == 0:
            return frame
        
        obb = detection_results[0].obb
        frame_copy = frame.copy()
        
        # Для каждого обнаруженного OBB ищем окружности
        for box in obb:
            coords = box.xyxyxyxy[0].cpu().numpy()
            rect = cv2.minAreaRect(coords.astype(np.float32))
            rect_center = (int(rect[0][0]), int(rect[0][1]))

            circles = detect_circles_in_rect_realtime(frame_copy, rect, min_radius=5, max_radius=50)
            
            if circles:
                # Рисуем найденные окружности на кадре
                for (x, y, r) in circles:
                    cv2.circle(frame_copy, (x, y), r, (0, 255, 0), 3)
                    cv2.circle(frame_copy, (x, y), 2, (0, 0, 255), 3)
                    
                    cv2.line(frame_copy, rect_center, (x, y), (255, 0, 0), 2)
                    
                    # Вычисляем угол
                    angle_deg, angle_rad = calculate_angle(rect_center, (x, y))
                    
                    angle_text = f"{angle_rad:.1f}"
                    cv2.putText(frame_copy, angle_text, ((rect_center[0] + x)//2 - 20, (rect_center[1] + y)//2 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
        return frame_copy
        
    except Exception as e:
        print(f"Ошибка при поиске окружностей: {e}")
        return frame

def update_video():
    global current_frame
    
    ret, frame = cap.read()
    if ret:
        current_frame = frame.copy()
        
        # Если детекция активна, отрисовываем результаты
        if detection_active and detection_results:
            frame_display = draw_detections(frame)
        else:
            frame_display = frame
        
        # Если активен непрерывный поиск окружностей
        if circle_detection_active:
            frame_display = process_circles(frame_display)
        
        # Преобразуем в RGB для отображения в Tkinter
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

buttonDetect = tk.Button(root, text="OBB detect", command=start_detection)
buttonDetect.place(x=50, y=600)

buttonStopDetect = tk.Button(root, text="Stop detect", command=stop_detection_func)
buttonStopDetect.place(x=50, y=680)

buttonDetectCircle = tk.Button(root, text="Detect angle", command=find_angle)
buttonDetectCircle.place(x=200, y=600)

print("\n=== ПРОГРАММА ЗАПУЩЕНА ===")
print("Для выхода закройте окно")
print("============================\n")

update_video()

root.mainloop()

cap.release()
cv2.destroyAllWindows()