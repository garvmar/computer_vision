import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time


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

def run_detection():
    """Выполнение детекции в отдельном потоке"""
    global detection_active, stop_detection, detection_results, current_frame
    
    print("\n=== ЗАПУЩЕНА OBB ДЕТЕКЦИЯ ===")
    detection_active = True
    stop_detection = False
    
    frame_count = 0
    
    while not stop_detection:
        if current_frame is not None:
            try:
                results = model(current_frame, verbose=False)
                
                if results and len(results) > 0:
                    detection_results = results
                    
                    # Обработка результатов для OBB
                    if results[0].obb is not None and len(results[0].obb) > 0:
                        obb = results[0].obb
                        frame_count += 1
                        
                        # Выводим информацию каждые 3 кадра
                        if frame_count % 3 == 0:
                            print(f"\n{'='*40}")
                            print(f"Кадр {frame_count} - Найдено объектов: {len(obb)}")
                            print(f"{'='*40}")
                            
                            for i, box in enumerate(obb):
                                # Получаем координаты OBB (4 точки)
                                coords = box.xyxyxyxy[0].cpu().numpy()
                                
                                # Вычисляем центр прямоугольника
                                center_x = int(np.mean(coords[:, 0]))
                                center_y = int(np.mean(coords[:, 1]))
                                
                                # Получаем класс и уверенность
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                cls_name = model.names[cls_id] if cls_id in model.names else f"Class {cls_id}"
                                

                                print(f"Объект {i+1} [{cls_name} {conf:.2%}]: Центр = ({center_x}, {center_y})")
                                
                    else:
                        if frame_count % 10 == 0:
                            print("Объекты не обнаружены...")
                    
            except Exception as e:
                print(f"Ошибка при детекции: {e}")
        
        time.sleep(0.1)
    
    detection_active = False
    detection_results = []
    print("=== OBB ДЕТЕКЦИЯ ОСТАНОВЛЕНА ===\n")

def start_detection():
    """Запуск детекции"""
    global detection_thread, stop_detection, detection_active
    
    if detection_active:
        print("Детекция уже запущена")
        return
    
    if current_frame is None:
        print("Ошибка: Нет видео с камеры")
        return
    
    detection_results = []
    stop_detection = False
    detection_thread = threading.Thread(target=run_detection, daemon=True)
    detection_thread.start()
    
    print("OBB детекция запущена. Ожидайте обнаружения объектов...")

def stop_detection_func():
    """Остановка детекции"""
    global stop_detection, detection_active
    
    if not detection_active:
        print("Детекция уже остановлена")
        return
    
    stop_detection = True
    detection_active = False
    print("Остановка детекции...")

def draw_detections(frame):
    """Отрисовка результатов OBB детекции на кадре"""
    if detection_results and len(detection_results) > 0:
        try:
            # Используем встроенную отрисовку YOLO
            annotated_frame = detection_results[0].plot()
            
            # Дополнительно рисуем центры объектов
            if detection_results[0].obb is not None:
                obb = detection_results[0].obb
                for box in obb:
                    # Получаем координаты OBB
                    coords = box.xyxyxyxy[0].cpu().numpy()
                    center_x = int(np.mean(coords[:, 0]))
                    center_y = int(np.mean(coords[:, 1]))
                    
                    # Рисуем центр красным крестиком
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

buttonDetect = tk.Button(root, text="OBB detect", command=start_detection)
buttonDetect.place(x=50, y=600)

buttonStopDetect = tk.Button(root, text="Stop detect", command=stop_detection_func)
buttonStopDetect.place(x=50, y=680)

print("\n=== ПРОГРАММА ЗАПУЩЕНА ===")
print("Для выхода закройте окно")
print("============================\n")

update_video()

root.mainloop()

cap.release()
cv2.destroyAllWindows()