import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np

cap = cv2.VideoCapture(2)

if not cap.isOpened():
    print("Ошибка: Не удалось открыть камеру")
    exit()

root = tk.Tk()
root.geometry("800x600")


current_frame = None


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

def update_video():
    global current_frame
    
    ret, frame = cap.read()
    if ret:
        current_frame = frame.copy()
        
        # Преобразуем в RGB для отображения в Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = resize_for_display(frame_rgb)
        img_pil = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(img_pil)
        
        label_video.config(image=photo)
        label_video.image = photo
    
    root.after(30, update_video)


label_video = tk.Label(root)
label_video.pack(expand=True)


print("\n=== ПРОГРАММА ЗАПУЩЕНА ===")
print("Для выхода закройте окно")
print("============================\n")

update_video()

root.mainloop()

cap.release()
cv2.destroyAllWindows()