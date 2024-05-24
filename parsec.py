import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

def select_image():
    path = filedialog.askopenfilename()
    if path:
        process_image(path)

def process_image(image_path):
    # Загружаем изображение
    image = cv2.imread(image_path)
    # Конвертируем в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Используем каскад Хаара для обнаружения лиц на изображении
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        print("Лица не обнаружены")
        return

    # Выделяем прямоугольник вокруг лица с учетом плеч и макушки
    (x, y, w, h) = faces[0]
    vertical_padding = int(0.3 * h)
    horizontal_padding = int(0.1 * w)
    face_img = image[max(0, y - vertical_padding):min(y + h + vertical_padding, image.shape[0]), 
                     max(0, x - horizontal_padding):min(x + w + horizontal_padding, image.shape[1])]

    # Применяем алгоритм GrabCut для сегментации лица
    mask = np.zeros(face_img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (10, 10, face_img.shape[1] - 10, face_img.shape[0] - 10)
    cv2.grabCut(face_img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Создаем маску, где лицо обозначено как передний план (1) и фон как задний план (0)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Обрезаем фон
    face_img = face_img * mask2[:, :, np.newaxis]

    # Создаем белый фон
    white_background = np.ones_like(face_img) * 255

    # Заменяем фон на белый цвет
    result = np.where(face_img == 0, white_background, face_img)

    # Сохраняем результат
    cv2.imwrite("face_cropped_image.jpg", result)

    # Обновляем изображение в интерфейсе
    display_image("face_cropped_image.jpg")

def display_image(image_path):
    image = Image.open(image_path)
    image.thumbnail((300, 300))
    photo = ImageTk.PhotoImage(image)
    label.config(image=photo)
    label.image = photo

# Создаем графический интерфейс
root = tk.Tk()
root.title("Обрезка изображения с лицом")

# Кнопка для выбора изображения
btn = tk.Button(root, text="Выбрать изображение", command=select_image)
btn.pack(side="top", fill="x", pady=10)

# Метка для отображения изображения
label = tk.Label(root)
label.pack(side="bottom", padx=10, pady=10)

root.mainloop()
