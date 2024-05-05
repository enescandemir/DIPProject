import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

import cv2
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from skimage.feature import canny
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class MainPage:
    def __init__(self, master):
        self.master = master
        self.master.title("Dijital Görüntü İşleme Ödev Platformu")

        self.label_header = tk.Label(self.master, text="Dijital Görüntü İşleme Ödev Platformu",
                                     font=("Helvetica", 16, "bold"))
        self.label_header.pack(pady=20)

        self.label_lesson_name = tk.Label(self.master, text="Ders Adı: Dijital Görüntü İşleme")
        self.label_lesson_name.pack()

        self.label_student_number = tk.Label(self.master, text="Öğrenci No: 221229064")
        self.label_student_number.pack()

        self.label_name = tk.Label(self.master, text="Adı Soyadı: Bilal Enes Candemir")
        self.label_name.pack()

        self.menu = tk.Menu(self.master)
        self.master.config(menu=self.menu)

        self.odev_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Ödevler", menu=self.odev_menu)

        self.odev_menu.add_command(label="Ödev 1: Temel İşlevselliği Oluştur", command=self.odev_1)
        self.odev_menu.add_command(label="Ödev 2: Filtre Uygulama", command=self.odev_2)
        self.odev_menu.add_command(label="Vize Ödevi", command=self.show_vize_options)

        self.image_label = None
        self.current_image = None

    def odev_1(self):
        self.clear_widgets()
        self.load_image()
        self.create_brightness_contrast_sliders(self.current_image)
        self.show_histogram_button()

    def odev_2(self):
        self.clear_widgets()
        self.load_image()
        self.show_image_processing_buttons()

    def show_vize_options(self):
        self.clear_widgets()

        s_curve_button = tk.Button(self.master, text="S-Curve", command=self.s_curve_option)
        s_curve_button.pack(pady=5)

        hough_transform_button = tk.Button(self.master, text="Hough Transform", command=self.hough_transform_option)
        hough_transform_button.pack(pady=5)

        deblurring_button = tk.Button(self.master, text="Deblurring", command=self.deblurring_option)
        deblurring_button.pack(pady=5)

        object_counting_button = tk.Button(self.master, text="Nesne Sayma", command=self.object_counting_option)
        object_counting_button.pack(pady=5)


    def load_image(self):
        file_path = filedialog.askopenfilename(title="Resim Seç", filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if file_path:
            image = Image.open(file_path)
            self.current_image = image
            photo = ImageTk.PhotoImage(image)

            if self.image_label:
                self.image_label.destroy()

            self.image_label = tk.Label(self.master, image=photo)
            self.image_label.image = photo
            self.image_label.pack()


    def create_brightness_contrast_sliders(self, image):
        brightness_scale = tk.Scale(self.master, from_=0.1, to=2.0, orient="horizontal", label="Parlaklık",
                                    resolution=0.1, command=lambda x: self.adjust_brightness_contrast(image, float(x)))
        brightness_scale.set(1.0)
        brightness_scale.pack(pady=5)

        contrast_scale = tk.Scale(self.master, from_=0.1, to=2.0, orient="horizontal", label="Kontrast", resolution=0.1,
                                  command=lambda x: self.adjust_brightness_contrast(image, float(x)))
        contrast_scale.set(1.0)
        contrast_scale.pack()

    def adjust_brightness_contrast(self, image, factor):
        adjusted_image = image.point(lambda p: p * factor)
        self.current_image = adjusted_image
        photo = ImageTk.PhotoImage(adjusted_image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

    def clear_widgets(self):
        for widget in self.master.winfo_children():
            if widget not in [self.label_header, self.label_lesson_name, self.label_student_number, self.label_name,
                              self.menu]:
                widget.destroy()


    def show_histogram_button(self):
        show_histogram_button = tk.Button(self.master, text="Histogramı Görüntüle", command=self.show_histogram)
        show_histogram_button.pack()

    def calculate_histogram(self, image):
        img_gray = image.convert("L")
        histogram_values, bins = np.histogram(img_gray.getdata(), bins=256, range=(0, 256), density=True)
        return histogram_values, bins

    def show_histogram(self):
        if self.current_image is not None:
            histogram_values, bins = self.calculate_histogram(self.current_image)

            histogram_window = tk.Toplevel(self.master)
            histogram_window.title("Histogram")

            figure = plt.Figure(figsize=(5, 4), dpi=100)
            subplot = figure.add_subplot(1, 1, 1)
            subplot.bar(bins[:-1], histogram_values, width=1, color='gray', alpha=0.75)
            subplot.set_title('Histogram')
            subplot.set_xlabel('Pixel Değeri')
            subplot.set_ylabel('Frekans')

            canvas = FigureCanvasTkAgg(figure, master=histogram_window)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def show_image_processing_buttons(self):
        button_frame = tk.Frame(self.master)
        button_frame.pack()

        button1 = tk.Button(button_frame, text="Görüntüyü Büyüt", command=self.resize_image)
        button1.pack(side=tk.LEFT, padx=5)

        button2 = tk.Button(button_frame, text="Görüntüyü Küçült", command=self.shrink_image)
        button2.pack(side=tk.LEFT, padx=5)

        button3 = tk.Button(button_frame, text="Zoom In", command=self.zoom_in)
        button3.pack(side=tk.LEFT, padx=5)

        button4 = tk.Button(button_frame, text="Zoom Out", command=self.zoom_out)
        button4.pack(side=tk.LEFT, padx=5)

        button5 = tk.Button(button_frame, text="Görüntüyü Döndür", command=self.rotate_image)
        button5.pack(side=tk.LEFT, padx=5)

        button6 = tk.Button(button_frame, text="Interpolasyon Yöntemleri", command=self.show_interpolation_options)
        button6.pack(side=tk.LEFT, padx=5)

    def resize_image(self):
        if self.current_image is not None:
            factor = simpledialog.askfloat("Büyütme Faktörü", "Büyütme faktörünü girin (örneğin, 1.2): ")
            if factor is not None:
                if factor <= 1:
                    messagebox.showerror("Hata", "Geçersiz giriş! Faktör 1'den büyük olmalıdır.")
                    return
                new_width = int(self.current_image.width * factor)
                new_height = int(self.current_image.height * factor)
                resized_image = self.current_image.resize((new_width, new_height))
                self.show_image(resized_image)

    def shrink_image(self):
        if self.current_image is not None:
            factor = simpledialog.askfloat("Küçültme Faktörü", "Küçültme faktörünü girin (örneğin, 0.8): ")
            if factor is not None:
                if factor >= 1:
                    messagebox.showerror("Hata", "Geçersiz giriş! Faktör 1'den küçük olmalıdır.")
                    return
                new_width = int(self.current_image.width * factor)
                new_height = int(self.current_image.height * factor)
                shrunk_image = self.current_image.resize((new_width, new_height))
                self.show_image(shrunk_image)

    def zoom_in(self):
        if self.current_image is not None:
            factor = simpledialog.askfloat("Yakınlaştırma Faktörü", "Yakınlaştırma faktörünü girin (örneğin, 1.2): ")
            if factor is not None:
                if factor <= 1:
                    messagebox.showerror("Hata", "Geçersiz giriş! Faktör 1'den büyük olmalıdır.")
                    return
                width, height = self.current_image.size
                new_width = int(width * factor)
                new_height = int(height * factor)
                new_image = Image.new(self.current_image.mode, (new_width, new_height))

                for x in range(new_width):
                    for y in range(new_height):
                        source_x = min(int(x / factor), width - 1)
                        source_y = min(int(y / factor), height - 1)
                        new_image.putpixel((x, y), self.current_image.getpixel((source_x, source_y)))

                self.show_image(new_image)

    def zoom_out(self):
        if self.current_image is not None:
            factor = simpledialog.askfloat("Uzaklaştırma Faktörü", "Uzaklaştırma faktörünü girin (örneğin, 0.8): ")
            if factor is not None:
                if factor >= 1:
                    messagebox.showerror("Hata", "Geçersiz giriş! Faktör 1'den küçük olmalıdır.")
                    return
                width, height = self.current_image.size
                new_width = int(width * factor)
                new_height = int(height * factor)
                new_image = Image.new(self.current_image.mode, (new_width, new_height))

                for x in range(new_width):
                    for y in range(new_height):
                        source_x = min(int(x / factor), width - 1)
                        source_y = min(int(y / factor), height - 1)
                        new_image.putpixel((x, y), self.current_image.getpixel((source_x, source_y)))

                self.show_image(new_image)

    def rotate_image(self):
        if self.current_image is not None:
            angle = simpledialog.askfloat("Döndürme Açısı", "Döndürme açısını girin (derece cinsinden): ")
            if angle is not None:
                rotated_image = self.current_image.rotate(angle)
                self.show_image(rotated_image)

    def show_interpolation_options(self):
        interpolation_window = tk.Toplevel(self.master)
        interpolation_window.title("İnterpolasyon Yöntemleri")

        bilinear_button = tk.Button(interpolation_window, text="Bilinear", command=self.apply_bilinear)
        bilinear_button.pack(pady=5)

        bicubic_button = tk.Button(interpolation_window, text="Bicubic", command=self.apply_bicubic)
        bicubic_button.pack(pady=5)

        average_button = tk.Button(interpolation_window, text="Average", command=self.apply_average)
        average_button.pack(pady=5)

    def apply_average(self):
        if self.current_image is not None:
            pil_image = np.array(self.current_image)
            height, width, _ = pil_image.shape
            new_height = height * 2
            new_width = width * 2

            new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
            for i in range(new_height):
                for j in range(new_width):
                    x = min(i // 2, height - 1)
                    y = min(j // 2, width - 1)
                    new_image[i, j] = pil_image[x, y]

            self.show_image(Image.fromarray(new_image))

    def apply_bilinear(self):
        if self.current_image is not None:
            new_image = self.current_image.resize((self.current_image.width*2, self.current_image.height*2), Image.BILINEAR)
            self.show_image(new_image)

    def apply_bicubic(self):
        if self.current_image is not None:
            new_image = self.current_image.resize((self.current_image.width*2, self.current_image.height*2), Image.BICUBIC)
            self.show_image(new_image)

    def s_curve_option(self):
            self.clear_widgets()

            button_frame = tk.Frame(self.master)
            button_frame.pack()

            standard_button = tk.Button(button_frame, text="Standart", command=self.apply_standard_s_curve)
            standard_button.pack(side=tk.LEFT, padx=5)

            horizontal_button = tk.Button(button_frame, text="Yatay", command=self.apply_horizontal_s_curve)
            horizontal_button.pack(side=tk.LEFT, padx=5)

            diagonal_button = tk.Button(button_frame, text="Eğimli", command=self.apply_diagonal_s_curve)
            diagonal_button.pack(side=tk.LEFT, padx=5)

            custom_button = tk.Button(button_frame, text="Özel", command=self.apply_custom_s_curve)
            custom_button.pack(side=tk.LEFT, padx=5)

            self.load_image()

    def apply_standard_s_curve(self):
        if self.current_image is not None:
            width, height = self.current_image.size
            standard_s_curve = Image.new(self.current_image.mode, (width, height))

            for x in range(width):
                for y in range(height):
                    r, g, b = self.current_image.getpixel((x, y))
                    new_r = int(255 * (1 / (1 + np.exp(-r / 255))))
                    new_g = int(255 * (1 / (1 + np.exp(-g / 255))))
                    new_b = int(255 * (1 / (1 + np.exp(-b / 255))))
                    standard_s_curve.putpixel((x, y), (new_r, new_g, new_b))

            self.show_image(standard_s_curve)

    def apply_horizontal_s_curve(self):
        if self.current_image is not None:
            width, height = self.current_image.size
            horizontal_s_curve = Image.new(self.current_image.mode, (width, height))

            for x in range(width):
                for y in range(height):
                    r, g, b = self.current_image.getpixel((x, y))
                    new_r = int(255 * (1 / (1 + np.exp(-(r - 128) / 255))))
                    new_g = int(255 * (1 / (1 + np.exp(-(g - 128) / 255))))
                    new_b = int(255 * (1 / (1 + np.exp(-(b - 128) / 255))))
                    horizontal_s_curve.putpixel((x, y), (new_r, new_g, new_b))

            self.show_image(horizontal_s_curve)

    def apply_diagonal_s_curve(self):
        if self.current_image is not None:
            width, height = self.current_image.size
            diagonal_s_curve = Image.new(self.current_image.mode, (width, height))

            for x in range(width):
                for y in range(height):
                    r, g, b = self.current_image.getpixel((x, y))
                    new_r = int(255 * (1 / (1 + np.exp(-((r + g + b) / 3 - 128) / 255))))
                    new_g = int(255 * (1 / (1 + np.exp(-((r + g + b) / 3 - 128) / 255))))
                    new_b = int(255 * (1 / (1 + np.exp(-((r + g + b) / 3 - 128) / 255))))
                    diagonal_s_curve.putpixel((x, y), (new_r, new_g, new_b))

            self.show_image(diagonal_s_curve)

    def apply_custom_s_curve(self):
        if self.current_image is not None:
            width, height = self.current_image.size
            custom_s_curve = Image.new(self.current_image.mode, (width, height))

            for x in range(width):
                for y in range(height):
                    r, g, b = self.current_image.getpixel((x, y))

                    new_r = int(255 * (1 / (1 + np.exp(-(r - 100) / 50))))
                    new_g = int(255 * (1 / (1 + np.exp(-(g - 150) / 60))))
                    new_b = int(255 * (1 / (1 + np.exp(-(b - 200) / 70))))

                    custom_s_curve.putpixel((x, y), (new_r, new_g, new_b))

            self.show_image(custom_s_curve)

    def hough_transform_option(self):
        self.clear_widgets()

        button_frame = tk.Frame(self.master)
        button_frame.pack()

        road_button = tk.Button(button_frame, text="Yol Çizgileri Tespiti", command=self.detect_lane_lines)
        road_button.pack(side=tk.LEFT, padx=5)

        face_button = tk.Button(button_frame, text="Göz Tespiti", command = self.detect_eye_regions)
        face_button.pack(side= tk.LEFT, padx=5)

        self.load_image()

    def detect_lane_lines(self):
        if self.current_image is not None:
            img_rgb = self.current_image.convert('RGB')
            img_rgb_array = np.array(img_rgb)

            img_gray = self.current_image.convert('L')
            img_gray_array = np.array(img_gray)

            edges = canny(img_gray_array, sigma=2)

            edges = edges.astype(np.uint8)

            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(img_rgb_array, (x1, y1), (x2, y2), (255, 0, 0), 2)

            self.show_image(Image.fromarray(img_rgb_array))

    def detect_eye_regions(self):
        if self.current_image is not None:
            img_gray = self.current_image.convert('L')
            img_gray_array = np.array(img_gray)

            blurred_img = cv2.GaussianBlur(img_gray_array, (5, 5), 0)

            circles = cv2.HoughCircles(blurred_img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10,
                                       maxRadius=50)

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                original_image_array = np.array(self.current_image)
                for (x, y, r) in circles:
                    cv2.circle(original_image_array, (x, y), r, (0, 255, 0), 2)

            self.show_image(Image.fromarray(original_image_array))

    def deblurring_option(self):
        self.clear_widgets()

        deblur_button = tk.Button(self.master, text="Deblurring Uygula", command=self.apply_deblurring)
        deblur_button.pack(pady=5)

        self.load_image()

    def apply_deblurring(self):
        if self.current_image is not None:
            deblurred_image = self.deblur_image(self.current_image)
            self.show_image(deblurred_image)

    def deblur_image(self, image):
        iterations = 200
        kernel_size = 3

        blurred_image = image.convert("L")
        blurred_array = np.array(blurred_image)

        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)

        for _ in range(iterations):
            blurred_estimate = convolve2d(blurred_array, kernel, mode='same', boundary='wrap')

            error = blurred_array / (blurred_estimate + 1e-10)

            blurred_estimate *= convolve2d(error, kernel[::-1, ::-1], mode='same', boundary='wrap')

        deblurred_array = np.clip(blurred_estimate, 0, 255).astype(np.uint8)
        deblurred_image = Image.fromarray(deblurred_array)

        return deblurred_image


    def object_counting_option(self):
        self.clear_widgets()

        button_frame = tk.Frame(self.master)
        button_frame.pack()

        detection_button = tk.Button(button_frame, text="Nesne Say", command=self.count_objects)
        detection_button.pack(side=tk.LEFT, padx=5)

        self.load_image()


    def count_objects(self):
        if self.current_image is not None:
            image_array = np.array(self.current_image)

            hsv_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)

            lower_dark_green = np.array([30, 50, 50])
            upper_dark_green = np.array([70, 255, 255])

            mask = cv2.inRange(hsv_image, lower_dark_green, upper_dark_green)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contour_properties = []

            count = 1

            for contour in contours:
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    area = M["m00"]
                else:
                    cX, cY = 0, 0
                    area = 0

                x, y, w, h = cv2.boundingRect(contour)
                rect_width = w
                rect_height = h
                rect_diagonal = np.sqrt(rect_width ** 2 + rect_height ** 2)

                mean_color = np.mean(image_array[y:y + h, x:x + w], axis=(0, 1))
                median_color = np.median(image_array[y:y + h, x:x + w], axis=(0, 1))

                energy = np.sum(np.square(image_array[y:y + h, x:x + w]))
                entropy = -np.sum(
                    image_array[y:y + h, x:x + w] * np.log2(image_array[y:y + h, x:x + w] + 1e-8))

                contour_properties.append(
                    [count, cX, cY, w, h, rect_diagonal, energy, entropy,
                     mean_color[0], median_color[0]])

                count += 1

            df = pd.DataFrame(contour_properties,
                              columns=['No', 'Center', 'Length', 'Width', 'Diagonal', 'Energy', 'Entropy', 'Mean',
                                       'Median', 'Additional Column'])

            df.drop(columns=['Additional Column'], inplace=True)

            formatted_df = df.style.format({
                'Center': '({:.0f}, {:.0f})'.format,
                'Length': '{{:.0f}}'.format,
                'Width': '{{:.0f}}'.format,
                'Diagonal': '{{:.0f}}'.format,
                'Energy': '{{:.2f}}'.format,
                'Entropy': '{{:.2f}}'.format,
                'Mean': '{{:.2f}}'.format,
                'Median': '{{:.2f}}'.format
            })

            formatted_df.to_excel('dark_green_regions.xlsx', index=False)

            result = cv2.bitwise_and(image_array, image_array, mask=mask)
            self.show_image(Image.fromarray(result))

    def show_image(self, image):
        if self.image_label:
            self.image_label.destroy()

        photo = ImageTk.PhotoImage(image)
        self.image_label = tk.Label(self.master, image=photo)
        self.image_label.image = photo
        self.image_label.pack()



def main():
    root = tk.Tk()
    app = MainPage(root)
    root.mainloop()


if __name__ == "__main__":
    main()
