import cv2
import tkinter as tk
from tkinter import ttk, filedialog
import PIL.Image, PIL.ImageTk
from predict_emotion import predict_emotion
import csv
from datetime import datetime
import os

LOG_FILE = "emotion_log.csv"

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện cảm xúc")
        self.root.geometry("1200x720")
        self.root.configure(bg="#dbe9ff")

        style = ttk.Style()
        style.configure("TButton",
                        font=("Segoe UI", 13),
                        padding=10)
        style.map("TButton",background=[("active", "#4da3ff")])

        main_frame = tk.Frame(root, bg="#dbe9ff")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)

        left_frame = tk.Frame(main_frame, bg="white", bd=0, relief="flat")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 20))

        # Khung bo góc viền webcam
        self.video_border = tk.Frame(left_frame, bg="white")
        self.video_border.pack(expand=True, fill="both", padx=10, pady=10)

        self.video_label = tk.Label(self.video_border, bg="black")
        self.video_label.pack(expand=True)

        right_frame = tk.Frame(main_frame, bg="#e6f2ff", width=350, bd=2, relief="ridge")
        right_frame.pack(side="right", fill="y")

        tk.Label(right_frame, text="MENU ",
                 bg="#e6f2ff", font=("Segoe UI", 18, "bold"),
                 fg="#003d80").pack(pady=15)

        # Nút bấm
        ttk.Button(right_frame, text="ảnh từ folder",
                   command=self.select_image).pack(pady=10, ipadx=10)
        ttk.Button(right_frame, text="video từ folder",
                   command=self.select_video).pack(pady=10, ipadx=10)
        ttk.Button(right_frame, text="webcam",
                   command=self.start_detection).pack(pady=10, ipadx=10)
        ttk.Button(right_frame, text="Thoát",
                   command=self.stop_detection).pack(pady=10, ipadx=10)

        #kết quả
        self.label_status = tk.Label(right_frame, text="",font=("Segoe UI", 12),bg="#e6f2ff", fg="#0059b3")
        self.label_status.pack(pady=15)

        self.label_result = tk.Label(right_frame, text="", wraplength=260,font=("Segoe UI", 13, "bold"), bg="#e6f2ff", fg="#008000")
        self.label_result.pack(pady=10)

        #bến
        self.running = False
        self.cap = None
        #tạo file csv
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["time", "label", "confidence"])

    def log_emotion(self, emotion, confidence):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([now, emotion, f"{float(confidence) * 100:.1f}"])

    #webcam
    def start_detection(self):
        if self.running:
            return

        self.running = True
        self.label_status.config(text="mở webcam")

        self.cap = cv2.VideoCapture(0)
        self.detect_loop()

    def detect_loop(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret:
            emotion, frame, confidence = predict_emotion(frame)
            confidence_value = float(confidence)  # ép kiểu float

            self.log_emotion(emotion, confidence_value)

            # Hiển thị nhãn và xác suất
            self.label_result.config(text=f"Cảm xúc: {emotion} ({confidence_value * 100:.1f}%)")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(rgb)
            imgtk = PIL.ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(15, self.detect_loop)
    #dung
    def stop_detection(self):
        if self.cap:
            self.cap.release()

        self.running = False
        self.video_label.configure(image="")
        self.label_status.config(text="Dừng nhận diện")

        # Gọi analyze.py
        try:
            from analyze import analyze_emotion_log
            summary, message = analyze_emotion_log()

            # In lên giao diện
            self.label_result.config(text=f"{summary}\n\n{message}")
        except Exception as e:
            self.label_result.config(text=f"Lỗi: {e}")

    #ảnh
    def select_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if not path:
            return

        frame = cv2.imread(path)
        emotion, frame, confidence = predict_emotion(frame)
        confidence_value = float(confidence)  # ép kiểu float

        self.log_emotion(emotion, confidence_value)

        # Hiển thị đồng thời nhãn và xác suất
        self.label_result.config(text=f"Cảm xúc: {emotion} ({confidence_value * 100:.1f}%)")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(rgb)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def select_video(self):
        path = filedialog.askopenfilename(
            filetypes=[("Video Files", "*.mp4 *.avi")])
        if not path:
            return
        cap = cv2.VideoCapture(path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            emotion, frame, confidence = predict_emotion(frame)
            confidence_value = float(confidence)  # ép kiểu float

            self.log_emotion(emotion, confidence_value)

            # Hiển thị đồng thời nhãn và xác suất
            self.label_result.config(text=f"Cảm xúc: {emotion} ({confidence_value * 100:.1f}%)")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(rgb)
            imgtk = PIL.ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            self.root.update()

        cap.release()
        self.label_status.config(text="Video đã phát xong")


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
