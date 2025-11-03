import cv2
import tkinter as tk
from tkinter import ttk
from predict_emotion import predict_emotion
import pandas as pd
import time
import PIL.Image, PIL.ImageTk

class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhận diện cảm xúc")
        self.root.geometry("800x700")

        #Thời gian chạy
        tk.Label(root, text="Thời gian chạy", font=("Arial", 16, "bold")).pack(pady=5)
        self.duration_var = tk.IntVar(value=15)
        ttk.Combobox(root, textvariable=self.duration_var, values=[10, 15, 20, 25, 30], font=("Arial", 14)).pack()

        #Chụp biẻu cảm
        tk.Label(root, text="Chụp", font=("Arial", 16, "bold")).pack(pady=5)
        self.interval_var = tk.DoubleVar(value=1.0)
        ttk.Combobox(root, textvariable=self.interval_var, values=[ 1.0, 1.5,  2.0], font=("Arial", 14)).pack()

        #Nút bắt đầu và thoát
        tk.Button(root, text="Bắt đầu nhận diện", font=("Arial", 14), command=self.start_detection).pack(pady=10)
        tk.Button(root, text="Thoát", font=("Arial", 14), command=self.stop_detection).pack(pady=5)

        #Label trạng thái
        self.label_status = tk.Label(root, text="", fg="blue", font=("Arial", 14))
        self.label_status.pack(pady=5)

        #Video display
        self.video_label = tk.Label(root)
        self.video_label.pack()

        #Label kết quả
        self.label_result = tk.Label(root, text="", fg="green", font=("Arial", 12, "bold"), wraplength=680, justify="center")
        self.label_result.pack(pady=10)


        #Dữ liệu lưu cảm xúc
        self.emotions_recorded = []
        self.running = False
        self.start_time = None
        self.cap = None
        self.last_capture_time = 0

    #Nhận diện
    def start_detection(self):
        if self.running:
            return
        self.running = True
        self.emotions_recorded.clear()
        self.label_status.config(text="Đang mở webcam")
        self.cap = cv2.VideoCapture(0)
        self.start_time = time.time()
        self.duration = self.duration_var.get()
        self.interval = self.interval_var.get()
        self.last_capture_time = 0
        self.root.after(100, self.detect_loop)

    #Dự đoán
    def detect_loop(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if ret and frame is not None:
            current_time = time.time() - self.start_time

            # Chụp theo interval
            if current_time - self.last_capture_time >= self.interval:
                emotion, frame, _ = predict_emotion(frame)
                self.emotions_recorded.append({'time': round(current_time, 2), 'label': emotion})
                self.last_capture_time = current_time

            # Hiển thị video
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame_rgb)
            imgtk = PIL.ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            # Update status
            self.label_status.config(text=f"Đang nhận diện... {int(current_time)} / {self.duration}s")

        # Dừng khi đủ duration
        if time.time() - self.start_time >= self.duration:
            self.stop_detection()
        else:
            self.root.after(10, self.detect_loop)  # chạy liên tục

    def stop_detection(self):
        if not self.running:
            return
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.configure(image="")
        self.label_status.config(text="Đã nhận diện")

        #Lưu biểu cảm
        if self.emotions_recorded:
            df = pd.DataFrame(self.emotions_recorded)
            df.to_csv("emotion_log.csv", index=False)
            self.label_result.config(text=f"Đã lưu {len(self.emotions_recorded)} cảm xúc vào emotion_log.csv")
            print("Đã lưu ảnh")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
