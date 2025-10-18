import cv2
import tkinter as tk
from tkinter import ttk
from predict_emotion import predict_emotion
from collections import Counter
import pandas as pd
import time
import PIL.Image, PIL.ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

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
        self.interval_var = tk.DoubleVar(value=0.5)
        ttk.Combobox(root, textvariable=self.interval_var, values=[ 0.2, 0.5, 1.0, 2.0], font=("Arial", 14)).pack()

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

        #Biểu đồ
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8,5))
        self.ax.set_title("Biểu đồ cảm xúc")
        self.ax.set_ylabel("Số lần xuất hiện")
        self.fig.show()

        #Dữ liệu lưu cảm xúc
        self.emotions_recorded = []
        self.running = False
        self.start_time = None
        self.cap = None

    #Nhận diện
    def start_detection(self):
        if self.running:
            return
        self.running = True
        self.emotions_recorded.clear()
        self.label_result.config(text="")
        self.label_status.config(text="Đang mở webcam")

        self.cap = cv2.VideoCapture(0)
        self.start_time = time.time()
        self.duration = self.duration_var.get()
        self.interval = int(self.interval_var.get() * 1000)

        self.root.after(100, self.detect_loop)

    #Dự đoán
    def detect_loop(self):
        try:
            if not self.running:
                return
            elapsed = time.time() - self.start_time
            if elapsed >= self.duration:
                self.label_status.config(text="Đã nhận diện")
                self.running = False
                return

            ret, frame = self.cap.read()
            if ret and frame is not None:
                emotion, frame, _ = predict_emotion(frame)
                if emotion:
                    self.emotions_recorded.append({'time': elapsed, 'label': emotion})
                    self.update_chart()

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = PIL.Image.fromarray(frame_rgb)
                imgtk = PIL.ImageTk.PhotoImage(image =img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image = imgtk)

        except Exception as e:
            print("Lỗi detect_loop:", e)

        finally:
            if self.running:
                self.root.after(self.interval, self.detect_loop)
    #Cập nhật biểu đồ
    def update_chart(self):
        EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

        #Đếm số cảm xúc xuất hiện
        labels = [e['label'] for e in self.emotions_recorded]
        counter = Counter(labels)

        counts = [counter.get(emotion, 0) for emotion in EMOTION_LABELS]


        self.ax.clear()
        self.ax.bar(EMOTION_LABELS, counts, color='skyblue')
        self.ax.set_title("Biểu đồ cảm xúc (theo tần suất)")
        self.ax.set_ylabel("Số lần xuất hiện")
        self.ax.set_ylim(0, max(counts) + 1 if any(counts) else 1)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        #Dừng nhận diện
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
            self.update_chart()

        #Kết quả
        self.analyze_emotions()

    #Phân tích biểu cảm
    def analyze_emotions(self):
        if not self.emotions_recorded:
            self.label_result.config(text="Không phát hiện biểu cảm")
            return

        labels = [e['label'] for e in self.emotions_recorded]
        counter = Counter(labels)
        most_common_emotion, count = counter.most_common(1)[0]
        suggestion = self.get_emotion_suggestion(most_common_emotion)

        result_text = (f"Biểu cảm xuất hiện nhiều nhất: {most_common_emotion} "
                       f"({count} lần)\n\n{suggestion}")
        self.label_result.config(text=result_text)

    #Biểu cảm xuất hện nhiều nhất
    def get_emotion_suggestion(self, emotion):
        messages = {
            'Happy': "Bạn có vẻ rất vui hôm nay. Hãy tận hưởng niềm vui và lan tỏa năng lượng tích cực nhé!",
            'Sad': "Có vẻ bạn đang hơi buồn. Hãy thư giãn và làm điều gì khiến bạn cảm thấy tốt hơn.",
            'Angry': "Bạn đang có vẻ tức giận. Hít thở sâu và giữ bình tĩnh nhé!",
            'Fear': "Bạn có vẻ lo lắng. Mọi chuyện rồi sẽ ổn thôi.",
            'Disgust': "Bạn có chút khó chịu. Hãy nghỉ ngơi một chút.",
            'Surprise': "Bạn có vẻ khá bất ngờ. Có chuyện gì thú vị vừa xảy ra chăng?",
            'Neutral': "Bạn đang khá bình tĩnh và tập trung đấy!"
        }
        return messages.get(emotion, "Không thấy biểu cảm")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionApp(root)
    root.mainloop()
