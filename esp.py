import csv
import os
import time
from datetime import datetime

import cv2
import PIL.Image
import PIL.ImageTk
import tkinter as tk
from tkinter import ttk

from analyze import analyze_emotion_log
from predict_emotion import draw_emotion_results, predict_emotion
from predict_face import draw_face_results, predict_face_id

ESP32_STREAM_URL = "http://10.62.123.117:81/stream"
LOG_FILE = "emotion_log.csv"
LOG_INTERVAL = 1.0
RECONNECT_INTERVAL_MS = 3000
MAX_READ_FAILS_BEFORE_RECONNECT = 5


class EspEmotionFaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ESP32 Cam - Emotion + Face Verification")
        self.root.geometry("1240x760")
        self.root.configure(bg="#eef3ff")

        self.running = False
        self.cap = None
        self.last_log = 0.0
        self.read_fail_count = 0
        self.reconnect_job = None

        self._build_ui()
        self._init_log()

    def _build_ui(self):
        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 12), padding=8)

        main = tk.Frame(self.root, bg="#eef3ff")
        main.pack(fill="both", expand=True, padx=20, pady=20)

        left = tk.Frame(main, bg="#111")
        left.pack(side="left", fill="both", expand=True, padx=(0, 20))

        self.video_label = tk.Label(left, bg="#111")
        self.video_label.pack(fill="both", expand=True)

        right = tk.Frame(main, bg="#f7f9ff", width=360)
        right.pack(side="right", fill="y")

        ttk.Label(right, text="ESP32 CAM", font=("Segoe UI", 18, "bold"), background="#f7f9ff").pack(pady=(20, 10))
        ttk.Button(right, text="Bat camera", command=self.start).pack(fill="x", padx=24, pady=6)
        ttk.Button(right, text="Dung", command=self.stop).pack(fill="x", padx=24, pady=6)

        self.status = tk.Label(right, text="San sang", font=("Segoe UI", 12, "bold"), bg="#f7f9ff", fg="#0b5")
        self.status.pack(pady=12)

        self.emotion_label = tk.Label(right, text="Emotion: -", font=("Segoe UI", 12), bg="#f7f9ff")
        self.emotion_label.pack(pady=6)

        self.face_label = tk.Label(right, text="Identity: -", font=("Segoe UI", 12), bg="#f7f9ff")
        self.face_label.pack(pady=6)

        self.summary = tk.Label(right, text="", wraplength=300, justify="left", bg="#f7f9ff", font=("Segoe UI", 11))
        self.summary.pack(pady=10, padx=12)

    def _init_log(self):
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        "time",
                        "source",
                        "emotion",
                        "emotion_confidence",
                        "identity",
                        "identity_confidence",
                        "verified",
                    ]
                )

    def _log(self, emotion, emotion_conf, identity, identity_conf, verified):
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "esp32",
                    emotion,
                    f"{emotion_conf * 100:.2f}",
                    identity,
                    f"{identity_conf * 100:.2f}",
                    int(bool(verified)),
                ]
            )

    def _release_capture(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _cancel_reconnect(self):
        if self.reconnect_job is not None:
            self.root.after_cancel(self.reconnect_job)
            self.reconnect_job = None

    def _try_connect(self):
        if not self.running:
            return

        self._release_capture()
        self.cap = cv2.VideoCapture(ESP32_STREAM_URL)

        if self.cap.isOpened():
            self.read_fail_count = 0
            self.status.config(text="Dang chay", fg="green")
            self.reconnect_job = None
            self.update_frame()
            return

        self.status.config(text="Mat ket noi, dang thu ket noi lai...", fg="#cc5500")
        self._schedule_reconnect()

    def _schedule_reconnect(self):
        if not self.running or self.reconnect_job is not None:
            return

        self.reconnect_job = self.root.after(RECONNECT_INTERVAL_MS, self._reconnect_callback)

    def _reconnect_callback(self):
        self.reconnect_job = None
        self._try_connect()

    def _handle_disconnect(self):
        self.status.config(text="Mat ket noi, dang thu ket noi lai...", fg="#cc5500")
        self._release_capture()
        self._schedule_reconnect()

    def start(self):
        if self.running:
            return

        self.running = True
        self.summary.config(text="")
        self._cancel_reconnect()
        self._try_connect()

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.read_fail_count += 1
            if self.read_fail_count >= MAX_READ_FAILS_BEFORE_RECONNECT:
                self._handle_disconnect()
            else:
                self.root.after(20, self.update_frame)
            return

        self.read_fail_count = 0

        emotion_results = predict_emotion(frame)
        face_results = predict_face_id(frame)

        draw_emotion_results(frame, emotion_results)
        draw_face_results(frame, face_results)

        emotion = "no-face"
        emotion_conf = 0.0
        if emotion_results:
            best_e = max(emotion_results, key=lambda x: x["confidence"])
            emotion = best_e["emotion"]
            emotion_conf = float(best_e["confidence"])

        identity = "unknown"
        identity_conf = 0.0
        verified = False
        if face_results:
            best_f = max(face_results, key=lambda x: x["confidence"])
            identity = best_f["identity"]
            identity_conf = float(best_f["confidence"])
            verified = bool(best_f["verified"])

        self.emotion_label.config(text=f"Emotion: {emotion} ({emotion_conf * 100:.1f}%)")
        self.face_label.config(text=f"Identity: {identity} ({identity_conf * 100:.1f}%) - {'verified' if verified else 'unknown'}")

        if time.time() - self.last_log >= LOG_INTERVAL:
            self._log(emotion, emotion_conf, identity, identity_conf, verified)
            self.last_log = time.time()

        frame = cv2.resize(frame, (760, 560))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = PIL.ImageTk.PhotoImage(PIL.Image.fromarray(rgb))
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(20, self.update_frame)

    def stop(self, auto=False):
        self.running = False
        self.read_fail_count = 0
        self._cancel_reconnect()
        self._release_capture()

        if not auto:
            self.status.config(text="Da dung", fg="#555")

        summary, message = analyze_emotion_log(LOG_FILE)
        self.summary.config(text=f"{summary}\n\n{message}")


if __name__ == "__main__":
    root = tk.Tk()
    app = EspEmotionFaceApp(root)
    root.mainloop()
