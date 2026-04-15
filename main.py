import csv
import os
import subprocess
import sys
import time
from datetime import datetime
import cv2
import PIL.Image
import PIL.ImageTk
import tkinter as tk
from tkinter import filedialog, ttk
from analyze import analyze_emotion_log
from predict_emotion import predict_emotion
from predict_face import is_face_model_ready, predict_face_id, reload_face_assets
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "emotion_log.csv"
TRAIN_SCRIPT = BASE_DIR / "train_face_model.py"
DEFAULT_ESP32_URL = "http://10.62.123.117:81/stream"

LOG_INTERVAL_SEC = 1.0
def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_distance(bbox_a, bbox_b):
    ax, ay = _bbox_center(bbox_a)
    bx, by = _bbox_center(bbox_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
def normalize_esp32_stream_url(raw_url: str) -> str:
    url = raw_url.strip()
    if not url:
        return ""

    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"

    parsed = urlparse(url)
    path = parsed.path.rstrip("/")

    if path == "":
        path = "/stream"
    elif path != "/stream":
        path = f"{path}/stream"

    return f"{parsed.scheme}://{parsed.netloc}{path}"

class EmotionFaceApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Nhan dien cam xuc + xac minh khuon mat")
        self.root.geometry("1280x760")
        self.root.configure(bg="#eaf2ff")

        self.running = False
        self.cap = None
        self.source_name = "unknown"
        self.last_log_time = 0.0
        self.esp32_url_var = tk.StringVar(value=DEFAULT_ESP32_URL)

        self.frame_count = 0
        self.last_emotion_results = []
        self.last_face_results = []
        self._build_ui()
        self._init_log()

    def _build_ui(self):
        style = ttk.Style()
        style.configure("TButton", font=("Segoe UI", 12), padding=8)

        container = tk.Frame(self.root, bg="#eaf2ff")
        container.pack(fill="both", expand=True, padx=16, pady=16)

        left = tk.Frame(container, bg="#111")
        left.pack(side="left", fill="both", expand=True, padx=(0, 16))
        self.video_label = tk.Label(left, bg="#111")
        self.video_label.pack(fill="both", expand=True)

        right = tk.Frame(container, bg="#f5f8ff", width=380, bd=1, relief="ridge")
        right.pack(side="right", fill="y")

        tk.Label(right, text="MENU", font=("Segoe UI", 18, "bold"), bg="#f5f8ff", fg="#1d3f7a").pack(pady=14)

        tk.Label(right, text="Link ESP32", font=("Segoe UI", 11, "bold"), bg="#f5f8ff", fg="#1d3f7a").pack(pady=(4, 4), padx=20, anchor="w")
        tk.Entry(right, textvariable=self.esp32_url_var, font=("Segoe UI", 11)).pack(fill="x", padx=20, pady=(0, 8))

        ttk.Button(right, text="ESP32 camera", command=self.start_esp32).pack(fill="x", padx=20, pady=6)
        ttk.Button(right, text="Anh tu folder", command=self.select_image).pack(fill="x", padx=20, pady=6)
        ttk.Button(right, text="Video tu folder", command=self.select_video).pack(fill="x", padx=20, pady=6)
        ttk.Button(right, text="Webcam laptop", command=self.start_webcam).pack(fill="x", padx=20, pady=6)
        ttk.Button(right, text="Dung", command=self.stop).pack(fill="x", padx=20, pady=6)

        self.status_label = tk.Label(right, text="San sang", font=("Segoe UI", 12, "bold"), bg="#f5f8ff", fg="#004a99")
        self.status_label.pack(pady=(14, 6))

        self.emotion_label = tk.Label(right, text="Emotion: -", font=("Segoe UI", 13), bg="#f5f8ff", fg="#0a5")
        self.emotion_label.pack(pady=4)

        self.face_label = tk.Label(right, text="Identity: -", font=("Segoe UI", 13), bg="#f5f8ff", fg="#aa5500")
        self.face_label.pack(pady=4)

        self.summary_label = tk.Label(
            right,
            text="",
            wraplength=320,
            justify="left",
            font=("Segoe UI", 11),
            bg="#f5f8ff",
            fg="#1f1f1f",
        )
        self.summary_label.pack(pady=10, padx=16)

    def _init_log(self):
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
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

    def _log_prediction(self, emotion, emotion_conf, identity, identity_conf, verified):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    now,
                    self.source_name,
                    emotion,
                    f"{emotion_conf * 100:.2f}",
                    identity,
                    f"{identity_conf * 100:.2f}",
                    int(bool(verified)),
                ]
            )

    def _draw_combined_results(self, frame, emotion_results, face_results):
        used_face_indexes = set()

        for emotion_item in emotion_results:
            bbox = emotion_item["bbox"]
            best_face_idx = None
            best_distance = float("inf")

            for idx, face_item in enumerate(face_results):
                if idx in used_face_indexes:
                    continue
                distance = _bbox_distance(bbox, face_item["bbox"])
                if distance < best_distance:
                    best_distance = distance
                    best_face_idx = idx

            x1, y1, x2, y2 = bbox
            color = (0, 180, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            emotion_label = f"Emotion: {emotion_item['emotion']} ({emotion_item['confidence'] * 100:.1f}%)"
            cv2.putText(frame, emotion_label, (x1, max(20, y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if best_face_idx is not None:
                used_face_indexes.add(best_face_idx)
                face_item = face_results[best_face_idx]
                status = face_item["identity"] if face_item["verified"] else "unknown"
                face_label = f"Identity: {status} ({face_item['confidence'] * 100:.1f}%)"
            else:
                face_label = "Identity: unknown (0.0%)"

            cv2.putText(frame, face_label, (x1, max(45, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for idx, face_item in enumerate(face_results):
            if idx in used_face_indexes:
                continue

            x1, y1, x2, y2 = face_item["bbox"]
            color = (0, 180, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            face_name = face_item["identity"] if face_item["verified"] else "unknown"
            face_label = f"Identity: {face_name} ({face_item['confidence'] * 100:.1f}%)"
            cv2.putText(frame, face_label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return frame

    def _analyze_frame(self, frame):
        emotion_results = predict_emotion(frame)
        face_results = predict_face_id(frame)

        frame = self._draw_combined_results(frame, emotion_results, face_results)

        top_emotion = "no-face"
        top_emotion_conf = 0.0
        if emotion_results:
            best = max(emotion_results, key=lambda x: x["confidence"])
            top_emotion = best["emotion"]
            top_emotion_conf = float(best["confidence"])

        top_identity = "unknown"
        top_identity_conf = 0.0
        top_verified = False
        if face_results:
            verified_faces = [x for x in face_results if x["verified"]]
            candidate = max(verified_faces or face_results, key=lambda x: x["confidence"])
            top_identity = candidate["identity"]
            top_identity_conf = float(candidate["confidence"])
            top_verified = bool(candidate["verified"])

        self.emotion_label.config(text=f"Emotion: {top_emotion} ({top_emotion_conf * 100:.1f}%)")
        self.face_label.config(
            text=f"Identity: {top_identity} ({top_identity_conf * 100:.1f}%) - {'verified' if top_verified else 'unknown'}"
        )

        now = time.time()
        if now - self.last_log_time >= LOG_INTERVAL_SEC:
            self._log_prediction(top_emotion, top_emotion_conf, top_identity, top_identity_conf, top_verified)
            self.last_log_time = now

        return frame

    def _show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(rgb)
        imgtk = PIL.ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def _update_stream(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.status_label.config(text="Khong doc duoc frame / ket thuc video", fg="#cc5500")
            self.stop(auto=True)
            return

        self.frame_count += 1

        # resize nhe hon de model chay nhanh hon
        frame = cv2.resize(frame, (640, 480))

        # chi nhan dien moi 4 frame
        if self.frame_count %  5 == 0:
            self.last_emotion_results = predict_emotion(frame)
            self.last_face_results = predict_face_id(frame)

            top_emotion = "no-face"
            top_emotion_conf = 0.0
            if self.last_emotion_results:
                best = max(self.last_emotion_results, key=lambda x: x["confidence"])
                top_emotion = best["emotion"]
                top_emotion_conf = float(best["confidence"])

            top_identity = "unknown"
            top_identity_conf = 0.0
            top_verified = False
            if self.last_face_results:
                verified_faces = [x for x in self.last_face_results if x["verified"]]
                candidate = max(verified_faces or self.last_face_results, key=lambda x: x["confidence"])
                top_identity = candidate["identity"]
                top_identity_conf = float(candidate["confidence"])
                top_verified = bool(candidate["verified"])

            self.emotion_label.config(text=f"Emotion: {top_emotion} ({top_emotion_conf * 100:.1f}%)")
            self.face_label.config(
                text=f"Identity: {top_identity} ({top_identity_conf * 100:.1f}%) - {'verified' if top_verified else 'unknown'}"
            )

        frame = self._draw_combined_results(frame, self.last_emotion_results, self.last_face_results)
        self._show_frame(frame)

        self.root.after(40, self._update_stream)

    def start_esp32(self):
        self.stop(auto=True)

        self.frame_count = 0
        self.last_emotion_results = []
        self.last_face_results = []

        esp32_url = self.esp32_url_var.get().strip()
        if not esp32_url:
            self.status_label.config(text="Chua nhap link ESP32", fg="red")
            return

        self.cap = cv2.VideoCapture(esp32_url)
        if not self.cap.isOpened():
            self.status_label.config(text="Khong mo duoc stream ESP32", fg="red")
            return

        self.source_name = "esp32"
        self.running = True
        self.status_label.config(text="Dang chay ESP32", fg="green")
        model_status = "Face model: san sang" if is_face_model_ready() else "Face model: chua train, se hien unknown"
        self.summary_label.config(text=model_status)
        self._update_stream()

    def start_webcam(self):
        self.stop(auto=True)

        self.frame_count = 0
        self.last_emotion_results = []
        self.last_face_results = []

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.status_label.config(text="Khong mo duoc webcam", fg="red")
            return

        self.source_name = "webcam"
        self.running = True
        self.status_label.config(text="Dang chay webcam", fg="green")
        model_status = "Face model: san sang" if is_face_model_ready() else "Face model: chua train, se hien unknown"
        self.summary_label.config(text=model_status)
        self._update_stream()
    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if not path:
            return

        self.stop(auto=True)
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.status_label.config(text="Khong mo duoc video", fg="red")
            return

        self.source_name = os.path.basename(path)
        self.running = True
        self.status_label.config(text=f"Dang phat: {self.source_name}", fg="green")
        model_status = "Face model: san sang" if is_face_model_ready() else "Face model: chua train, se hien unknown"
        self.summary_label.config(text=model_status)
        self._update_stream()

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            self.status_label.config(text="Khong doc duoc anh", fg="red")
            return

        self.source_name = os.path.basename(path)
        self.status_label.config(text=f"Dang xu ly anh: {self.source_name}", fg="#004a99")
        self.summary_label.config(text="Face model: san sang" if is_face_model_ready() else "Face model: chua train, se hien unknown")
        frame = self._analyze_frame(frame)
        self._show_frame(frame)

    def stop(self, auto=False):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if not auto:
            self.status_label.config(text="Da dung", fg="#555")

        summary, message = analyze_emotion_log(LOG_FILE)
        self.summary_label.config(text=f"{summary}\n\n{message}")


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionFaceApp(root)
    root.mainloop()
