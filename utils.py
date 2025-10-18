import cv2
import numpy as np
from config import EMOTION_IMG_SIZE

def safe_crop(frame, box):

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w-1))
    y1 = max(0, min(int(y1), h-1))
    x2 = max(0, min(int(x2), w-1))
    y2 = max(0, min(int(y2), h-1))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

def preprocess_face_for_emotion(face_bgr, img_size=EMOTION_IMG_SIZE):


    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

    face_resized = cv2.resize(face_rgb, img_size)

    face_norm = face_resized.astype("float32") / 255.0


    return np.expand_dims(face_norm, axis=0)
