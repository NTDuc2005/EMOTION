import os

ROOT = os.path.dirname(__file__)
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

YOLO_FACE_WEIGHTS = os.path.join(MODELS_DIR, "yolov8n-face-lindevs.pt")
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.h5")


FER_CSV_PATH = os.path.join(ROOT, "data", "fer2013.csv")# Nếu bạn train từ FER2013 csv, đặt file ở data/fer2013.csv


EMOTION_IMG_SIZE = (64, 64)   #dùng MobileNetV2 / transfer learning
EMOTION_NUM_CLASSES = 7
EMOTION_LABELS = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

DATA_DIR = os.path.join(ROOT, "data")
TRAIN_DIR = r"D:\PYCHARM\EMOTION\data\train"
TEST_DIR = r"D:\PYCHARM\EMOTION\data\test"

