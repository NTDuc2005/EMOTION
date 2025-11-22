import os

#models
ROOT = os.path.dirname(__file__)
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

YOLO_FACE_WEIGHTS = os.path.join(MODELS_DIR, "yolov8n-face-lindevs.pt")
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.pth")  # PyTorch

#data
DATA_DIR = os.path.join(ROOT, "data")
TRAIN_DIR = r"D:\PYCHARM\EMOTION\data\train"
TEST_DIR = r"D:\PYCHARM\EMOTION\data\test"
FER_CSV_PATH = os.path.join(DATA_DIR, "fer2013.csv")  # Nếu dùng csv FER2013

#cấu hình
EMOTION_IMG_SIZE = (48, 48)   # Nếu dùng CNN nhỏ (grayscale), nếu MobileNetV2 thì (64,64) hoặc (224,224)
EMOTION_NUM_CLASSES = 7
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


BATCH_SIZE = 32
EPOCHS = 50
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
