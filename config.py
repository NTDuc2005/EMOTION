import os

ROOT = os.path.dirname(__file__)
MODELS_DIR = os.path.join(ROOT, "models")
DATA_DIR = os.path.join(ROOT, "data")

os.makedirs(MODELS_DIR, exist_ok=True)

YOLO_FACE_WEIGHTS = os.path.join(MODELS_DIR, "yolov8n-face-lindevs.pt")

EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion_model.pth")
EMOTION_IMG_SIZE = (48, 48)
EMOTION_NUM_CLASSES = 7
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_identity_model.pth")
FACE_CLASSES_PATH = os.path.join(MODELS_DIR, "face_classes.json")
FACE_IMG_SIZE = (112, 112)
FACE_MIN_CONFIDENCE = 0.35

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
FACE_TRAIN_DIR = os.path.join(DATA_DIR, "faces", "train")
FACE_TEST_DIR = os.path.join(DATA_DIR, "faces", "test")

BATCH_SIZE = 32
EPOCHS = 30
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
