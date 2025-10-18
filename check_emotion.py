import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import EMOTION_MODEL_PATH, EMOTION_IMG_SIZE, EMOTION_NUM_CLASSES

def main():
    print("Kiểm tra mô hình")

    # 1. Kiểm tra file model
    if not os.path.exists(EMOTION_MODEL_PATH):
        print(f"Chưa tìm thấy mô hình tại: {EMOTION_MODEL_PATH}")
        print("Hãy chạy file train_model.py để huấn luyện trước")
        return

    # 2. Load mô hình
    print(f"Đã tìm thấy mô hình: {EMOTION_MODEL_PATH}")
    try:
        model = load_model(EMOTION_MODEL_PATH)
        print("Load mô hình thành công.")
    except Exception as e:
        print("Lỗi khi load mô hình:", e)
        return

    # 3. In tóm tắt cấu trúc mô hình
    print("Thông tin mô hình")
    model.summary()

    # 4. Kiểm tra thiết bị TensorFlow đang sử dụng
    print("TensorFlow")
    devices = tf.config.list_physical_devices()
    for d in devices:
        print(f"- {d.device_type}: {d.name}")

    # 5. Thử tạo input giả để kiểm tra mô hình hoạt động
    print("Kiểm tra")
    import numpy as np
    dummy_input = np.random.rand(1, EMOTION_IMG_SIZE[0], EMOTION_IMG_SIZE[1], 3)
    pred = model.predict(dummy_input)
    print(f"Dự đoán mẫu: {pred}")
    print(f"Số lớp cảm xúc: {EMOTION_NUM_CLASSES}")

if __name__ == "__main__":
    main()
