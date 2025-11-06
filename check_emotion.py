import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from config import EMOTION_MODEL_PATH, EMOTION_IMG_SIZE, EMOTION_NUM_CLASSES

def main():
    print("=== KIỂM TRA MÔ HÌNH CẢM XÚC ===")

    # 1. Kiểm tra file mô hình
    if not os.path.exists(EMOTION_MODEL_PATH):
        print(f"❌ Không tìm thấy mô hình tại: {EMOTION_MODEL_PATH}")
        print("👉 Hãy chạy file train_model.py để huấn luyện trước.")
        return

    # 2. Load mô hình
    print(f"✅ Đã tìm thấy mô hình: {EMOTION_MODEL_PATH}")
    try:
        model = load_model(EMOTION_MODEL_PATH)
        print("✅ Load mô hình thành công.\n")
    except Exception as e:
        print(f"❌ Lỗi khi load mô hình: {e}")
        return

    # 3. In tóm tắt cấu trúc mô hình
    print("=== CẤU TRÚC MÔ HÌNH ===")
    model.summary()

    # 4. Kiểm tra thiết bị TensorFlow
    print("\n=== THIẾT BỊ TENSORFLOW ===")
    devices = tf.config.list_physical_devices()
    for d in devices:
        print(f"- {d.device_type}: {d.name}")

    # 5. Thử tạo input giả để test mô hình
    print("\n=== KIỂM TRA DỰ ĐOÁN THỬ ===")
    dummy_input = np.random.rand(1, EMOTION_IMG_SIZE[0], EMOTION_IMG_SIZE[1], 3)
    pred = model.predict(dummy_input)
    print(f"📈 Dự đoán đầu ra (mẫu ngẫu nhiên):\n{pred}")
    print(f"📊 Số lớp cảm xúc: {EMOTION_NUM_CLASSES}")
    print("✅ Mô hình hoạt động bình thường.")

if __name__ == "__main__":
    main()
