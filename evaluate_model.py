import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from config import TEST_DIR, EMOTION_MODEL_PATH, EMOTION_LABELS, EMOTION_IMG_SIZE

print("Đang tải mô hình")
model = tf.keras.models.load_model(EMOTION_MODEL_PATH)
#Tải dữ liệu test
print("Đang tải dữ liệu test từ thư mục:", TEST_DIR)

datagen_test = ImageDataGenerator(rescale=1./255)
test_gen = datagen_test.flow_from_directory(
    TEST_DIR,
    target_size=EMOTION_IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)


#Đánh giá mô hình
loss, acc = model.evaluate(test_gen, verbose=1)
print(f"Accuracy: {acc*100:.2f}% | Loss: {loss:.4f}")


#Dự đoán và vẽ biểu đồ xác suất

print("Đang dự đoán và vẽ xác suất cảm xúc")
pred_probs = model.predict(test_gen, verbose=1)
pred_labels = np.argmax(pred_probs, axis=1)

true_labels = test_gen.classes

# Hiển thị 5 ảnh ngẫu nhiên cùng xác suất
indices = np.random.choice(len(test_gen.filenames), 5, replace=False)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(indices):
    img_path = test_gen.filepaths[idx]
    img = plt.imread(img_path)
    true_label = EMOTION_LABELS[true_labels[idx]]
    pred_label = EMOTION_LABELS[pred_labels[idx]]
    probs = pred_probs[idx]

    plt.subplot(2, 5, i+1)
    plt.imshow(img)
    plt.title(f"True: {true_label} Pred: {pred_label}")
    plt.axis('off')

    plt.subplot(2, 5, i+6)
    plt.bar(EMOTION_LABELS, probs)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.title("Probabilities")

plt.tight_layout()
plt.show()


#Ma trận nhầm lẫn

print("Vẽ Confusion Matrix...")
cm = confusion_matrix(true_labels, pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=EMOTION_LABELS)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix - Emotion Classification")
plt.show()
