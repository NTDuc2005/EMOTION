import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from config import TRAIN_DIR, TEST_DIR, EMOTION_MODEL_PATH, EMOTION_IMG_SIZE, EMOTION_NUM_CLASSES

def build_model(img_size=EMOTION_IMG_SIZE, num_classes=EMOTION_NUM_CLASSES):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    base.trainable = False
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    out = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base.input, outputs=out)
    return model

def main():
    print("Đang load dữ liệu từ thư mục ảnh")
    datagen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True
    )
    datagen_test = ImageDataGenerator(rescale=1./255)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_gen = datagen_train.flow_from_directory(
        TRAIN_DIR,
        target_size=EMOTION_IMG_SIZE,
        batch_size=2,
        class_mode='categorical',
        color_mode='rgb'
    )

    test_gen = datagen_test.flow_from_directory(
        TEST_DIR,
        target_size=EMOTION_IMG_SIZE,
        batch_size=2,
        class_mode='categorical',
        color_mode='rgb'
    )

    print("Xây dựng mô hình")
    model = build_model()
    model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(EMOTION_MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]

    print("Bắt đầu huấn luyện")
    print("-"*100)
    model.fit(train_gen, validation_data=test_gen, epochs=5, callbacks=callbacks)


    print("Huấn luyện xong mô hình đã lưu:", EMOTION_MODEL_PATH)

if __name__ == "__main__":
    main()
