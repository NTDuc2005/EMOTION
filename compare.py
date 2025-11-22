import os
from collections import Counter

# Thay đường dẫn tới thư mục train/test
train_dir = r"D:\PYCHARM\EMOTION\data\train"
test_dir  = r"D:\PYCHARM\EMOTION\data\test"

def count_images(folder):
    class_counts = {}
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[class_name] = len(images)
    return class_counts

train_counts = count_images(train_dir)
test_counts  = count_images(test_dir)

print("=== Train set ===")
total_train = 0
for cls, cnt in train_counts.items():
    print(f"{cls}: {cnt} ảnh")
    total_train += cnt
print(f"Tổng số ảnh train: {total_train}\n")

print("=== Test set ===")
total_test = 0
for cls, cnt in test_counts.items():
    print(f"{cls}: {cnt} ảnh")
    total_test += cnt
print(f"Tổng số ảnh test: {total_test}\n")

# So sánh train/test
print("=== So sánh train vs test ===")
for cls in train_counts.keys():
    train_n = train_counts.get(cls,0)
    test_n  = test_counts.get(cls,0)
    ratio = test_n / train_n if train_n > 0 else 0
    print(f"{cls}: train={train_n}, test={test_n}, test/train ratio={ratio:.2f}")
    if ratio < 0.15:
        print(f" -> Lớp '{cls}' test quá ít, cần cân bằng hoặc augmentation")
