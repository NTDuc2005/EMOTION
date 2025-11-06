import os
import cv2

DATASET_DIRS = [
    "data/train",   # thư mục ảnh train
    "data/test"     # thư mục ảnh test
]

def convert_gray_to_rgb(image_path):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        print(f"Lỗi đọc ảnh: {image_path}")
        return
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(image_path, img_rgb)

def main():
    for dataset_dir in DATASET_DIRS:
        print(f"Đang xử lý thư mục: {dataset_dir}")
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(('.jpg')):
                    path = os.path.join(root, file)
                    convert_gray_to_rgb(path)
        print(f"Chuyển ảnh: {dataset_dir}\n")

if __name__ == "__main__":
    main()
