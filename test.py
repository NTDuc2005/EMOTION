import cv2
import torch
from PIL import Image
from ultralytics import YOLO
import torchvision.transforms as transforms
import argparse
import os
from models.model_cnn import ModalEncoder  # Import lớp CNN của bạn

def main():
    parser = argparse.ArgumentParser(description="Dự đoán cảm xúc trên ảnh tĩnh.")
    parser.add_argument('--image_path', type=str, required=True, help="Đường dẫn đến ảnh.")
    parser.add_argument('--output_path', type=str, default=None, help="Đường dẫn lưu ảnh kết quả.")
    parser.add_argument('--yolo_model', type=str, default='models/yolov8n-face-lin.pt', help="Mô hình YOLO.")
    parser.add_argument('--emotion_model', type=str, default='models/emotion_model.h5', help="Mô hình cảm xúc (.h5).")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"Lỗi: Không tìm thấy ảnh tại '{args.image_path}'")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    # Tải YOLO
    try:
        yolo_model = YOLO(args.yolo_model)
        print(f"Đã tải mô hình YOLO từ '{args.yolo_model}'")
    except Exception as e:
        print(f"Lỗi khi tải YOLO: {e}")
        return

    # Tải mô hình cảm xúc
    class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    try:
        emotion_model = ModalEncoder(num_class=len(class_names)).to(device)
        emotion_model.load_state_dict(torch.load(args.emotion_model, map_location=device))
        emotion_model.eval()
        print(f"Đã tải mô hình cảm xúc từ '{args.emotion_model}'")
    except Exception as e:
        print(f"Lỗi khi tải mô hình cảm xúc: {e}")
        return

    # Transform
    emotion_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    frame = cv2.imread(args.image_path)
    if frame is None:
        print(f"Lỗi: Không thể đọc file ảnh '{args.image_path}'")
        return

    results = yolo_model(frame, verbose=False)
    result = results[0]

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = box.conf[0]
        cls = int(box.cls[0])

        if cls == 0 and confidence > 0.5:
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            face_tensor = emotion_transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                output = emotion_model(face_tensor)
                _, predicted_idx = torch.max(output, 1)
                predicted_class_name = class_names[predicted_idx.item()]

            label = f'{predicted_class_name}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    output_path = args.output_path if args.output_path else os.path.splitext(args.image_path)[0] + '_output.jpg'
    cv2.imwrite(output_path, frame)
    print(f"Đã lưu ảnh kết quả tại: '{output_path}'")

if __name__ == '__main__':
    main()
