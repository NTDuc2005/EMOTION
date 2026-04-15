import argparse
import os
import time

import cv2
from ultralytics import YOLO

from config import FACE_TRAIN_DIR, YOLO_FACE_WEIGHTS


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def open_source(source, url):
    if source == "webcam":
        return cv2.VideoCapture(0)
    return cv2.VideoCapture(url)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Ten nguoi can chup")
    parser.add_argument("--source", choices=["webcam", "esp"], default="webcam")
    parser.add_argument("--url", default="http://10.62.123.117:81/stream", help="URL stream ESP32")
    parser.add_argument("--num", type=int, default=80, help="So anh can luu")
    parser.add_argument("--interval", type=float, default=0.25, help="Khoang cach giua 2 lan chup (giay)")
    parser.add_argument("--padding", type=int, default=12)
    args = parser.parse_args()

    person = args.name.strip().lower()
    save_dir = os.path.join(FACE_TRAIN_DIR, person)
    ensure_dir(save_dir)

    detector = YOLO(YOLO_FACE_WEIGHTS)
    cap = open_source(args.source, args.url)

    if not cap.isOpened():
        raise RuntimeError("Khong mo duoc camera/stream")

    print(f"Dang chup cho: {person}")
    print("Nhan q de dung som")

    count = 0
    last_shot = 0.0

    while count < args.num:
        ok, frame = cap.read()
        if not ok:
            continue

        result = detector(frame, conf=0.5)[0]
        if result.boxes is not None and len(result.boxes) > 0:
            x1, y1, x2, y2 = map(int, result.boxes[0].xyxy[0].tolist())
            h, w, _ = frame.shape
            x1, y1 = max(0, x1 - args.padding), max(0, y1 - args.padding)
            x2, y2 = min(w, x2 + args.padding), min(h, y2 + args.padding)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

            now = time.time()
            if now - last_shot >= args.interval:
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    out = os.path.join(save_dir, f"{person}_{count+1:04d}.jpg")
                    cv2.imwrite(out, face)
                    count += 1
                    last_shot = now

        cv2.putText(frame, f"Saved: {count}/{args.num}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.imshow("Capture Faces", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Da luu {count} anh vao: {save_dir}")


if __name__ == "__main__":
    main()
