Du an nhan dien cam xuc + xac minh khuon mat (ESP32/webcam)

Nhan dien khuon mat theo hoc sau (CNN)
1) Du lieu train
- data/faces/train/<ten_nguoi>/*.jpg

2) Train model khuon mat (local)
- python train_face_model.py

3) Train truc tiep tu Google Drive
- python train_face_model.py --train-dir "D:\Google Drive\face_data\train" --test-dir "D:\Google Drive\face_data\test"
- Neu muon luu model len Drive:
  python train_face_model.py --train-dir "D:\Google Drive\face_data\train" --output-model "D:\Google Drive\face_model\face_identity_model.pth" --output-classes "D:\Google Drive\face_model\face_classes.json"

4) Them nguoi moi
- Copy anh vao data/faces/train/<ten_moi>/
- Chay lai: python train_face_model.py

5) Chay app
- Webcam laptop: python main.py
- ESP32-CAM: python esp.py (nho sua ESP32_STREAM_URL)

6) Kiem tra nhanh
- python test.py
- python evaluate_model.py

7) So sanh 2 anh mat
- python compare.py --img-a <path_a> --img-b <path_b> --threshold 0.75

Ghi chu
- Mode deep learning se can train lai khi them class nguoi moi.
- Log du lieu: emotion_log.csv
