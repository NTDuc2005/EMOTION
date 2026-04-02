import argparse
import os
import shutil

from config import FACE_TRAIN_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Ten nguoi muon them")
    parser.add_argument("--dir", required=True, help="Thu muc chua anh goc")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise FileNotFoundError(f"Khong tim thay thu muc: {args.dir}")

    person = args.name.strip().lower()
    target_dir = os.path.join(FACE_TRAIN_DIR, person)
    os.makedirs(target_dir, exist_ok=True)

    count = 0
    for fn in os.listdir(args.dir):
        if not fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        src = os.path.join(args.dir, fn)
        dst = os.path.join(target_dir, fn)
        shutil.copy2(src, dst)
        count += 1

    print(f"Da copy {count} anh vao {target_dir}")
    print("Can chay lai: python train_face_model.py")


if __name__ == "__main__":
    main()
