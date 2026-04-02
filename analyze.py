import csv
from collections import Counter


def analyze_emotion_log(csv_path="emotion_log.csv"):
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        return "khong tim thay file log", ""

    if not rows:
        return "log rong", ""

    # Support both old schema (label) and new schema (emotion)
    emotion_values = []
    for row in rows:
        value = row.get("emotion") or row.get("label")
        if value:
            emotion_values.append(value.strip().lower())

    if not emotion_values:
        return "khong co cot emotion/label hop le", ""

    counter = Counter(emotion_values)
    emotion, count = counter.most_common(1)[0]

    messages = {
        "happy": "Ban co ve vui ve. Giu nang luong tich cuc nhe!",
        "sad": "Ban co ve buon. Thu nghi ngoi mot chut va thu gian.",
        "angry": "Ban co ve dang cang thang. Thu hit tho sau de binh tinh.",
        "fear": "Ban dang hoi lo lang. Moi chuyen roi se on.",
        "disgust": "Ban co chut kho chiu. Thu doi khong gian de de chiu hon.",
        "surprise": "Ban dang bat ngo. Co ve vua co dieu thu vi xay ra.",
        "neutral": "Ban dang kha binh tinh va tap trung.",
    }

    summary = f"Cam xuc xuat hien nhieu nhat: {emotion} ({count} lan)"
    message = messages.get(emotion, "Khong xac dinh duoc thong diep phu hop.")
    return summary, message


if __name__ == "__main__":
    s, m = analyze_emotion_log()
    print(s)
    print(m)
