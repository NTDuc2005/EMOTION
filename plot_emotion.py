import matplotlib.pyplot as plt
import pandas as pd

from config import EMOTION_LABELS


def plot_emotion(csv_path="emotion_log.csv"):
    try:
        df = pd.read_csv(csv_path, on_bad_lines="skip")
    except FileNotFoundError:
        print(f"File {csv_path} khong ton tai.")
        return

    if df.empty:
        print("CSV khong co du lieu hop le.")
        return

    col = "emotion" if "emotion" in df.columns else "label" if "label" in df.columns else None
    if col is None:
        print("CSV khong co cot emotion/label.")
        return

    values = df[col].astype(str).str.lower().str.strip()
    counts = values.value_counts().reindex(EMOTION_LABELS).fillna(0)
    total = counts.sum()
    if total == 0:
        print("Khong co du lieu emotion de ve.")
        return

    percentages = counts / total * 100

    plt.figure(figsize=(10, 6))
    bars = plt.bar(counts.index, counts.values, color="skyblue")

    for bar, pct in zip(bars, percentages):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{pct:.1f}%", ha="center", va="bottom")

    plt.title("Counts and probabilities of detected emotions")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.ylim(0, max(counts.values) * 1.2 if len(counts.values) else 1)
    plt.show()


if __name__ == "__main__":
    plot_emotion("emotion_log.csv")
