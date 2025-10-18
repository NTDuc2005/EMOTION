import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import EMOTION_LABELS

def plot_emotion(csv_path="emotion_log.csv"):

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"File {csv_path} không tồn tại.")
        return

    if df.empty:
        print("CSV không có dữ liệu.")
        return

    if 'label' not in df.columns:
        print("CSV không có cột 'label'.")
        return

    counts = df['label'].value_counts().reindex(EMOTION_LABELS).fillna(0)

    plt.figure(figsize=(8,5))
    plt.bar(counts.index, counts.values, color='skyblue')
    plt.title("Counts of detected emotions (log)")
    plt.ylabel("Số lần xuất hiện")
    plt.xlabel("Cảm xúc")
    plt.show()

if __name__ == "__main__":
    plot_from_log("emotion_log.csv")
