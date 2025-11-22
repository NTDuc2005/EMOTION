import pandas as pd
import matplotlib.pyplot as plt
from config import EMOTION_LABELS

def plot_emotion(csv_path="emotion_log.csv"):
    # Đọc CSV
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except FileNotFoundError:
        print(f"File {csv_path} không tồn tại.")
        return

    if df.empty:
        print("CSV không có dữ liệu hợp lệ.")
        return

    if 'label' not in df.columns:
        print("CSV không có cột 'label'.")
        return

    #Tính số lần xuất hiện
    counts = df['label'].value_counts().reindex(EMOTION_LABELS).fillna(0)
    total = counts.sum()
    percentages = counts / total * 100

    #Vẽ bar chart
    plt.figure(figsize=(10,6))
    bars = plt.bar(counts.index, counts.values, color='skyblue')

    # Hiển thị % trên mỗi cột
    for bar, pct in zip(bars, percentages):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, f"{pct:.1f}%", ha='center', va='bottom')

    plt.title("Counts and probabilities of detected emotions (log)")
    plt.xlabel("Cảm xúc")
    plt.ylabel("Số lần xuất hiện")
    plt.ylim(0, max(counts.values)*1.2)
    plt.show()


if __name__ == "__main__":
    plot_emotion("emotion_log.csv")
