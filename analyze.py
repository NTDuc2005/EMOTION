import pandas as pd
from collections import Counter

#
def analyze_emotion_log(csv_path="emotion_log.csv"):
    try:
        df = pd.read_csv(csv_path, on_bad_lines='skip')
    except FileNotFoundError:
        return "không có csv ", ""

    if df.empty:
        return "Không có dữ liệu ", ""

    if 'label' not in df.columns:
        return "không có cột label", ""

    labels = df['label'].tolist()
    counter = Counter(labels)
    most_common_emotion, count = counter.most_common(1)[0]

    messages = {
        'Happy': "Bạn có vẻ rất vui hôm nay. Hãy tận hưởng niềm vui và lan tỏa năng lượng tích cực nhé!",
        'Sad': "Có vẻ bạn đang hơi buồn. Hãy thư giãn và làm điều gì khiến bạn cảm thấy tốt hơn.",
        'Angry': "Bạn đang có vẻ tức giận. Hít thở sâu và giữ bình tĩnh nhé!",
        'Fear': "Bạn có vẻ lo lắng. Mọi chuyện rồi sẽ ổn thôi.",
        'Disgust': "Bạn có chút khó chịu. Hãy nghỉ ngơi một chút.",
        'Surprise': "Bạn có vẻ khá bất ngờ. Có chuyện gì thú vị vừa xảy ra chăng?",
        'Neutral': "Bạn đang khá bình tĩnh và tập trung đấy!"
    }

    summary = f"Cảm xúc xuất hiện nhiều nhất: {most_common_emotion} ({count} lần)"
    message = messages.get(most_common_emotion, "Không xác định cảm xúc.")

    return summary, message

if __name__ == "__main__":
    s, m = analyze_emotion_log()
    print(s)
    print("Dự đoán tâm trạng:", m)
