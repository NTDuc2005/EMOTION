import pandas as pd
from collections import Counter

df = pd.read_csv("emotion_log.csv")

if df.empty:
    print("không có dữ liệu")
else:
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

print(f"Cảm xúc xuất hiện nhiều nhất: {most_common_emotion}({count}lần)")
print("Dự đoán tâm trạng:", messages.get(most_common_emotion,"không x"))
