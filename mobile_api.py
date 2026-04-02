import os
from collections import Counter
from datetime import datetime
from urllib.parse import urljoin

import cv2
from flask import Flask, Response, jsonify, request, send_from_directory

from predict_emotion import predict_emotion
from predict_face import predict_face_id

APP_ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(APP_ROOT, 'mobile_api_data')
UPLOAD_DIR = os.path.join(DATA_ROOT, 'uploads')
OUTPUT_DIR = os.path.join(DATA_ROOT, 'outputs')
ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}
ESP32_BASE_URL = os.environ.get('ESP32_BASE_URL', 'http://192.168.1.47/')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)


def _normalize_base_url(raw: str) -> str:
    base = (raw or '').strip()
    if not base:
        return ''
    if not base.startswith('http://') and not base.startswith('https://'):
        base = 'http://' + base
    if not base.endswith('/'):
        base += '/'
    return base


def _esp32_stream_url() -> str:
    base = _normalize_base_url(ESP32_BASE_URL).rstrip('/')
    return f'{base}:81/stream'


def _esp32_capture_url() -> str:
    base = _normalize_base_url(ESP32_BASE_URL).rstrip('/')
    return f'{base}/capture'


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_distance(bbox_a, bbox_b):
    ax, ay = _bbox_center(bbox_a)
    bx, by = _bbox_center(bbox_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _draw_combined_results(frame, emotion_results, face_results):
    used_face_indexes = set()

    for emotion_item in emotion_results:
        bbox = emotion_item['bbox']
        best_face_idx = None
        best_distance = float('inf')

        for idx, face_item in enumerate(face_results):
            if idx in used_face_indexes:
                continue
            distance = _bbox_distance(bbox, face_item['bbox'])
            if distance < best_distance:
                best_distance = distance
                best_face_idx = idx

        x1, y1, x2, y2 = bbox
        color = (0, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        emotion_label = f"Emotion: {emotion_item['emotion']} ({emotion_item['confidence'] * 100:.1f}%)"
        cv2.putText(frame, emotion_label, (x1, max(20, y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if best_face_idx is not None:
            used_face_indexes.add(best_face_idx)
            face_item = face_results[best_face_idx]
            identity = face_item['identity'] if face_item['verified'] else 'unknown'
            face_label = f"Identity: {identity} ({face_item['confidence'] * 100:.1f}%)"
        else:
            face_label = 'Identity: unknown (0.0%)'

        cv2.putText(frame, face_label, (x1, max(45, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for idx, face_item in enumerate(face_results):
        if idx in used_face_indexes:
            continue
        x1, y1, x2, y2 = face_item['bbox']
        color = (0, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        identity = face_item['identity'] if face_item['verified'] else 'unknown'
        face_label = f"Identity: {identity} ({face_item['confidence'] * 100:.1f}%)"
        cv2.putText(frame, face_label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


def _top_results(emotion_results, face_results):
    top_emotion = {'emotion': 'no-face', 'confidence': 0.0}
    if emotion_results:
        best = max(emotion_results, key=lambda x: x['confidence'])
        top_emotion = {'emotion': best['emotion'], 'confidence': float(best['confidence'])}

    top_identity = {'identity': 'unknown', 'confidence': 0.0, 'verified': False}
    if face_results:
        verified_faces = [x for x in face_results if x['verified']]
        candidate = max(verified_faces or face_results, key=lambda x: x['confidence'])
        top_identity = {
            'identity': candidate['identity'],
            'confidence': float(candidate['confidence']),
            'verified': bool(candidate['verified']),
        }

    return top_emotion, top_identity


def _save_upload(file_storage, allowed_exts):
    filename = file_storage.filename or ''
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_exts:
        raise ValueError(f'Unsupported file type: {ext}')

    stamped_name = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + ext
    save_path = os.path.join(UPLOAD_DIR, stamped_name)
    file_storage.save(save_path)
    return save_path, stamped_name


def _absolute_media_url(filename):
    return urljoin(request.host_url, f'media/{filename}')


def _save_output_image(frame, prefix):
    out_name = datetime.now().strftime(f'{prefix}_%Y%m%d_%H%M%S_%f.jpg')
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, frame)
    return out_name, out_path


def _analyze_frame(frame):
    emotion_results = predict_emotion(frame)
    face_results = predict_face_id(frame)
    top_emotion, top_identity = _top_results(emotion_results, face_results)
    rendered = _draw_combined_results(frame.copy(), emotion_results, face_results)
    return rendered, emotion_results, face_results, top_emotion, top_identity


def _capture_esp32_frame():
    cap = cv2.VideoCapture(_esp32_capture_url())
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    return frame


@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'esp32_base_url': ESP32_BASE_URL})


@app.get('/media/<path:filename>')
def media(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.get('/esp32-stream-analyze')
def esp32_stream_analyze():
    stream_url = _esp32_stream_url()

    def generate():
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            return
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                rendered, _, _, _, _ = _analyze_frame(frame)
                ok_encode, buffer = cv2.imencode('.jpg', rendered)
                if not ok_encode:
                    continue
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
        finally:
            cap.release()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.get('/esp32-snapshot-analyze')
def esp32_snapshot_analyze():
    frame = _capture_esp32_frame()
    if frame is None:
        return jsonify({'error': 'Khong chup duoc anh tu ESP32'}), 400

    rendered, _, face_results, top_emotion, top_identity = _analyze_frame(frame)
    out_name, _ = _save_output_image(rendered, 'camera')
    return jsonify({
        'message': 'Nhan dien tu camera thanh cong',
        'emotion': top_emotion,
        'identity': top_identity,
        'faces_detected': len(face_results),
        'result_url': _absolute_media_url(out_name),
    })


@app.post('/analyze-image')
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400
    try:
        upload_path, _ = _save_upload(request.files['file'], ALLOWED_IMAGE_EXTS)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    frame = cv2.imread(upload_path)
    if frame is None:
        return jsonify({'error': 'Khong doc duoc anh'}), 400

    rendered, _, face_results, top_emotion, top_identity = _analyze_frame(frame)
    out_name, _ = _save_output_image(rendered, 'image')
    return jsonify({
        'message': 'Nhan dien anh thanh cong',
        'emotion': top_emotion,
        'identity': top_identity,
        'result_url': _absolute_media_url(out_name),
        'faces_detected': len(face_results),
    })


@app.post('/analyze-video')
def analyze_video():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400
    try:
        upload_path, _ = _save_upload(request.files['file'], ALLOWED_VIDEO_EXTS)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400

    cap = cv2.VideoCapture(upload_path)
    if not cap.isOpened():
        return jsonify({'error': 'Khong mo duoc video'}), 400

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    out_name = datetime.now().strftime('video_%Y%m%d_%H%M%S_%f.mp4')
    out_path = os.path.join(OUTPUT_DIR, out_name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    emotion_counter = Counter()
    identity_counter = Counter()
    processed_frames = 0
    face_frames = 0
    preview_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rendered, emotion_results, face_results, top_emotion, top_identity = _analyze_frame(frame)
        if emotion_results or face_results:
            face_frames += 1
        if top_emotion['emotion'] != 'no-face':
            emotion_counter[top_emotion['emotion']] += 1
        if top_identity['identity'] != 'unknown' and top_identity['verified']:
            identity_counter[top_identity['identity']] += 1
        writer.write(rendered)
        if preview_frame is None and (emotion_results or face_results):
            preview_frame = rendered.copy()
        processed_frames += 1

    cap.release()
    writer.release()

    dominant_emotion = emotion_counter.most_common(1)[0][0] if emotion_counter else 'no-face'
    dominant_identity = identity_counter.most_common(1)[0][0] if identity_counter else 'unknown'

    preview_url = None
    if preview_frame is not None:
        preview_name, _ = _save_output_image(preview_frame, 'video_preview')
        preview_url = _absolute_media_url(preview_name)

    return jsonify({
        'message': 'Nhan dien video thanh cong',
        'frames_processed': processed_frames,
        'frames_with_face': face_frames,
        'dominant_emotion': dominant_emotion,
        'dominant_identity': dominant_identity,
        'emotion_distribution': dict(emotion_counter),
        'identity_distribution': dict(identity_counter),
        'result_url': _absolute_media_url(out_name),
        'preview_url': preview_url,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
