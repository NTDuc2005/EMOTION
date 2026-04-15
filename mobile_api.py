import csv
import os
import threading
import time
from collections import Counter, deque
from datetime import datetime
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory

from predict_emotion import predict_emotion
from predict_face import detect_face_boxes, predict_face_id

APP_ROOT = os.path.dirname(__file__)
DATA_ROOT = os.path.join(APP_ROOT, 'mobile_api_data')
UPLOAD_DIR = os.path.join(DATA_ROOT, 'uploads')
OUTPUT_DIR = os.path.join(DATA_ROOT, 'outputs')
AUTO_SAVE_DIR = os.path.join(OUTPUT_DIR, 'camera_auto')
LOG_FILE = os.path.join(DATA_ROOT, 'camera_history.csv')

ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
ALLOWED_VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}
ESP32_BASE_URL = os.environ.get('ESP32_BASE_URL', 'http://10.62.123.117/')

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUTO_SAVE_DIR, exist_ok=True)

app = Flask(__name__)

# -----------------------------
# Tracking + smoothing
# -----------------------------
_TRACK_HISTORY = []
_TRACK_SEQUENCE = 1
_TRACK_TTL_SECONDS = 3.0
_TRACK_MAX_EMOTIONS = 8
_TRACK_MAX_IDENTITIES = 10
_TRACK_MAX_BBOXES = 6

# -----------------------------
# Live analyzer state
# -----------------------------
_LIVE_ANALYZER_LOCK = threading.Lock()
_LIVE_ANALYZER = {
    'base_url': '',
    'thread': None,
    'stop_event': None,
    'latest_jpeg': None,
    'latest_payload': None,
    'latest_timestamp': 0,
    'last_error': None,
}

# tốc độ xử lý backend
_LIVE_ANALYZE_INTERVAL_SECONDS = 0.12

# tự lưu kết quả camera mỗi N giây
AUTO_SAVE_INTERVAL_SECONDS = 2.0
_LAST_AUTO_SAVE_TS = 0.0


def _init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'time',
                'source',
                'emotion',
                'emotion_confidence',
                'identity',
                'identity_confidence',
                'verified',
                'face_status',
                'faces_detected',
                'result_image',
            ])


_init_log()


def _append_camera_log(subject, faces_detected, result_image_name=''):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            now,
            'esp32',
            subject.get('emotion', 'no-face'),
            f"{float(subject.get('emotion_confidence', 0.0)) * 100:.2f}",
            subject.get('identity', 'unknown'),
            f"{float(subject.get('identity_confidence', 0.0)) * 100:.2f}",
            int(bool(subject.get('verified', False))),
            subject.get('face_status', 'Khong co thong tin khuon mat'),
            int(faces_detected),
            result_image_name,
        ])


def _next_track_id():
    global _TRACK_SEQUENCE
    track_id = _TRACK_SEQUENCE
    _TRACK_SEQUENCE += 1
    return track_id


def _cleanup_tracks(now):
    _TRACK_HISTORY[:] = [
        track for track in _TRACK_HISTORY
        if now - track['last_seen'] <= _TRACK_TTL_SECONDS
    ]


def _normalize_base_url(raw: str) -> str:
    base = (raw or '').strip()
    if not base:
        return ''
    if not base.startswith('http://') and not base.startswith('https://'):
        base = 'http://' + base
    if not base.endswith('/'):
        base += '/'
    return base


def _get_esp32_base_url() -> str:
    requested = request.args.get('base_url', '')
    return _normalize_base_url(requested or ESP32_BASE_URL)


def _esp32_stream_url() -> str:
    base = _get_esp32_base_url().rstrip('/')
    return f'{base}:81/stream'


def _esp32_capture_url() -> str:
    base = _get_esp32_base_url().rstrip('/')
    return f'{base}/capture'


def _live_result_url(base_url: str, timestamp: int) -> str:
    encoded_base_url = quote(base_url, safe='')
    return urljoin(
        request.host_url,
        f'esp32-live-result.jpg?base_url={encoded_base_url}&ts={timestamp}',
    )


def _absolute_media_url(filename):
    return urljoin(request.host_url, f'media/{filename}')


def _save_output_image(frame, prefix, folder=None):
    target_dir = folder or OUTPUT_DIR
    os.makedirs(target_dir, exist_ok=True)
    out_name = datetime.now().strftime(f'{prefix}_%Y%m%d_%H%M%S_%f.jpg')
    out_path = os.path.join(target_dir, out_name)
    cv2.imwrite(out_path, frame)
    return out_name, out_path


def _save_upload(file_storage, allowed_exts):
    filename = file_storage.filename or ''
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed_exts:
        raise ValueError(f'Unsupported file type: {ext}')

    stamped_name = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + ext
    save_path = os.path.join(UPLOAD_DIR, stamped_name)
    file_storage.save(save_path)
    return save_path, stamped_name


# --------------------------------
# Live analyzer thread
# --------------------------------
def _start_live_analyzer_locked(base_url: str):
    stop_event = threading.Event()
    thread = threading.Thread(
        target=_run_live_analyzer,
        args=(base_url, stop_event),
        daemon=True,
        name='esp32-live-analyzer',
    )
    _LIVE_ANALYZER.update({
        'base_url': base_url,
        'thread': thread,
        'stop_event': stop_event,
        'latest_jpeg': None,
        'latest_payload': None,
        'latest_timestamp': 0,
        'last_error': None,
    })
    thread.start()


def _ensure_live_analyzer(base_url: str):
    normalized = _normalize_base_url(base_url)
    with _LIVE_ANALYZER_LOCK:
        current_thread = _LIVE_ANALYZER.get('thread')
        current_base_url = _LIVE_ANALYZER.get('base_url')
        if current_thread is not None and current_thread.is_alive() and current_base_url == normalized:
            return

        current_stop_event = _LIVE_ANALYZER.get('stop_event')
        if current_stop_event is not None:
            current_stop_event.set()

        _start_live_analyzer_locked(normalized)


def _run_live_analyzer(base_url: str, stop_event: threading.Event):
    stream_url = f"{base_url.rstrip('/')}:81/stream"

    global _LAST_AUTO_SAVE_TS

    while not stop_event.is_set():
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            with _LIVE_ANALYZER_LOCK:
                if _LIVE_ANALYZER.get('base_url') == base_url:
                    _LIVE_ANALYZER['last_error'] = 'Khong mo duoc stream ESP32'
            stop_event.wait(1.0)
            continue

        failed_reads = 0
        max_failed_reads = 15

        try:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failed_reads += 1
                    with _LIVE_ANALYZER_LOCK:
                        if _LIVE_ANALYZER.get('base_url') == base_url:
                            _LIVE_ANALYZER['last_error'] = f'Mat frame tam thoi ({failed_reads}/{max_failed_reads})'

                    if failed_reads < max_failed_reads:
                        stop_event.wait(0.05)
                        continue

                    # thử mở lại stream thay vì break ngay
                    cap.release()
                    stop_event.wait(0.25)
                    cap = cv2.VideoCapture(stream_url)
                    failed_reads = 0
                    if not cap.isOpened():
                        with _LIVE_ANALYZER_LOCK:
                            if _LIVE_ANALYZER.get('base_url') == base_url:
                                _LIVE_ANALYZER['last_error'] = 'Khong mo lai duoc stream ESP32'
                        stop_event.wait(0.5)
                    continue

                failed_reads = 0

                frame_height, frame_width = frame.shape[:2]
                rendered, emotion_results, face_results, subjects, top_emotion, top_identity, primary_subject = _analyze_frame(frame)

                ok_encode, buffer = cv2.imencode('.jpg', rendered)
                if not ok_encode:
                    continue

                timestamp = int(time.time() * 1000)
                payload = {
                    'message': 'Nhan dien tu camera thanh cong',
                    'emotion': top_emotion,
                    'identity': top_identity,
                    'primary_subject': primary_subject,
                    'subjects': subjects,
                    'faces_detected': len(face_results),
                    'frame_width': int(frame_width),
                    'frame_height': int(frame_height),
                    'ts': timestamp,
                }

                # tự lưu định kỳ
                now = time.time()
                if now - _LAST_AUTO_SAVE_TS >= AUTO_SAVE_INTERVAL_SECONDS:
                    save_name, _ = _save_output_image(rendered, 'camera_auto', folder=AUTO_SAVE_DIR)
                    _append_camera_log(primary_subject, len(face_results), save_name)
                    _LAST_AUTO_SAVE_TS = now

                with _LIVE_ANALYZER_LOCK:
                    if _LIVE_ANALYZER.get('base_url') != base_url or _LIVE_ANALYZER.get('stop_event') is not stop_event:
                        return
                    _LIVE_ANALYZER['latest_jpeg'] = buffer.tobytes()
                    _LIVE_ANALYZER['latest_payload'] = payload
                    _LIVE_ANALYZER['latest_timestamp'] = timestamp
                    _LIVE_ANALYZER['last_error'] = None

                stop_event.wait(_LIVE_ANALYZE_INTERVAL_SECONDS)
        finally:
            cap.release()

        stop_event.wait(0.2)


def _await_live_payload(base_url: str, wait_timeout: float = 4.0):
    normalized = _normalize_base_url(base_url)
    _ensure_live_analyzer(normalized)
    deadline = time.time() + wait_timeout

    while time.time() < deadline:
        with _LIVE_ANALYZER_LOCK:
            if _LIVE_ANALYZER.get('base_url') == normalized:
                payload = _LIVE_ANALYZER.get('latest_payload')
                timestamp = int(_LIVE_ANALYZER.get('latest_timestamp') or 0)
                last_error = _LIVE_ANALYZER.get('last_error')
                if payload is not None and timestamp:
                    return dict(payload), timestamp, None
                error = last_error
            else:
                error = None
        time.sleep(0.05)

    return None, 0, error or 'Chua nhan duoc khung hinh moi tu ESP32'


# --------------------------------
# bbox helpers
# --------------------------------
def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _bbox_size(bbox):
    return max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1])


def _bbox_area(bbox):
    width, height = _bbox_size(bbox)
    return width * height


def _bbox_distance(bbox_a, bbox_b):
    ax, ay = _bbox_center(bbox_a)
    bx, by = _bbox_center(bbox_b)
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def _bbox_iou(bbox_a, bbox_b):
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter_area / float(area_a + area_b - inter_area)


def _merge_bbox(bbox_a, bbox_b):
    return (
        int(min(bbox_a[0], bbox_b[0])),
        int(min(bbox_a[1], bbox_b[1])),
        int(max(bbox_a[2], bbox_b[2])),
        int(max(bbox_a[3], bbox_b[3])),
    )


def _smooth_bbox(history):
    weights = list(range(1, len(history) + 1))
    total = float(sum(weights))
    blended = [0.0, 0.0, 0.0, 0.0]
    for weight, bbox in zip(weights, history):
        for idx, value in enumerate(bbox):
            blended[idx] += weight * float(value)
    return tuple(int(round(value / total)) for value in blended)


def _identity_value(face_item):
    if not face_item:
        return 'unknown'
    return str(face_item.get('identity', 'unknown') or 'unknown')


def _display_identity_value(face_item):
    if not face_item:
        return 'unknown'
    return str(
        face_item.get('display_identity')
        or face_item.get('raw_identity')
        or face_item.get('identity')
        or 'unknown'
    )


def _emotion_value(emotion_item):
    if not emotion_item:
        return 'no-face'
    return str(emotion_item.get('emotion', 'no-face') or 'no-face')


def _display_emotion_value(emotion_item):
    if not emotion_item:
        return 'no-face'
    return str(
        emotion_item.get('display_emotion')
        or emotion_item.get('raw_emotion')
        or emotion_item.get('emotion')
        or 'no-face'
    )

def _face_status_from_values(identity_confidence, verified):
    if not verified:
        return 'Khong xac nhan duoc danh tinh'
    confidence = float(identity_confidence)
    if confidence >= 0.8:
        return 'Khuon mat khop rat tot'
    if confidence >= 0.6:
        return 'Khuon mat khop kha tot'
    return 'Khuon mat khop muc co ban'


def _face_status(face_item):
    if not face_item or not face_item.get('verified'):
        return 'Khong xac nhan duoc danh tinh'
    return _face_status_from_values(face_item.get('confidence', 0.0), True)


# --------------------------------
# Track smoothing
# --------------------------------
def _create_track(subject, now):
    track = {
        'track_id': _next_track_id(),
        'bbox': subject['bbox'],
        'bbox_history': deque([subject['bbox']], maxlen=_TRACK_MAX_BBOXES),
        'emotions': deque(maxlen=_TRACK_MAX_EMOTIONS),
        'identities': deque(maxlen=_TRACK_MAX_IDENTITIES),
        'last_seen': now,
        'stable_emotion': 'no-face',
        'stable_emotion_confidence': 0.0,
        'stable_identity': 'unknown',
        'stable_identity_confidence': 0.0,
        'stable_verified': False,
    }
    return track


def _score_track_match(subject, track):
    bbox = subject['bbox']
    track_bbox = track['bbox']
    iou = _bbox_iou(bbox, track_bbox)
    distance = _bbox_distance(bbox, track_bbox)
    width, height = _bbox_size(bbox)
    track_width, track_height = _bbox_size(track_bbox)
    max_dim = max(width, height, track_width, track_height, 1.0)
    normalized_distance = distance / max_dim
    size_similarity = min(_bbox_area(bbox), _bbox_area(track_bbox)) / max(_bbox_area(bbox), _bbox_area(track_bbox), 1.0)

    if iou < 0.04 and normalized_distance > 1.25:
        return -1.0

    score = (
        (iou * 0.6)
        + (max(0.0, 1.0 - normalized_distance) * 0.28)
        + (size_similarity * 0.12)
    )

    stable_identity = track.get('stable_identity', 'unknown')
    if subject.get('verified') and stable_identity != 'unknown':
        if subject.get('identity') == stable_identity:
            score += 0.22
        else:
            score -= 0.18

    if subject.get('emotion') and subject.get('emotion') != 'no-face':
        stable_emotion = track.get('stable_emotion', 'no-face')
        if stable_emotion != 'no-face' and subject.get('emotion') == stable_emotion:
            score += 0.04

    return score


def _resolve_stable_emotion(track, fallback_emotion, fallback_confidence):
    weighted_scores = Counter()
    recent_label = None
    recent_confidence = 0.0

    for idx, entry in enumerate(track['emotions'], start=1):
        label = entry['emotion']
        if not label or label == 'no-face':
            continue
        weight = idx * max(0.15, float(entry.get('confidence', 0.0)))
        if entry.get('reliable'):
            weight *= 1.25
        weighted_scores[label] += weight
        recent_label = label
        recent_confidence = float(entry.get('confidence', 0.0))

    if not weighted_scores:
        return fallback_emotion, float(fallback_confidence)

    best_emotion, best_score = weighted_scores.most_common(1)[0]
    total_score = float(sum(weighted_scores.values())) or 1.0
    smoothed_confidence = min(1.0, max(float(fallback_confidence), best_score / total_score))

    if fallback_emotion != 'no-face' and fallback_emotion == best_emotion:
        smoothed_confidence = max(smoothed_confidence, float(fallback_confidence))
    elif fallback_emotion == 'no-face' and recent_label == best_emotion:
        smoothed_confidence = max(smoothed_confidence, recent_confidence)

    return best_emotion, smoothed_confidence


def _resolve_stable_identity(track, fallback_identity, fallback_confidence, fallback_verified):
    weighted_scores = Counter()
    best_direct_confidence = {}

    for idx, entry in enumerate(track['identities'], start=1):
        label = entry['identity']
        if not label or label == 'unknown':
            continue
        confidence = float(entry.get('confidence', 0.0))
        verified = bool(entry.get('verified', False))
        weight = idx * max(0.2, confidence)
        if verified:
            weight *= 1.35
        weighted_scores[label] += weight
        best_direct_confidence[label] = max(best_direct_confidence.get(label, 0.0), confidence)

    if not weighted_scores:
        if fallback_verified and fallback_identity != 'unknown':
            return fallback_identity, float(fallback_confidence), True
        return 'unknown', 0.0, False

    best_identity, best_score = weighted_scores.most_common(1)[0]
    total_score = float(sum(weighted_scores.values())) or 1.0
    smoothed_confidence = min(
        1.0,
        max(best_direct_confidence.get(best_identity, 0.0), best_score / total_score)
    )

    if fallback_verified and fallback_identity == best_identity:
        smoothed_confidence = max(smoothed_confidence, float(fallback_confidence))

    return best_identity, smoothed_confidence, True


def _refresh_subject_labels(subject):
    subject['face_status'] = _face_status_from_values(
        subject.get('identity_confidence', 0.0),
        subject.get('verified', False),
    )

    shown_identity = (
        subject.get('display_identity')
        or subject.get('raw_identity')
        or subject.get('identity')
        or 'unknown'
    )
    # emotion chi hien thi neu reliable
    if subject.get('emotion_reliable'):
        shown_emotion = subject.get('emotion', 'no-face')
    else:
        shown_emotion = 'uncertain'
    subject['label'] = f"{shown_identity}: {shown_emotion}"
    subject['summary'] = (
        f"{shown_identity} | Cam xuc: {shown_emotion} | {subject['face_status']}"
    )
    return subject

def _stabilize_subjects(subjects):
    now = time.time()
    _cleanup_tracks(now)
    unmatched_tracks = list(_TRACK_HISTORY)

    prioritized_subjects = sorted(
        subjects,
        key=lambda item: (
            not bool(item.get('verified')),
            -float(item.get('identity_confidence', 0.0)),
            -float(item.get('emotion_confidence', 0.0)),
            -_bbox_area(item['bbox']),
        ),
    )

    for subject in prioritized_subjects:
        bbox = subject.get('bbox')
        if not bbox:
            continue

        matched_track = None
        best_score = -1.0
        for track in unmatched_tracks:
            score = _score_track_match(subject, track)
            if score > best_score:
                best_score = score
                matched_track = track

        if matched_track is None or best_score < 0.18:
            matched_track = _create_track(subject, now)
            _TRACK_HISTORY.append(matched_track)
        else:
            unmatched_tracks.remove(matched_track)

        matched_track['last_seen'] = now
        matched_track['bbox_history'].append(subject['bbox'])
        matched_track['bbox'] = _smooth_bbox(matched_track['bbox_history'])
        subject['bbox'] = matched_track['bbox']
        subject['track_id'] = matched_track['track_id']

        matched_track['emotions'].append({
            'emotion': subject.get('emotion', 'no-face'),
            'confidence': float(subject.get('emotion_confidence', 0.0)),
            'reliable': bool(subject.get('emotion_reliable', False)),
        })
        stable_emotion, stable_emotion_conf = _resolve_stable_emotion(
            matched_track,
            subject.get('emotion', 'no-face'),
            subject.get('emotion_confidence', 0.0),
        )
        matched_track['stable_emotion'] = stable_emotion
        matched_track['stable_emotion_confidence'] = stable_emotion_conf
        subject['emotion'] = stable_emotion
        subject['emotion_confidence'] = stable_emotion_conf

        if subject.get('display_emotion') in (None, '', 'uncertain', 'no-face'):
            subject['display_emotion'] = (
                subject.get('raw_emotion')
                or stable_emotion
                or 'no-face'
            )
        matched_track['identities'].append({
            'identity': subject.get('identity', 'unknown'),
            'confidence': float(subject.get('identity_confidence', 0.0)),
            'verified': bool(subject.get('verified', False)),
        })
        stable_identity, stable_identity_conf, stable_verified = _resolve_stable_identity(
            matched_track,
            subject.get('identity', 'unknown'),
            subject.get('identity_confidence', 0.0),
            subject.get('verified', False),
        )
        matched_track['stable_identity'] = stable_identity
        matched_track['stable_identity_confidence'] = stable_identity_conf
        matched_track['stable_verified'] = stable_verified

        if stable_verified:
            subject['identity'] = stable_identity
            subject['display_identity'] = stable_identity
            subject['raw_identity'] = stable_identity
            subject['identity_confidence'] = max(
                float(subject.get('identity_confidence', 0.0)),
                stable_identity_conf,
            )
            subject['verified'] = True
        else:
            subject['identity'] = 'unknown'
            subject['display_identity'] = (
                subject.get('display_identity')
                or subject.get('raw_identity')
                or stable_identity
                or 'unknown'
            )
            subject['raw_identity'] = (
                subject.get('raw_identity')
                or subject['display_identity']
            )
            subject['identity_confidence'] = float(subject.get('identity_confidence', 0.0))
            subject['verified'] = False
        _refresh_subject_labels(subject)

    return sorted(
        prioritized_subjects,
        key=lambda item: (
            not bool(item.get('verified')),
            -float(item.get('identity_confidence', 0.0)),
            -float(item.get('emotion_confidence', 0.0)),
            -_bbox_area(item['bbox']),
        ),
    )


# --------------------------------
# Subject build + drawing
# --------------------------------
def _build_subjects(emotion_results, face_results):
    subjects = []
    used_face_indexes = set()

    for emotion_item in emotion_results:
        bbox = emotion_item['bbox']
        best_face_idx = None
        best_iou = 0.0
        best_distance = float('inf')

        box_w = max(1, bbox[2] - bbox[0])
        box_h = max(1, bbox[3] - bbox[1])
        max_center_distance = max(box_w, box_h) * 0.85

        for idx, face_item in enumerate(face_results):
            if idx in used_face_indexes:
                continue
            face_bbox = face_item['bbox']
            iou = _bbox_iou(bbox, face_bbox)
            distance = _bbox_distance(bbox, face_bbox)
            if iou > best_iou or (iou == best_iou and distance < best_distance):
                best_iou = iou
                best_distance = distance
                best_face_idx = idx

        face_item = None
        if best_face_idx is not None:
            candidate = face_results[best_face_idx]
            if best_iou >= 0.2 or best_distance <= max_center_distance:
                used_face_indexes.add(best_face_idx)
                face_item = candidate

        face_status = _face_status(face_item)

        identity = _identity_value(face_item)
        display_identity = identity
        raw_identity = display_identity

        emotion = _emotion_value(emotion_item)
        display_emotion = emotion
        raw_emotion = display_emotion

        subject_bbox = _merge_bbox(bbox, face_item['bbox']) if face_item else bbox

        subjects.append({
            'identity': identity,
            'display_identity': display_identity,
            'raw_identity': raw_identity,
            'identity_confidence': float(face_item['confidence']) if face_item else 0.0,
            'verified': bool(face_item['verified']) if face_item else False,
            'face_status': face_status,
            'emotion': emotion,
            'display_emotion': display_emotion,
            'raw_emotion': raw_emotion,
            'emotion_confidence': float(emotion_item['confidence']),
            'emotion_reliable': bool(emotion_item.get('reliable', False)),
            'bbox': subject_bbox,
            'detect_confidence': max(
                float(emotion_item.get('detect_confidence', 0.0)),
                float(face_item.get('detect_confidence', 0.0)) if face_item else 0.0,
            ),
            'label': f'{display_identity}: {display_emotion}',
            'summary': f'{display_identity} | Cam xuc: {display_emotion} | {face_status}',
        })

    for idx, face_item in enumerate(face_results):
        if idx in used_face_indexes:
            continue

        identity = _identity_value(face_item)
        display_identity = _display_identity_value(face_item)
        raw_identity = display_identity
        face_status = _face_status(face_item)

        subjects.append({
            'identity': identity,
            'display_identity': display_identity,
            'raw_identity': raw_identity,
            'identity_confidence': float(face_item['confidence']),
            'verified': bool(face_item['verified']),
            'face_status': face_status,
            'emotion': 'no-face',
            'display_emotion': 'no-face',
            'raw_emotion': 'no-face',
            'emotion_confidence': 0.0,
            'emotion_reliable': False,
            'bbox': face_item['bbox'],
            'detect_confidence': float(face_item.get('detect_confidence', 0.0)),
            'label': f'{display_identity}: no-face',
            'summary': f'{display_identity} | Cam xuc: no-face | {face_status}',
        })

    return _stabilize_subjects(subjects)

def _draw_combined_results(frame, subjects):
    for subject in subjects:
        x1, y1, x2, y2 = subject['bbox']

        color = (0, 180, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        shown_emotion = (
            subject.get('display_emotion')
            or subject.get('raw_emotion')
            or subject.get('emotion')
            or 'no-face'
        )
        emotion_label = (
            f"Emotion: {shown_emotion} "
            f"({subject['emotion_confidence'] * 100:.1f}%)"
        )
        cv2.putText(
            frame,
            emotion_label,
            (x1, max(20, y1 - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # identity chi hien thi neu verified
        if subject.get('verified'):
            shown_identity = subject.get('identity', 'unknown')
        else:
            shown_identity = 'unknown'
        face_label = (
            f"Identity: {shown_identity} "
            f"({subject['identity_confidence'] * 100:.1f}%)"
        )
        cv2.putText(
            frame,
            face_label,
            (x1, max(45, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    return frame
def _top_results(emotion_results, face_results, subjects):
    top_emotion = {
        'emotion': 'no-face',
        'display_emotion': 'no-face',
        'raw_emotion': 'no-face',
        'confidence': 0.0,
    }
    if emotion_results:
        best = max(emotion_results, key=lambda x: x['confidence'])
        top_emotion = {
            'emotion': best.get('emotion', 'no-face'),
            'display_emotion': best.get('display_emotion', best.get('raw_emotion', best.get('emotion', 'no-face'))),
            'raw_emotion': best.get('raw_emotion', best.get('emotion', 'no-face')),
            'confidence': float(best['confidence']),
        }

    top_identity = {
        'identity': 'unknown',
        'display_identity': 'unknown',
        'raw_identity': 'unknown',
        'confidence': 0.0,
        'verified': False,
    }
    if face_results:
        candidate = max(face_results, key=lambda x: x['confidence'])
        top_identity = {
            'identity': candidate.get('identity', 'unknown'),
            'display_identity': candidate.get('identity', 'unknown'),
            'raw_identity': candidate.get(
                'raw_identity',
                candidate.get('identity', 'unknown'),
            ),
            'confidence': float(candidate['confidence']),
            'verified': bool(candidate['verified']),
        }

    primary_subject = max(
        subjects,
        key=lambda item: (
            bool(item.get('verified')),
            float(item.get('identity_confidence', 0.0)),
            float(item.get('emotion_confidence', 0.0)),
            _bbox_area(item['bbox']),
        ),
        default={
            'identity': 'unknown',
            'display_identity': 'unknown',
            'raw_identity': 'unknown',
            'identity_confidence': 0.0,
            'verified': False,
            'face_status': 'Khong phat hien khuon mat',
            'emotion': 'no-face',
            'display_emotion': 'no-face',
            'raw_emotion': 'no-face',
            'emotion_confidence': 0.0,
            'bbox': None,
            'label': 'unknown: no-face',
            'summary': 'unknown | Cam xuc: no-face | Khong phat hien khuon mat',
        },
    )

    return top_emotion, top_identity, primary_subject

def _analyze_frame(frame):
    # tang detect_conf de box on dinh hon
    detections = detect_face_boxes(
        frame,
        detect_conf=0.55,
        padding=12,
        min_face_size=48,
    )
    emotion_results = predict_emotion(
        frame,
        detections=detections,
        padding=12,
    )
    face_results = predict_face_id(
        frame,
        detections=detections,
        padding=12,
    )
    subjects = _build_subjects(emotion_results, face_results)
    top_emotion, top_identity, primary_subject = _top_results(
        emotion_results,
        face_results,
        subjects,
    )
    rendered = _draw_combined_results(frame.copy(), subjects)
    return rendered, emotion_results, face_results, subjects, top_emotion, top_identity, primary_subject


# --------------------------------
# ESP32 capture helpers
# --------------------------------
def _capture_esp32_stream_frame(stream_url, warmup_frames: int = 3):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        return None

    frame = None
    try:
        for _ in range(max(1, warmup_frames)):
            ok, current = cap.read()
            if ok and current is not None:
                frame = current
    finally:
        cap.release()

    return frame


def _capture_esp32_frame():
    capture_url = _esp32_capture_url()
    stream_url = _esp32_stream_url()

    try:
        req = Request(capture_url, headers={'User-Agent': 'emotion-mobile-api/1.0'})
        with urlopen(req, timeout=4) as resp:
            payload = resp.read()
        if payload:
            arr = np.frombuffer(payload, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                return frame
    except Exception:
        pass

    return _capture_esp32_stream_frame(stream_url)


# --------------------------------
# API routes
# --------------------------------
@app.get('/health')
def health():
    return jsonify({
        'status': 'ok',
        'esp32_base_url': _normalize_base_url(ESP32_BASE_URL),
    })


@app.get('/media/<path:filename>')
def media(filename):
    # thử trong outputs gốc
    path_main = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path_main):
        return send_from_directory(OUTPUT_DIR, filename)

    # thử trong camera_auto
    path_auto = os.path.join(AUTO_SAVE_DIR, filename)
    if os.path.exists(path_auto):
        return send_from_directory(AUTO_SAVE_DIR, filename)

    return jsonify({'error': 'Khong tim thay file'}), 404


@app.get('/esp32-stream-analyze')
def esp32_stream_analyze():
    stream_url = _esp32_stream_url()

    def generate():
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            return

        failed_reads = 0
        max_failed_reads = 15

        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    failed_reads += 1
                    time.sleep(0.05)

                    # KHONG break ngay de tranh nhap nhay
                    if failed_reads < max_failed_reads:
                        continue

                    cap.release()
                    time.sleep(0.25)
                    cap = cv2.VideoCapture(stream_url)
                    failed_reads = 0
                    if not cap.isOpened():
                        time.sleep(0.5)
                    continue

                failed_reads = 0

                rendered, _, _, _, _, _, _ = _analyze_frame(frame)
                ok_encode, buffer = cv2.imencode('.jpg', rendered)
                if not ok_encode:
                    continue

                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + buffer.tobytes()
                    + b'\r\n'
                )
        finally:
            cap.release()

    return Response(
        generate(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.get('/esp32-live-result.jpg')
def esp32_live_result():
    base_url = _get_esp32_base_url()
    _ensure_live_analyzer(base_url)
    deadline = time.time() + 4.0

    while time.time() < deadline:
        with _LIVE_ANALYZER_LOCK:
            latest_jpeg = _LIVE_ANALYZER.get('latest_jpeg')
            last_error = _LIVE_ANALYZER.get('last_error')
        if latest_jpeg:
            response = Response(latest_jpeg, mimetype='image/jpeg')
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response
        time.sleep(0.05)

    return jsonify({'error': last_error or 'Khong co khung hinh live moi tu ESP32'}), 503


@app.get('/esp32-snapshot-analyze')
def esp32_snapshot_analyze():
    base_url = _get_esp32_base_url()
    payload, timestamp, live_error = _await_live_payload(base_url, wait_timeout=4.0)
    if payload is not None:
        payload['result_url'] = _live_result_url(base_url, timestamp)
        return jsonify(payload)

    frame = _capture_esp32_frame()
    if frame is None:
        return jsonify({'error': live_error or 'Khong chup duoc anh tu ESP32'}), 400

    frame_height, frame_width = frame.shape[:2]
    rendered, _, face_results, subjects, top_emotion, top_identity, primary_subject = _analyze_frame(frame)
    out_name, _ = _save_output_image(rendered, 'camera')
    _append_camera_log(primary_subject, len(face_results), out_name)

    return jsonify({
        'message': 'Nhan dien tu camera thanh cong',
        'emotion': top_emotion,
        'identity': top_identity,
        'primary_subject': primary_subject,
        'subjects': subjects,
        'faces_detected': len(face_results),
        'frame_width': int(frame_width),
        'frame_height': int(frame_height),
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

    rendered, _, face_results, subjects, top_emotion, top_identity, primary_subject = _analyze_frame(frame)
    out_name, _ = _save_output_image(rendered, 'image')

    return jsonify({
        'message': 'Nhan dien anh thanh cong',
        'emotion': top_emotion,
        'identity': top_identity,
        'primary_subject': primary_subject,
        'subjects': subjects,
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
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    emotion_counter = Counter()
    identity_counter = Counter()
    subject_counter = Counter()
    processed_frames = 0
    face_frames = 0
    preview_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rendered, emotion_results, face_results, subjects, top_emotion, _, primary_subject = _analyze_frame(frame)

        if emotion_results or face_results:
            face_frames += 1

        shown_emotion = (
            top_emotion.get('display_emotion')
            or top_emotion.get('raw_emotion')
            or top_emotion.get('emotion')
            or 'no-face'
        )
        if shown_emotion != 'no-face':
            emotion_counter[shown_emotion] += 1

        shown_identity = (
            primary_subject.get('display_identity')
            or primary_subject.get('raw_identity')
            or primary_subject.get('identity')
            or 'unknown'
        )
        if shown_identity:
            identity_counter[shown_identity] += 1

        subject_identity = shown_identity
        subject_emotion = (
            primary_subject.get('display_emotion')
            or primary_subject.get('raw_emotion')
            or primary_subject.get('emotion')
            or 'no-face'
        )
        if subject_identity or subject_emotion:
            subject_counter[f"{subject_identity}|{subject_emotion}"] += 1
        writer.write(rendered)
        if preview_frame is None and (subjects or emotion_results or face_results):
            preview_frame = rendered.copy()

        processed_frames += 1

    cap.release()
    writer.release()

    dominant_emotion = emotion_counter.most_common(1)[0][0] if emotion_counter else 'no-face'
    dominant_identity = identity_counter.most_common(1)[0][0] if identity_counter else 'unknown'
    dominant_subject_key = subject_counter.most_common(1)[0][0] if subject_counter else 'unknown|no-face'
    dominant_subject_identity, dominant_subject_emotion = dominant_subject_key.split('|', 1)

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
        'dominant_subject': {
            'identity': dominant_subject_identity,
            'emotion': dominant_subject_emotion,
        },
        'emotion_distribution': dict(emotion_counter),
        'identity_distribution': dict(identity_counter),
        'subject_distribution': dict(subject_counter),
        'result_url': _absolute_media_url(out_name),
        'preview_url': preview_url,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)