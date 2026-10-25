"""
emotion_detector.py
Professional real-time facial-expression detector.

Primary backend: DeepFace emotion analysis.
Fallback backend: FER, if DeepFace is unavailable.

The detector uses:
- original-color face analysis (no grayscale conversion)
- largest/highest-quality face selection
- temporal EMA smoothing
- class-specific confidence gates
- short no-face grace period
- shared camera frame for the GUI
"""
import threading
import time
from collections import deque
import cv2
import numpy as np

try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except Exception:
    DeepFace = None
    HAS_DEEPFACE = False

try:
    from fer import FER
except Exception:
    FER = None


EMOTIONS = ("angry", "disgust", "fear", "happy", "sad", "surprise", "neutral")


class EmotionDetector:
    def __init__(
        self,
        camera_index=0,
        sample_interval=0.35,
        show_preview=False,
        smoothing_window=4,
        min_confidence=0.28,
    ):
        self.camera_index = camera_index
        self.sample_interval = sample_interval
        self.show_preview = show_preview
        self.smoothing_window = max(2, smoothing_window)
        self.min_confidence = min_confidence

        self._cap = None
        self._fer = None
        self._thread = None
        self._running = False
        self._lock = threading.Lock()

        self.latest_emotion = None
        self.latest_scores = {}
        self.latest_frame_has_face = False
        self.latest_frame = None
        self.latest_box = None

        self._history = deque(maxlen=self.smoothing_window)
        self._last_valid_emotion = None
        self._last_valid_scores = {}
        self._last_face_time = 0.0
        self._paused = False

        self.on_sample = None
        self.backend_name = "DeepFace" if HAS_DEEPFACE else "FER fallback"

    def start(self):
        if not HAS_DEEPFACE and FER is None:
            raise RuntimeError(
                "No emotion engine found. Install requirements.txt first."
            )

        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permissions.")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap.set(cv2.CAP_PROP_FPS, 30)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not HAS_DEEPFACE:
            self._fer = FER(mtcnn=True)

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def set_paused(self, paused=True):
        self._paused = bool(paused)

    def is_paused(self):
        return self._paused

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
        if self.show_preview:
            cv2.destroyAllWindows()

    def _loop(self):
        last_sample = 0.0
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            now = time.time()
            with self._lock:
                self.latest_frame = frame.copy()

            if not self._paused and now - last_sample >= self.sample_interval:
                last_sample = now
                self._analyze(frame, now)

            time.sleep(0.005)

    @staticmethod
    def _normalize_scores(raw):
        scores = {e: 0.0 for e in EMOTIONS}
        for k, v in (raw or {}).items():
            k = str(k).lower()
            if k in scores:
                scores[k] = float(v) / 100.0 if float(v) > 1.0 else float(v)
        total = sum(scores.values())
        if total > 0:
            # DeepFace is already percentages, but normalization makes the
            # confidence values consistent across backends.
            scores = {k: v / total for k, v in scores.items()}
        return scores

    def _deepface(self, frame):
        # DeepFace internally performs face detection + emotion classification.
        result = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            align=True,
            silent=True,
        )
        if isinstance(result, dict):
            result = [result]
        valid = []
        for item in result or []:
            scores = self._normalize_scores(item.get("emotion", {}))
            region = item.get("region") or {}
            area = max(1, int(region.get("w", 0))) * max(1, int(region.get("h", 0)))
            if scores and area > 1000:
                valid.append((area, scores, region))
        if not valid:
            return None, {}, None
        _, scores, region = max(valid, key=lambda x: x[0])
        return scores, scores, region

    def _fer_predict(self, frame):
        results = self._fer.detect_emotions(frame)
        if not results:
            return None, {}, None
        face = max(
            results,
            key=lambda r: max(1, r["box"][2]) * max(1, r["box"][3])
        )
        return (
            self._normalize_scores(face.get("emotions", {})),
            self._normalize_scores(face.get("emotions", {})),
            face.get("box"),
        )

    def _analyze(self, frame, ts):
        try:
            if HAS_DEEPFACE:
                _, raw_scores, box = self._deepface(frame)
            else:
                _, raw_scores, box = self._fer_predict(frame)
        except Exception:
            raw_scores, box = {}, None

        has_face = bool(raw_scores)
        emotion = None
        scores = {}

        if raw_scores:
            self._history.append(raw_scores)
            # Exponential weighting: recent expressions have more influence.
            weights = np.arange(1, len(self._history) + 1, dtype=float)
            weights /= weights.sum()
            scores = {}
            for e in EMOTIONS:
                scores[e] = float(sum(w * h.get(e, 0.0)
                                      for w, h in zip(weights, self._history)))

            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top, confidence = ranked[0]
            second = ranked[1][1]

            # Neutral gets a higher gate. Expressive classes are allowed through
            # when they are clearly above their runner-up.
            # Expression-sensitive gates. Sad/Fear/Disgust are allowed through
            # with lower gates, while Neutral must clearly dominate to win.
            gates = {
                "angry": 0.18,
                "disgust": 0.14,
                "fear": 0.16,
                "happy": 0.22,
                "sad": 0.17,
                "surprise": 0.17,
                "neutral": 0.42,
            }

            neutral_score = scores.get("neutral", 0.0)
            expressive = {"angry", "disgust", "fear", "sad", "surprise", "happy"}

            if top in expressive:
                expression_margin = 0.035 if top in {"disgust", "fear", "sad"} else 0.045
                accepted = (
                    confidence >= gates[top]
                    and confidence >= neutral_score + expression_margin
                )
            else:
                best_expression = max((scores[e] for e in expressive), default=0.0)
                accepted = (
                    confidence >= gates["neutral"]
                    and confidence >= best_expression + 0.075
                )

            if accepted:
                emotion = top
                self._last_valid_emotion = emotion
                self._last_valid_scores = dict(scores)
                self._last_face_time = ts
            elif self._last_valid_emotion and ts - self._last_face_time <= 1.0:
                emotion = self._last_valid_emotion
                scores = self._last_valid_scores
            else:
                emotion = "neutral" if confidence >= 0.30 else None

        elif self._last_valid_emotion and ts - self._last_face_time <= 1.0:
            emotion = self._last_valid_emotion
            scores = self._last_valid_scores
            has_face = True

        with self._lock:
            self.latest_emotion = emotion
            self.latest_scores = dict(scores)
            self.latest_frame_has_face = has_face
            self.latest_box = box

        if self.on_sample:
            self.on_sample(emotion, dict(scores), ts)

    def get_latest(self):
        with self._lock:
            return (
                self.latest_emotion,
                dict(self.latest_scores),
                self.latest_frame_has_face,
            )

    def get_emotion_scores(self):
        with self._lock:
            return dict(self.latest_scores)

    def get_latest_frame(self):
        with self._lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def get_latest_box(self):
        with self._lock:
            return self.latest_box
