"""
session_manager.py
Ties video playback + emotion detection together, and keeps a log
of every emotion sample tagged with (video_name, video_time, emotion, scores).
"""

import os
import time
import pandas as pd
import json

from utils import ensure_dir, timestamp_str


class SessionManager:
    def __init__(self, child_name="Unknown", child_age="", log_dir="logs"):
        self.child_name = child_name
        self.child_age = child_age
        self.log_dir = ensure_dir(log_dir)
        self.session_id = timestamp_str()

        self.records = []          # list of dicts, one per emotion sample
        self.session_start_time = None
        self.session_end_time = None
        self.current_video = None
        self._video_start_wall_time = None
        self.paused_seconds = 0.0

    # ------------------------------------------------------------------

    def start_session(self):
        self.session_start_time = time.time()

    def end_session(self):
        self.session_end_time = time.time()
        self.save_raw_log()
        self.export_json()

    def start_video(self, video_name):
        self.current_video = video_name
        self._video_start_wall_time = time.time()

    def log_sample(self, emotion, scores, video_time_sec):
        if emotion is None:
            emotion = "no_face_detected"
        row = {
            "session_id": self.session_id,
            "child_name": self.child_name,
            "child_age": self.child_age,
            "video": self.current_video,
            "video_time_sec": round(video_time_sec, 2),
            "wall_time": time.time(),
            "emotion": emotion,
        }
        # flatten per-emotion confidence scores as separate columns
        for emo, val in scores.items():
            row[f"score_{emo}"] = val
        self.records.append(row)

    # ------------------------------------------------------------------

    def as_dataframe(self):
        return pd.DataFrame(self.records)


    def analytics(self):
        df = self.as_dataframe()
        if df.empty:
            return {
                "face_visibility": 0.0,
                "expression_diversity": 0,
                "stability": 0.0,
                "dominant_emotion": "N/A",
                "samples": 0,
            }
        valid = df[df["emotion"] != "no_face_detected"]
        visibility = 100.0 * len(valid) / len(df)
        diversity = int(valid["emotion"].nunique()) if not valid.empty else 0
        dominant = valid["emotion"].value_counts().idxmax() if not valid.empty else "N/A"
        if len(valid) > 1:
            changes = (valid["emotion"].values[1:] != valid["emotion"].values[:-1]).sum()
            stability = max(0.0, 100.0 * (1.0 - changes / max(1, len(valid)-1)))
        else:
            stability = 100.0
        return {
            "face_visibility": round(visibility, 1),
            "expression_diversity": diversity,
            "stability": round(stability, 1),
            "dominant_emotion": dominant,
            "samples": int(len(df)),
        }

    def export_json(self, path=None):
        df = self.as_dataframe()
        if path is None:
            path = os.path.join(self.log_dir, f"session_{self.session_id}.json")
        payload = {
            "session_id": self.session_id,
            "child_name": self.child_name,
            "child_age": self.child_age,
            "session_start_time": self.session_start_time,
            "session_end_time": self.session_end_time,
            "analytics": self.analytics(),
            "records": df.to_dict(orient="records"),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        return path

    def save_raw_log(self):
        df = self.as_dataframe()
        if df.empty:
            return None
        path = os.path.join(self.log_dir, f"session_{self.session_id}.csv")
        df.to_csv(path, index=False)
        return path
