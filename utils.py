"""
utils.py
Common helper functions and constants used across the project.
"""

import os
import time
import glob

# ---- Config ----------------------------------------------------------

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

EMOTION_COLORS = {
    "happy": "#2ecc71",
    "sad": "#3498db",
    "angry": "#e74c3c",
    "fear": "#9b59b6",
    "surprise": "#f39c12",
    "disgust": "#16a085",
    "neutral": "#95a5a6",
}

EMOTION_EMOJI = {
    "happy": "😊",
    "sad": "😢",
    "angry": "😠",
    "fear": "😨",
    "surprise": "😮",
    "disgust": "🤢",
    "neutral": "😐",
}


# ---- Helpers -----------------------------------------------------------

def list_videos(folder_path):
    """Return a sorted list of video file paths found in a folder."""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob.glob(os.path.join(folder_path, f"*{ext}")))
        videos.extend(glob.glob(os.path.join(folder_path, f"*{ext.upper()}")))
    return sorted(set(videos))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def timestamp_str():
    return time.strftime("%Y%m%d_%H%M%S")


def seconds_to_mmss(seconds):
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    return f"{m:02d}:{s:02d}"
