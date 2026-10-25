"""
generate_sample_videos.py
Creates 3 short synthetic test videos WITH audio in sample_videos/, so you
can run the whole project end-to-end immediately without needing real
video files.

These are simple colour + shape animations with a musical tone track —
NOT copyrighted cartoon content. For real sessions with children, replace
these with age-appropriate videos of your choice (cartoons, stories,
educational clips etc.) in the same folder.

Run: python generate_sample_videos.py
"""

import os
import numpy as np
from moviepy.editor import VideoClip, AudioClip
from utils import ensure_dir

OUT_DIR = ensure_dir("sample_videos")


def make_video(filename, duration, bg_color_fn, tone_freq):
    def make_frame(t):
        h, w = 480, 640
        color = bg_color_fn(t)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = color
        # a pulsing circle in the middle for visual movement
        cy, cx = h // 2, w // 2
        r = int(60 + 40 * np.sin(2 * np.pi * 0.5 * t))
        yy, xx = np.ogrid[:h, :w]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        frame[mask] = [255, 255, 255]
        return frame

    def make_audio(t):
        # simple sine tone, gentle volume, works for scalar or array t
        return 0.2 * np.sin(2 * np.pi * tone_freq * t)

    video = VideoClip(make_frame, duration=duration)
    audio = AudioClip(make_audio, duration=duration, fps=44100)
    video = video.set_audio(audio)
    path = os.path.join(OUT_DIR, filename)
    video.write_videofile(path, fps=24, codec="libx264", audio_codec="aac", logger=None)
    print(f"Created: {path}")


if __name__ == "__main__":
    make_video(
        "sample_calm_blue.mp4", duration=8,
        bg_color_fn=lambda t: (30, 60, 120), tone_freq=220,
    )
    make_video(
        "sample_bright_yellow.mp4", duration=8,
        bg_color_fn=lambda t: (240, 200, 30), tone_freq=440,
    )
    make_video(
        "sample_energetic_red.mp4", duration=8,
        bg_color_fn=lambda t: (200, 40, 40), tone_freq=330,
    )
    print("\nDone! 3 sample videos with audio created in sample_videos/")
    print("Ab 'python main.py' chalao aur is folder ko select karo.")
