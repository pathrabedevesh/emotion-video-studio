"""
video_player.py
Thin wrapper around python-vlc so a video (with audio) can be embedded
inside a Tkinter frame and controlled (play / stop / get current time).

Requires VLC media player to be installed on the system
(python-vlc is just bindings to the real libvlc engine).
Download: https://www.videolan.org/vlc/
"""

import sys
import os
os.add_dll_directory(r"C:\Program Files\VideoLAN\VLC")

import vlc


class VideoPlayer:
    def __init__(self, tk_frame):
        """
        tk_frame: a Tkinter Frame/Canvas widget the video will be rendered into.
        """
        self.tk_frame = tk_frame
        self.instance = vlc.Instance("--no-xlib") if sys.platform.startswith("linux") else vlc.Instance()
        self.player = self.instance.media_player_new()
        self._embed()

        self._ended_callback = None
        events = self.player.event_manager()
        events.event_attach(vlc.EventType.MediaPlayerEndReached, self._on_end)

    def _embed(self):
        handle = self.tk_frame.winfo_id()
        if sys.platform.startswith("linux"):
            self.player.set_xwindow(handle)
        elif sys.platform == "win32":
            self.player.set_hwnd(handle)
        elif sys.platform == "darwin":
            self.player.set_nsobject(handle)

    def load(self, video_path):
        media = self.instance.media_new(video_path)
        self.player.set_media(media)

    def play(self):
        self.player.play()

    def pause(self):
        self.player.pause()

    def stop(self):
        self.player.stop()

    def is_playing(self):
        return self.player.is_playing()

    def get_time_seconds(self):
        """Current playback position in seconds (float)."""
        ms = self.player.get_time()
        return max(0.0, ms / 1000.0)

    def get_length_seconds(self):
        ms = self.player.get_length()
        return max(0.0, ms / 1000.0)

    def set_ended_callback(self, callback):
        """callback() is invoked (from a background thread) when playback ends."""
        self._ended_callback = callback

    def _on_end(self, event):
        if self._ended_callback:
            self._ended_callback()

    def release(self):
        self.player.stop()
        self.player.release()
