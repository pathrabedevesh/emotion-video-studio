"""
main.py
Emotion-Aware Video Session — main application.

Flow:
  1. Welcome screen: enter child's name/age, pick a folder of videos.
  2. Session screen: videos play one by one (with audio) via VLC,
     embedded in the window. A small live webcam preview + detected
     emotion label is shown in a side panel while each video plays.
     Emotion samples are logged automatically every second.
  3. When the last video finishes (or user clicks "Finish & Report"),
     a polished HTML report is generated and opened in the browser.

Run:  python main.py
"""

import os
import sys
import time
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime

import cv2
from PIL import Image, ImageTk

from utils import list_videos, EMOTION_EMOJI, EMOTION_COLORS
from emotion_detector import EmotionDetector
from video_player import VideoPlayer
from session_manager import SessionManager
from report_generator import generate_report


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Emotion-Aware Video Session")
        self.geometry("1280x760")
        self.minsize(1050, 680)
        self.configure(bg="#0b1220")

        self.video_folder = None
        self.video_list = []
        self.current_index = 0

        self.session = None
        self.detector = None
        self.player = None
        self._preview_running = False
        self._paused = False
        self._emotion_history = []
        self._snapshot_dir = "snapshots"

        self._build_welcome_screen()

    # ------------------------------------------------------------------
    # Screen 1: Welcome / setup
    # ------------------------------------------------------------------

    def _build_welcome_screen(self):
        self._clear()
        bg, card, accent, muted, white = "#0b1220", "#111c2e", "#6c63ff", "#94a3b8", "#f8fafc"
        wrap = tk.Frame(self, bg=bg)
        wrap.pack(fill="both", expand=True)

        tk.Label(wrap, text="EMOTION • VISION", font=("Segoe UI", 10, "bold"),
                 bg=bg, fg="#8b85ff").pack(pady=(55, 10))
        tk.Label(wrap, text="Emotion-Aware Video Studio", font=("Segoe UI", 30, "bold"),
                 bg=bg, fg=white).pack()
        tk.Label(wrap, text="Real-time facial-expression analysis • Live monitoring • Professional reports",
                 font=("Segoe UI", 11), bg=bg, fg=muted).pack(pady=(8, 32))

        panel = tk.Frame(wrap, bg=card, highlightthickness=1, highlightbackground="#24324a")
        panel.pack(padx=70, ipadx=30, ipady=25)

        form = tk.Frame(panel, bg=card)
        form.pack(padx=35, pady=25)
        labels = [("CHILD NAME", 0), ("AGE", 1)]
        for text, row in labels:
            tk.Label(form, text=text, font=("Segoe UI", 9, "bold"), bg=card, fg=muted).grid(
                row=row, column=0, sticky="w", pady=(8, 4))
        self.name_entry = tk.Entry(form, font=("Segoe UI", 11), width=32,
                                   bg="#0b1220", fg=white, insertbackground=white,
                                   relief="flat")
        self.name_entry.grid(row=0, column=1, padx=(22,0), pady=5)
        self.age_entry = tk.Entry(form, font=("Segoe UI", 11), width=32,
                                  bg="#0b1220", fg=white, insertbackground=white,
                                  relief="flat")
        self.age_entry.grid(row=1, column=1, padx=(22,0), pady=5)

        tk.Label(form, text="VIDEO LIBRARY", font=("Segoe UI", 9, "bold"),
                 bg=card, fg=muted).grid(row=2, column=0, sticky="w", pady=(18,4))
        folder_row = tk.Frame(form, bg=card)
        folder_row.grid(row=2, column=1, sticky="ew", padx=(22,0), pady=10)
        self.folder_label = tk.Label(folder_row, text="No folder selected",
                                      font=("Segoe UI", 9), bg=card, fg=muted)
        self.folder_label.pack(side="left")
        tk.Button(folder_row, text="  Browse  ", command=self._pick_folder,
                  bg="#24324a", fg=white, relief="flat", padx=10, pady=5).pack(side="right")

        self.start_btn = tk.Button(
            wrap, text="  START ANALYSIS  →  ", font=("Segoe UI", 12, "bold"),
            bg=accent, fg="white", activebackground="#554ee8", activeforeground="white",
            relief="flat", padx=22, pady=12, command=self._start_session, state="disabled",
        )
        self.start_btn.pack(pady=30)

        feature_bar = tk.Frame(wrap, bg=bg)
        feature_bar.pack(pady=(6, 0))
        for text in ("●  Live AI", "●  7 Emotions", "●  Session Analytics", "●  Export JSON/CSV"):
            tk.Label(feature_bar, text=text, font=("Segoe UI", 8, "bold"),
                     bg=bg, fg="#64748b").pack(side="left", padx=12)

        tk.Label(wrap, text="Professional analysis suite • expression trends • confidence scoring • reaction snapshots",
                 font=("Segoe UI", 9), bg=bg, fg="#475569").pack(pady=(12, 0))

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing videos")
        if not folder:
            return
        videos = list_videos(folder)
        if not videos:
            messagebox.showwarning("No videos found", "Is folder mein koi video file nahi mili (.mp4/.mov/.avi/.mkv/.webm)")
            return
        self.video_folder = folder
        self.video_list = videos
        self.folder_label.config(text=f"{len(videos)} video(s) found")
        self.start_btn.config(state="normal")

    # ------------------------------------------------------------------
    # Screen 2: Session (video + live emotion panel)
    # ------------------------------------------------------------------

    def _start_session(self):
        name = self.name_entry.get().strip() or "Unknown"
        age = self.age_entry.get().strip()

        self.session = SessionManager(child_name=name, child_age=age)
        self.session.start_session()
        self.current_index = 0

        try:
            # Faster sampling + built-in temporal smoothing prevents
            # emotion labels from jumping every frame.
            self.detector = EmotionDetector(
                sample_interval=0.35,
                smoothing_window=4,
                min_confidence=0.28,
            )
            self.detector.on_sample = self._on_emotion_sample
            self.detector.start()
        except Exception as e:
            messagebox.showerror("Webcam / model error", str(e))
            return

        self._build_session_screen()
        self._play_current_video()
        self._update_preview_loop()
        self._refresh_live_analytics()

    def _build_session_screen(self):
        self._clear()
        bg, panel, accent, white, muted = "#0b1220", "#111c2e", "#6c63ff", "#f8fafc", "#94a3b8"

        top = tk.Frame(self, bg=panel, height=64)
        top.pack(fill="x")
        tk.Label(top, text="EMOTION • LIVE", font=("Segoe UI", 11, "bold"),
                 bg=panel, fg="#8b85ff").pack(side="left", padx=22)
        self.video_title_label = tk.Label(top, text="", font=("Segoe UI", 11),
                                          bg=panel, fg=white)
        self.video_title_label.pack(side="left", padx=15)
        self.pause_btn = tk.Button(top, text="⏸ Pause Analysis", command=self._toggle_pause,
                  bg="#24324a", fg=white, relief="flat", padx=12, pady=7)
        self.pause_btn.pack(side="right", padx=4)
        tk.Button(top, text="📸 Snapshot", command=self._capture_snapshot,
                  bg="#24324a", fg=white, relief="flat", padx=12, pady=7).pack(side="right", padx=4)
        tk.Button(top, text="Skip →", command=self._next_video,
                  bg="#24324a", fg=white, relief="flat", padx=12, pady=7).pack(side="right", padx=4)
        tk.Button(top, text="Finish & Report", command=self._finish_session,
                  bg="#e05252", fg="white", relief="flat", padx=12, pady=7).pack(side="right", padx=8)

        body = tk.Frame(self, bg=bg)
        body.pack(fill="both", expand=True, padx=14, pady=14)

        self.video_frame = tk.Frame(body, bg="black", highlightthickness=1,
                                    highlightbackground="#24324a")
        self.video_frame.pack(side="left", fill="both", expand=True, padx=(0,14))
        self.player = VideoPlayer(self.video_frame)
        self.player.set_ended_callback(self._on_video_end)

        side = tk.Frame(body, bg=panel, width=315, highlightthickness=1,
                        highlightbackground="#24324a")
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        tk.Label(side, text="LIVE FACIAL ANALYSIS", font=("Segoe UI", 10, "bold"),
                 bg=panel, fg=muted).pack(pady=(18,8))
        self.webcam_canvas = tk.Label(side, bg="#050912")
        self.webcam_canvas.pack(padx=16, pady=2)

        self.emotion_label = tk.Label(side, text="—", font=("Segoe UI", 26, "bold"),
                                      bg=panel, fg=white)
        self.emotion_label.pack(pady=(20,3))
        tk.Label(side, text="CURRENT EXPRESSION", font=("Segoe UI", 8, "bold"),
                 bg=panel, fg=muted).pack()

        self.confidence_label = tk.Label(side, text="Confidence  —",
                                         font=("Segoe UI", 10), bg=panel, fg=muted)
        self.confidence_label.pack(pady=(14,2))

        self.score_label = tk.Label(
            side, text="All emotion scores\n—", justify="left",
            font=("Consolas", 9), bg=panel, fg="#cbd5e1"
        )
        self.score_label.pack(pady=(10, 4), padx=18, anchor="w")
        self.backend_label = tk.Label(side, text=f"Engine: {getattr(self.detector,'backend_name','Emotion Model')}",
                                      font=("Segoe UI", 8), bg=panel, fg="#64748b")
        self.backend_label.pack(pady=(2, 4))

        self.face_status = tk.Label(side, text="● Face quality: checking…",
                                    font=("Segoe UI", 9, "bold"), bg=panel, fg="#94a3b8")
        self.face_status.pack(pady=3)
        self.live_stats = tk.Label(side, text="Samples: 0  •  Expressions: 0",
                                   font=("Segoe UI", 8), bg=panel, fg="#64748b")
        self.live_stats.pack(pady=2)

        tk.Label(side, text="EMOTION TREND", font=("Segoe UI", 8, "bold"),
                 bg=panel, fg=muted).pack(pady=(12,2))
        self.trend_canvas = tk.Canvas(side, width=280, height=75, bg="#0b1220",
                                      highlightthickness=0)
        self.trend_canvas.pack(padx=16, pady=4)

        self.analytics_label = tk.Label(side, text="Session stability: —",
                                        font=("Segoe UI", 9), bg=panel, fg="#cbd5e1")
        self.analytics_label.pack(pady=(8,2))

        self.progress_label = tk.Label(side, text="", font=("Segoe UI", 9),
                                       bg=panel, fg=muted)
        self.progress_label.pack(side="bottom", pady=18)

    def _play_current_video(self):
        if self.current_index >= len(self.video_list):
            self._finish_session()
            return
        path = self.video_list[self.current_index]
        name = os.path.basename(path)
        self.video_title_label.config(text=f"Now playing: {name}")
        self.progress_label.config(text=f"Video {self.current_index + 1} of {len(self.video_list)}")
        self.session.start_video(name)
        self.player.load(path)
        self.player.play()
        self._poll_video_time()

    def _poll_video_time(self):
        """Periodically read current video time so emotion samples can be
        tagged with the correct in-video timestamp."""
        if self.player is None:
            return
        self._current_video_time = self.player.get_time_seconds()
        self.after(500, self._poll_video_time)

    def _on_video_end(self):
        # called from a VLC background thread -> hop back to main thread
        self.after(0, self._next_video)

    def _next_video(self):
        self.current_index += 1
        if self.current_index >= len(self.video_list):
            self._finish_session()
        else:
            self._play_current_video()

    # ------------------------------------------------------------------
    # Live emotion + webcam preview
    # ------------------------------------------------------------------

    def _on_emotion_sample(self, emotion, scores, ts):
        video_time = getattr(self, "_current_video_time", 0.0)
        self.session.log_sample(emotion, scores, video_time)
        if emotion:
            self._emotion_history.append(emotion)
            self._emotion_history = self._emotion_history[-40:]

        def update_label():
            if emotion:
                emoji = EMOTION_EMOJI.get(emotion, "")
                color = EMOTION_COLORS.get(emotion, "#636e72")
                self.emotion_label.config(text=f"{emoji} {emotion.capitalize()}", fg=color)
                conf = max(scores.values()) * 100 if scores else 0
                self.confidence_label.config(text=f"Confidence  {conf:.0f}%")
                if scores and hasattr(self, "score_label"):
                    order = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
                    self.score_label.config(text="\n".join(
                        f"{e.title():9s} {scores.get(e, 0)*100:5.1f}%"
                        for e in order
                    ))
                valid = sum(1 for x in self._emotion_history if x)
                self.live_stats.config(text=f"Samples: {len(self.session.records)}  •  Expressions: {len(set(self._emotion_history))}")
                self.face_status.config(text="● Face quality: GOOD", fg="#2ecc71")
                self._draw_trend()
            else:
                self.emotion_label.config(text="No face detected", fg="#b2bec3")
        self.after(0, update_label)

    def _update_preview_loop(self):
        """Show a small live webcam thumbnail in the side panel."""
        if self.detector is None:
            return

        # IMPORTANT: do not read the webcam here. The detector thread owns
        # the camera and shares its newest frame with the GUI.
        frame = self.detector.get_latest_frame()
        if frame is not None:
            frame = cv2.resize(frame, (280, 210))

            emotion, scores, has_face = self.detector.get_latest()
            if has_face and emotion:
                cv2.putText(
                    frame, f"{emotion.capitalize()}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 255, 255), 2, cv2.LINE_AA
                )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.webcam_canvas.imgtk = img
            self.webcam_canvas.config(image=img)

        self.after(100, self._update_preview_loop)

    # ------------------------------------------------------------------
    # Professional live controls / analytics
    # ------------------------------------------------------------------

    def _toggle_pause(self):
        if not self.detector or not self.player:
            return
        self._paused = not self._paused
        self.detector.set_paused(self._paused)
        if self._paused:
            self.player.pause()
            self.pause_btn.config(text="▶ Resume Analysis", bg="#2d7d5a")
        else:
            self.player.play()
            self.pause_btn.config(text="⏸ Pause Analysis", bg="#24324a")

    def _capture_snapshot(self):
        if not self.detector:
            return
        frame = self.detector.get_latest_frame()
        if frame is None:
            messagebox.showwarning("Snapshot", "Live camera frame is not available yet.")
            return
        os.makedirs(self._snapshot_dir, exist_ok=True)
        emotion, scores, has_face = self.detector.get_latest()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = emotion or "no_face"
        filename = os.path.join(self._snapshot_dir, f"{self.session.session_id}_{stamp}_{label}.jpg")
        cv2.putText(frame, f"Emotion: {label}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
        cv2.imwrite(filename, frame)
        messagebox.showinfo("Snapshot Saved", f"Saved:\n{os.path.abspath(filename)}")

    def _draw_trend(self):
        if not hasattr(self, "trend_canvas"):
            return
        c = self.trend_canvas
        c.delete("all")
        history = self._emotion_history[-30:]
        if not history:
            c.create_text(140, 38, text="Waiting for emotion samples…", fill="#64748b")
            return
        colors = EMOTION_COLORS
        w, h = 280, 75
        for i, emotion in enumerate(history):
            x = 8 + i * (w-16) / max(1, len(history)-1)
            y = h - 12 - (list(EMOTION_COLORS.keys()).index(emotion) / 6.0) * (h-24)
            if i:
                pe = history[i-1]
                px = 8 + (i-1) * (w-16) / max(1, len(history)-1)
                py = h - 12 - (list(EMOTION_COLORS.keys()).index(pe) / 6.0) * (h-24)
                c.create_line(px, py, x, y, fill="#475569", width=2)
            c.create_oval(x-3, y-3, x+3, y+3, fill=colors.get(emotion, "#94a3b8"), outline="")

    def _refresh_live_analytics(self):
        if self.session:
            a = self.session.analytics()
            self.analytics_label.config(
                text=f"Visibility {a['face_visibility']:.0f}%  •  Stability {a['stability']:.0f}%"
            )
        self.after(1000, self._refresh_live_analytics)

    # ------------------------------------------------------------------
    # Finish + report
    # ------------------------------------------------------------------

    def _finish_session(self):
        if self.player:
            self.player.release()
        if self.detector:
            self.detector.stop()
        self.session.end_session()

        df = self.session.as_dataframe()
        if df.empty:
            messagebox.showinfo("No data", "Koi emotion data record nahi hua. Report skip.")
            self._build_welcome_screen()
            return

        session_info = {
            "child_name": self.session.child_name,
            "child_age": self.session.child_age,
            "session_id": self.session.session_id,
            "session_start_time": self.session.session_start_time,
            "session_end_time": self.session.session_end_time,
        }
        report_path = generate_report(df, session_info)
        csv_path = self.session.save_raw_log()
        json_path = self.session.export_json()
        webbrowser.open(f"file://{os.path.abspath(report_path)}")
        messagebox.showinfo(
            "Analysis Complete",
            "Professional report generated successfully.\n\n"
            f"HTML Report: {report_path}\n"
            f"CSV Data: {csv_path}\n"
            f"JSON Data: {json_path}"
        )
        self._build_welcome_screen()

    # ------------------------------------------------------------------

    def _clear(self):
        for w in self.winfo_children():
            w.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
