"""
report_generator.py
Takes the emotion log (pandas DataFrame) from a session and produces a
polished, self-contained HTML report with:
  - summary cards (dominant emotion, total watch time, videos watched)
  - overall emotion distribution pie chart
  - emotion timeline per video (line chart over time)
  - dominant emotion per video (bar chart)
  - a plain-language insight paragraph
  - a detailed data table

Charts are rendered with matplotlib (Agg backend, no display needed)
and embedded directly into the HTML as base64 images, so the report
is a single portable .html file (open it in any browser, or print to PDF).
"""

import os
import base64
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import EMOTION_COLORS, EMOTION_EMOJI, ensure_dir, seconds_to_mmss


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _color_for(emotion):
    return EMOTION_COLORS.get(emotion, "#7f8c8d")


# ---------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------

def _pie_chart(df):
    counts = df[df["emotion"] != "no_face_detected"]["emotion"].value_counts()
    if counts.empty:
        return None
    colors = [_color_for(e) for e in counts.index]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        counts.values,
        labels=[f"{e.capitalize()}" for e in counts.index],
        autopct="%1.0f%%",
        colors=colors,
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax.set_title("Overall Emotion Distribution", fontsize=13, fontweight="bold")
    return _fig_to_base64(fig)


def _timeline_chart(df, video_name):
    vdf = df[(df["video"] == video_name) & (df["emotion"] != "no_face_detected")]
    if vdf.empty:
        return None
    emotions_order = list(EMOTION_COLORS.keys())
    y_pos = {e: i for i, e in enumerate(emotions_order)}

    fig, ax = plt.subplots(figsize=(9, 3))
    xs = vdf["video_time_sec"]
    ys = [y_pos.get(e, len(emotions_order)) for e in vdf["emotion"]]
    colors = [_color_for(e) for e in vdf["emotion"]]
    ax.scatter(xs, ys, c=colors, s=45, zorder=3)
    ax.plot(xs, ys, color="#cccccc", linewidth=1, zorder=1)

    ax.set_yticks(range(len(emotions_order)))
    ax.set_yticklabels([e.capitalize() for e in emotions_order], fontsize=9)
    ax.set_xlabel("Video time (seconds)", fontsize=9)
    ax.set_title(f"Emotion timeline — {video_name}", fontsize=11, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return _fig_to_base64(fig)


def _bar_chart_per_video(df):
    grouped = (
        df[df["emotion"] != "no_face_detected"]
        .groupby(["video", "emotion"])
        .size()
        .unstack(fill_value=0)
    )
    if grouped.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4))
    grouped.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=[_color_for(e) for e in grouped.columns],
    )
    ax.set_ylabel("Number of samples")
    ax.set_xlabel("")
    ax.set_title("Emotion breakdown per video", fontsize=12, fontweight="bold")
    ax.legend(title="Emotion", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    return _fig_to_base64(fig)


# ---------------------------------------------------------------------
# Insight text
# ---------------------------------------------------------------------

def _build_insight(df):
    valid = df[df["emotion"] != "no_face_detected"]
    if valid.empty:
        return ("No reliable facial-expression samples were captured during this session. "
                "Please review camera positioning, lighting, and face visibility.")

    counts = valid["emotion"].value_counts()
    dominant = counts.idxmax()
    dominant_pct = round(100 * counts.max() / len(valid))
    n_videos = df["video"].nunique()
    face_rate = round(100 * len(valid) / len(df)) if len(df) else 0
    diversity = int(valid["emotion"].nunique())

    if len(valid) > 1:
        changes = int((valid["emotion"].values[1:] != valid["emotion"].values[:-1]).sum())
        stability = round(100 * (1 - changes / max(1, len(valid)-1)))
    else:
        stability = 100

    lines = [
        f"The session was predominantly characterized by <b>{dominant.capitalize()} "
        f"{EMOTION_EMOJI.get(dominant,'')}</b>, representing approximately "
        f"<b>{dominant_pct}%</b> of valid expression samples.",
        f"The analysis covered <b>{n_videos}</b> video(s), with a face-visibility rate of "
        f"<b>{face_rate}%</b> and <b>{diversity}/7</b> expression categories observed.",
        f"Expression stability was measured at <b>{stability}%</b>, indicating how consistently "
        f"the detected expression remained over the captured samples.",
        "<b>Interpretation:</b> these results describe visible facial-expression patterns "
        "detected by the AI model. They are not a clinical diagnosis or a definitive "
        "measurement of a person's internal emotional state.",
    ]

    for v, sub in valid.groupby("video"):
        top = sub["emotion"].value_counts().idxmax()
        pct = round(100 * sub["emotion"].value_counts().max() / len(sub))
        lines.append(
            f"<b>{v}</b> — dominant observed expression: "
            f"<b>{top.capitalize()} {EMOTION_EMOJI.get(top,'')}</b> ({pct}%)."
        )
    return "<br><br>".join(lines)


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def generate_report(df: pd.DataFrame, session_info: dict, output_dir="reports"):
    """
    df: the raw emotion log dataframe from SessionManager
    session_info: dict with keys child_name, child_age, session_id,
                   session_start_time, session_end_time, videos (list)
    Returns: path to the generated .html file
    """
    ensure_dir(output_dir)

    pie_b64 = _pie_chart(df)
    bar_b64 = _bar_chart_per_video(df)
    timeline_charts = []
    for video in df["video"].dropna().unique():
        chart = _timeline_chart(df, video)
        if chart:
            timeline_charts.append((video, chart))

    insight_html = _build_insight(df)

    duration_sec = 0
    if session_info.get("session_start_time") and session_info.get("session_end_time"):
        duration_sec = session_info["session_end_time"] - session_info["session_start_time"]

    valid = df[df["emotion"] != "no_face_detected"]
    dominant = valid["emotion"].value_counts().idxmax() if not valid.empty else "N/A"
    face_visibility = round(100 * len(valid) / len(df)) if len(df) else 0
    diversity = int(valid["emotion"].nunique()) if not valid.empty else 0
    if len(valid) > 1:
        changes = int((valid["emotion"].values[1:] != valid["emotion"].values[:-1]).sum())
        stability = round(100 * (1 - changes / max(1, len(valid)-1)))
    else:
        stability = 100

    # table rows (limit to keep file size sane; full CSV also saved separately)
    table_df = df[["video", "video_time_sec", "emotion"]].copy()
    table_df["video_time_sec"] = table_df["video_time_sec"].apply(seconds_to_mmss)
    table_rows = "".join(
        f"<tr><td>{r.video}</td><td>{r.video_time_sec}</td>"
        f"<td><span class='pill' style='background:{_color_for(r.emotion)}'>"
        f"{EMOTION_EMOJI.get(r.emotion,'')} {r.emotion}</span></td></tr>"
        for r in table_df.itertuples()
    )

    timeline_html = "".join(
        f"""
        <div class="chart-card">
            <img src="data:image/png;base64,{chart}" alt="timeline chart"/>
        </div>
        """
        for _, chart in timeline_charts
    )

    html = f"""<!DOCTYPE html>
<html lang="hi">
<head>
<meta charset="UTF-8">
<title>Emotion Report — {session_info.get('child_name','')}</title>
<style>
    :root {{
        --primary: #6c5ce7;
        --primary-light: #a29bfe;
        --bg: #f5f6fa;
        --card-bg: #ffffff;
        --text: #2d3436;
        --muted: #636e72;
    }}
    * {{ box-sizing: border-box; }}
    body {{
        font-family: 'Segoe UI', Roboto, Arial, sans-serif;
        background: var(--bg);
        color: var(--text);
        margin: 0;
        padding: 0 0 60px 0;
    }}
    header {{
        background: linear-gradient(135deg, var(--primary), #00cec9);
        color: white;
        padding: 40px 30px;
        border-radius: 0 0 24px 24px;
        box-shadow: 0 4px 18px rgba(0,0,0,0.15);
    }}
    header h1 {{ margin: 0 0 6px 0; font-size: 28px; }}
    header p {{ margin: 2px 0; opacity: 0.95; }}
    .container {{ max-width: 1180px; margin: 0 auto; padding: 0 24px; }}
    .cards {{
        display: flex; gap: 18px; flex-wrap: wrap;
        margin-top: -30px;
    }}
    .card {{
        background: var(--card-bg);
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
        flex: 1 1 200px;
    }}
    .card .label {{ color: var(--muted); font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px;}}
    .card .value {{ font-size: 26px; font-weight: 700; margin-top: 6px; color: var(--primary); }}
    section {{ margin-top: 40px; }}
    section h2 {{
        font-size: 20px; border-left: 5px solid var(--primary);
        padding-left: 12px; margin-bottom: 16px;
    }}
    .insight-box {{
        background: #eef2ff; border-radius: 14px; padding: 20px 24px;
        line-height: 1.8; font-size: 15px;
    }}
    .chart-card {{
        background: var(--card-bg); border-radius: 16px; padding: 16px;
        box-shadow: 0 6px 16px rgba(0,0,0,0.06); margin-bottom: 20px;
        text-align: center;
    }}
    .chart-card img {{ max-width: 100%; border-radius: 8px; }}
    .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 800px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}
    table {{
        width: 100%; border-collapse: collapse; background: var(--card-bg);
        border-radius: 12px; overflow: hidden; font-size: 14px;
    }}
    th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; }}
    th {{ background: var(--primary); color: white; }}
    tr:hover {{ background: #f9f9ff; }}
    .pill {{
        color: white; padding: 3px 10px; border-radius: 999px;
        font-size: 12px; font-weight: 600; text-transform: capitalize;
    }}
    .quality {{
        background: #0f172a; color: #e2e8f0; border-radius: 16px;
        padding: 20px 24px; line-height: 1.7;
    }}
    .toolbar {{ text-align:right; margin-top:20px; }}
    .toolbar button {{
        border:0; border-radius:10px; padding:10px 16px;
        background:#6c5ce7; color:white; cursor:pointer; font-weight:600;
    }}
    @media print {{ .toolbar {{ display:none; }} }}
    footer {{
        text-align: center; color: var(--muted); margin-top: 50px; font-size: 12px;
    }}
</style>
</head>
<body>

<header>
    <div class="container">
        <h1>🎬 Emotion Response Report</h1>
        <p><b>Child:</b> {session_info.get('child_name','N/A')} &nbsp; | &nbsp;
           <b>Age:</b> {session_info.get('child_age','N/A')}</p>
        <p><b>Session ID:</b> {session_info.get('session_id','')} &nbsp; | &nbsp;
           <b>Generated:</b> {datetime.now().strftime('%d %b %Y, %I:%M %p')}</p>
    </div>
</header>

<div class="container">
    <div class="toolbar"><button onclick="window.print()">Print / Save as PDF</button></div>
    <div class="cards">
        <div class="card">
            <div class="label">Videos Watched</div>
            <div class="value">{df['video'].nunique()}</div>
        </div>
        <div class="card">
            <div class="label">Session Duration</div>
            <div class="value">{seconds_to_mmss(duration_sec)}</div>
        </div>
        <div class="card">
            <div class="label">Total Emotion Samples</div>
            <div class="value">{len(df)}</div>
        </div>
        <div class="card">
            <div class="label">Dominant Emotion</div>
            <div class="value">{EMOTION_EMOJI.get(dominant,'')} {str(dominant).capitalize()}</div>
        </div>
        <div class="card">
            <div class="label">Face Visibility</div>
            <div class="value">{face_visibility}%</div>
        </div>
        <div class="card">
            <div class="label">Expression Diversity</div>
            <div class="value">{diversity}/7</div>
        </div>
        <div class="card">
            <div class="label">Expression Stability</div>
            <div class="value">{stability}%</div>
        </div>
    </div>

    <section>
        <h2>📝 Summary & Insights</h2>
        <div class="insight-box">{insight_html}</div>
    </section>

    <section>
        <h2>📊 Overall Emotion Distribution</h2>
        <div class="grid-2">
            <div class="chart-card">
                {"<img src='data:image/png;base64," + pie_b64 + "'/>" if pie_b64 else "<p>No data</p>"}
            </div>
            <div class="chart-card">
                {"<img src='data:image/png;base64," + bar_b64 + "'/>" if bar_b64 else "<p>No data</p>"}
            </div>
        </div>
    </section>

    <section>
        <h2>⏱️ Emotion Timeline (per video)</h2>
        {timeline_html if timeline_html else "<p>No timeline data available.</p>"}
    </section>

    <section>
        <h2>🔎 Data Quality & Model Notes</h2>
        <div class="quality">
            <b>Face visibility:</b> {face_visibility}% of captured samples contained a reliable face.<br>
            <b>Expression diversity:</b> {diversity} of 7 supported expression classes were observed.<br>
            <b>Expression stability:</b> {stability}% based on consecutive valid samples.<br>
            <b>Model note:</b> confidence scores are model estimates and can be affected by
            lighting, camera angle, occlusion, image quality and facial pose.
        </div>
    </section>

    <section>
        <h2>📋 Detailed Log</h2>
        <table>
            <tr><th>Video</th><th>Time</th><th>Emotion</th></tr>
            {table_rows}
        </table>
    </section>

    <footer>
        Generated automatically by Emotion-Aware Video Session tool. For educational /
        research use only — not a clinical or diagnostic assessment.
    </footer>
</div>
</body>
</html>
"""

    out_path = os.path.join(output_dir, f"report_{session_info.get('session_id','')}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path
