from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from statsbomb_toolkit.sber_exports import viz

st.set_page_config(
    page_title="Тифлокомментарии Euro-2024",
    layout="wide",
    initial_sidebar_state="expanded",
)

INDEX_CANDIDATES = [
    ROOT / "outputs" / "euro2024_top15" / "processed_json" / "index.csv",
    ROOT / "outputs" / "euro2024_all" / "processed_json" / "index_to_speak_top15.csv",
    ROOT / "outputs" / "euro2024_all" / "processed_json" / "index_all.csv",
]

MANIFEST_CANDIDATES = [
    ROOT / "outputs" / "euro2024_all" / "euro2024_manifest_to_speak_top15.csv",
    ROOT / "outputs" / "euro2024_all" / "euro2024_manifest_all.csv",
]

VIDEO_TRACK_CANDIDATES = [
    APP_DIR / "My Movie 1.mp4",
    APP_DIR / "match_3930158_video.mp4",
    APP_DIR / "germany_scotland_7min.mp4",
]

AUDIO_TRACK_CANDIDATES = {
    "Женский голос": [
        APP_DIR / "match_3930158_related_stop_cap_full_with_noise (1).wav",
        APP_DIR / "match_3930158_related_stop_cap_period1_with_noise (1).wav",
    ],
    "Мужской голос": [
        APP_DIR / "match_3930158_related_stop_cap_full_with_noise.wav",
        APP_DIR / "match_3930158_related_stop_cap_period1_with_noise.wav",
    ],
}

LOGO_CANDIDATES = [
    APP_DIR / "assets" / "statsbomb_logo.png",
    APP_DIR / "assets" / "statsbomb_logo.svg",
    APP_DIR / "statsbomb_logo.svg",
    APP_DIR / "statsbomb_logo.png",
]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800;900&display=swap');

:root {
    --bg: #10110f;
    --panel: #191a16;
    --cream: #f6f1e8;
    --paper: #fffaf1;
    --ink: #171914;
    --muted: #6f7568;
    --grass: #2f8f3a;
    --grass-2: #17692b;
    --line: #bfe8c5;
    --rust: #2f8f3a;
    --gold: #7fd28a;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
}

.stApp {
    background:
      radial-gradient(circle at 16% 8%, rgba(127, 210, 138, 0.16), transparent 28%),
      radial-gradient(circle at 90% 18%, rgba(47, 143, 58, 0.22), transparent 26%),
      linear-gradient(135deg, #10110f 0%, #151712 50%, #0d0f0c 100%);
}

.block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.4rem;
    max-width: 1460px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #23241f 0%, #191a16 100%);
    border-right: 1px solid rgba(191, 232, 197, 0.20);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p {
    color: #f6f1e8 !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: 1px solid rgba(191, 232, 197, 0.20);
    padding-top: 10px;
    margin-bottom: 22px;
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    min-height: 44px;
    padding: 10px 18px;
    border-radius: 999px 999px 0 0;
    color: #d8d0bf;
    font-weight: 900;
    font-size: 16px;
}

.stTabs [aria-selected="true"] {
    color: #fffaf1 !important;
    background: rgba(47, 143, 58, 0.22);
}

.stTabs [data-baseweb="tab-highlight"] {
    background-color: #7fd28a;
    height: 4px;
}

.hero {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(191, 232, 197, 0.35);
    border-radius: 30px;
    padding: 34px 38px;
    background:
      linear-gradient(90deg, rgba(25,26,22,0.96), rgba(25,26,22,0.82)),
      repeating-linear-gradient(90deg, rgba(47,143,58,0.32) 0 90px, rgba(23,105,43,0.32) 90px 180px);
    box-shadow: 0 22px 60px rgba(0,0,0,0.28);
    margin-bottom: 22px;
}

.hero::before {
    content: "";
    position: absolute;
    inset: 18px;
    border: 1px solid rgba(246, 241, 232, 0.16);
    border-radius: 22px;
    pointer-events: none;
}

.hero::after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    border: 2px solid rgba(246, 241, 232, 0.14);
    border-radius: 50%;
    right: 70px;
    top: 50%;
    transform: translateY(-50%);
    pointer-events: none;
}

.hero h1 {
    position: relative;
    margin: 0 0 14px 0;
    max-width: 980px;
    font-size: clamp(36px, 4vw, 58px);
    line-height: 0.98;
    color: #fffaf1;
    letter-spacing: -0.055em;
}

.hero p {
    position: relative;
    max-width: 820px;
    margin: 0;
    color: #d8d0bf;
    font-size: 18px;
    line-height: 1.5;
}

.creator-row {
    position: relative;
    margin-top: 22px;
    color: #fffaf1;
    font-size: 14px;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.powered-row {
    position: relative;
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 22px;
    color: #7fd28a;
    font-size: 13px;
    font-weight: 900;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.sb-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 999px;
    color: #171914;
    background: #fffaf1;
    font-weight: 900;
    text-transform: none;
    letter-spacing: -0.02em;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}

.sb-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: linear-gradient(135deg, #2f8f3a, #7fd28a);
    display: inline-block;
}

.card {
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(191, 232, 197, 0.35);
    border-radius: 28px;
    padding: 26px 28px;
    background:
      linear-gradient(180deg, rgba(255,250,241,0.98) 0%, rgba(246,241,232,0.97) 100%);
    box-shadow: 0 18px 44px rgba(0, 0, 0, 0.18);
    min-height: 190px;
}

.card::before {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    border-radius: 50%;
    background: rgba(127, 210, 138, 0.14);
    right: -80px;
    top: -90px;
}

.card h2 {
    position: relative;
    margin: 0 0 10px 0;
    font-size: 30px;
    color: #171914;
    letter-spacing: -0.04em;
}

.card p {
    position: relative;
    color: #4f5749;
    font-size: 16px;
    line-height: 1.5;
}

.video-placeholder {
    position: relative;
    height: 310px;
    border-radius: 22px;
    border: 2px dashed rgba(127, 210, 138, 0.78);
    background:
      radial-gradient(circle at center, rgba(255,250,241,0.55), transparent 24%),
      repeating-linear-gradient(90deg, rgba(47,143,58,0.45) 0 56px, rgba(23,105,43,0.45) 56px 112px),
      #1f7a34;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    color: #fffaf1;
    margin-top: 18px;
    box-shadow: inset 0 0 0 1px rgba(255,250,241,0.12);
}

.video-placeholder::before {
    content: "";
    position: absolute;
    inset: 22px;
    border: 2px solid rgba(255,250,241,0.28);
    border-radius: 14px;
}

.video-placeholder::after {
    content: "";
    position: absolute;
    width: 90px;
    height: 90px;
    border: 2px solid rgba(255,250,241,0.30);
    border-radius: 50%;
}

.video-placeholder .play {
    width: 66px;
    height: 66px;
    border-radius: 999px;
    background: #2f8f3a;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 28px;
    margin-bottom: 14px;
    z-index: 1;
    box-shadow: 0 12px 28px rgba(0, 0, 0, 0.28);
}

.video-placeholder strong,
.video-placeholder span {
    z-index: 1;
    text-shadow: 0 1px 12px rgba(0,0,0,0.35);
}

.audio-note {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: #dff6e3;
    color: #17692b;
    font-weight: 900;
    font-size: 13px;
    margin-bottom: 14px;
}

.viewer-title {
    margin-top: 6px;
    font-size: 30px;
    font-weight: 900;
    color: #fffaf1;
    letter-spacing: -0.035em;
}

[data-testid="stCaptionContainer"] {
    color: #d8d0bf;
}

.stAudio audio {
    width: 100%;
    border-radius: 999px;
}

div[data-testid="stVideo"] {
    overflow: hidden;
    border-radius: 24px;
    border: 1px solid rgba(191, 232, 197, 0.35);
    box-shadow: 0 18px 44px rgba(0, 0, 0, 0.22);
}

div[data-testid="stVideo"] video {
    max-height: 520px;
    object-fit: contain;
    background: #0d0f0c;
}

code {
    color: #2f8f3a !important;
    background: rgba(47, 143, 58, 0.12) !important;
}
</style>
"""


st.markdown(CSS, unsafe_allow_html=True)


def resolve_manifest_path() -> Path | None:
    for p in MANIFEST_CANDIDATES:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_manifest_csv(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def build_match_labels(idx_df: pd.DataFrame) -> dict[str, str]:
    labels = {str(int(mid)): str(int(mid)) for mid in idx_df["match_id"].tolist()}
    mp = resolve_manifest_path()
    if mp is None:
        return labels
    mdf = load_manifest_csv(str(mp))
    if mdf.empty or "match_id" not in mdf.columns:
        return labels
    if "to_speak" in mdf.columns:
        mdf = mdf[mdf["to_speak"] == 1]
    for _, r in mdf.iterrows():
        try:
            mid = str(int(r.get("match_id")))
        except Exception:
            continue
        home = str(r.get("home_team", "")).strip()
        away = str(r.get("away_team", "")).strip()
        stage = str(r.get("stage", "")).strip()
        if home and away and stage:
            labels[mid] = f"{home} — {away} | {stage}"
        elif home and away:
            labels[mid] = f"{home} — {away}"
    return labels


def _json_load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_index_path() -> Path | None:
    for p in INDEX_CANDIDATES:
        if p.exists():
            return p
    return None


def resolve_video_path() -> Path | None:
    for p in VIDEO_TRACK_CANDIDATES:
        if p.exists():
            return p
    for ext in ("*.mp4", "*.mov", "*.m4v", "*.webm"):
        found = sorted(APP_DIR.glob(ext))
        if found:
            return found[0]
    return None


def resolve_audio_tracks() -> dict[str, Path]:
    tracks = {}
    used = set()

    for label, candidates in AUDIO_TRACK_CANDIDATES.items():
        for p in candidates:
            if p.exists():
                tracks[label] = p
                used.add(p.resolve())
                break

    # Fallback: show any extra audio files if user adds more variants later.
    extra_files = []
    for ext in ("*.wav", "*.mp3", "*.m4a", "*.ogg", "*.flac"):
        extra_files.extend(sorted(APP_DIR.glob(ext)))

    extra_n = 1
    for p in extra_files:
        if p.resolve() in used:
            continue
        label = f"Аудиодорожка {extra_n}"
        tracks[label] = p
        used.add(p.resolve())
        extra_n += 1

    return tracks


def resolve_logo_path() -> Path | None:
    for p in LOGO_CANDIDATES:
        if p.exists():
            return p
    return None


@st.cache_data(show_spinner=False)
def load_index_csv(path_str: str) -> pd.DataFrame:
    p = Path(path_str)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_match_artifacts(row_dict: dict):
    events_path = Path(row_dict["events_std"])
    sb360_path = Path(row_dict["sb360_std"])
    meta_path = Path(row_dict["meta"])
    bad_ids_path = (
        Path(row_dict.get("bad_ids", "")) if row_dict.get("bad_ids") else None
    )

    events_std = _json_load(events_path)
    sb360_std = _json_load(sb360_path)
    meta = _json_load(meta_path) if meta_path.exists() else {}
    bad_ids = (
        set(_json_load(bad_ids_path))
        if bad_ids_path and bad_ids_path.exists()
        else set()
    )

    jersey_by_id, jersey_by_name = viz.build_jersey_maps(events_std)
    df_events_viz = viz.make_viz_events_df(events_std, jersey_by_id, jersey_by_name)
    frames_by_id, visible_by_id = viz.make_360_indexes(sb360_std)

    team_colors = dict(meta.get("team_colors", {}) or {})
    teams = set(meta.get("teams", []) or [])
    if {"Spain", "Georgia"}.issubset(teams):
        team_colors["Spain"] = "#c62828"
        team_colors["Georgia"] = "#2563eb"

    return {
        "meta": meta,
        "bad_ids": bad_ids,
        "df_events_viz": df_events_viz,
        "frames_by_id": frames_by_id,
        "visible_by_id": visible_by_id,
        "team_colors": team_colors,
    }


def render_event(ctx: dict, event_id: str, disable_360: bool):
    df_events_viz = ctx["df_events_viz"]
    frames_by_id = {} if disable_360 else ctx["frames_by_id"]
    visible_by_id = {} if disable_360 else ctx["visible_by_id"]

    meta = ctx["meta"]
    ref_map = meta.get("ref_by_period", {}) or {}
    ref_map = (
        {int(k): v for k, v in ref_map.items()} if isinstance(ref_map, dict) else {}
    )
    ref_team = ref_map.get(1)

    fig, _ = viz.draw_event_keep_style_half_switch_full(
        df_events_viz,
        event_id,
        frames_by_id,
        visible_by_id,
        reference_team=ref_team,
        ref_by_period=ref_map,
        team_colors=ctx["team_colors"],
        show_badge=True,
        show_header=True,
        show_ref=False,
        show_actor_source=False,
    )

    fig.set_size_inches(10.8, 6.0, forward=True)
    return fig


def render_home() -> None:
    logo_path = resolve_logo_path()
    if logo_path is not None:
        suffix = logo_path.suffix.lower()
        mime = "image/svg+xml" if suffix == ".svg" else "image/png"
        encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        logo_html = f'<img src="data:{mime};base64,{encoded}" alt="StatsBomb Data" style="height:42px; width:auto; display:inline-block; vertical-align:middle;" />'
    else:
        logo_html = (
            '<span class="sb-badge"><span class="sb-dot"></span>StatsBomb Data</span>'
        )

    st.markdown(
        f"""
        <section class="hero">
          <h1>Тифлокомментарии и визуализации матчей Euro-2024</h1>
          <p>Демо-сервис для просмотра футбольных эпизодов, визуализаций StatsBomb 360 и готовой аудиодорожки тифлокомментария.</p>
          <div class="creator-row">created by Angelina Myasnikova</div>
          <div class="powered-row">inspired by {logo_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        video_path = resolve_video_path()
        if video_path is not None:
            st.markdown(
                """
                <div class="card">
                  <h2>Видео матча</h2>
                  <p>Первые 7,5 минут матча Шотландия — Германия. Видео сделано с готовой аудиодорожкой тифлокомментария.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.video(str(video_path))
        else:
            st.markdown(
                """
                <div class="card">
                  <h2>Видео матча</h2>
                  <p>Здесь будет видео матча. Позже можно будет синхронизировать его с аудиодорожкой и временными метками комментариев.</p>
                  <div class="video-placeholder">
                    <div class="play">▶</div>
                    <strong>Плашка под видео</strong>
                    <span>файл будет добавлен позже</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown(
            """
            <div class="card">
              <span class="audio-note">полностью озвученный матч</span>
              <h2>Шотландия — Германия</h2>
              <p>Тифлокомментарии сгенерированы по цепочкам событий StatsBomb и собраны в аудиодорожку.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        audio_tracks = resolve_audio_tracks()
        if audio_tracks:
            selected_audio_label = st.radio(
                "Выбор аудиодорожки",
                list(audio_tracks.keys()),
                horizontal=True,
                label_visibility="collapsed",
            )
            st.audio(str(audio_tracks[selected_audio_label]))
        else:
            st.info(
                "Положи аудиофайл в `streamlit_match_viewer`, и он появится здесь автоматически."
            )


def render_visual_viewer(idx_df: pd.DataFrame, match_labels: dict[str, str]) -> None:
    st.markdown(
        '<div class="viewer-title">Просмотр визуализаций матча</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.title("Управление")
    match_options = idx_df["match_id"].astype(int).astype(str).tolist()
    selected_match = st.sidebar.selectbox(
        "Матч",
        match_options,
        index=0,
        format_func=lambda x: match_labels.get(str(x), str(x)),
    )
    period_filter = st.sidebar.selectbox("Тайм", ["all", 1, 2], index=0)
    fps = st.sidebar.slider("FPS", min_value=1, max_value=12, value=4)
    st.sidebar.caption("FPS — скорость автопроигрывания: сколько событий показывается за секунду.")

    row = (
        idx_df[idx_df["match_id"].astype(int).astype(str) == selected_match]
        .iloc[0]
        .to_dict()
    )
    ctx = load_match_artifacts(row)

    edf = ctx["df_events_viz"].copy()
    if period_filter != "all":
        edf = edf[edf["period"] == int(period_filter)]

    edf = edf.sort_values(["index", "timestamp"]) if "index" in edf.columns else edf
    event_ids = [x for x in edf["id"].tolist() if isinstance(x, str)]
    if not event_ids:
        st.warning("Нет событий для текущего фильтра")
        st.stop()

    if "last_match" not in st.session_state:
        st.session_state.last_match = None
    if "frame_idx" not in st.session_state:
        st.session_state.frame_idx = 0
    if "playing" not in st.session_state:
        st.session_state.playing = True

    if st.session_state.last_match != selected_match:
        st.session_state.last_match = selected_match
        st.session_state.frame_idx = 0
        st.session_state.playing = True

    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    with c1:
        if st.button("◀ Prev", use_container_width=True):
            st.session_state.frame_idx = (st.session_state.frame_idx - 1) % len(
                event_ids
            )
            st.session_state.playing = False
    with c2:
        if st.button("Next ▶", use_container_width=True):
            st.session_state.frame_idx = (st.session_state.frame_idx + 1) % len(
                event_ids
            )
            st.session_state.playing = False
    with c3:
        if st.button("Play/Pause", use_container_width=True):
            st.session_state.playing = not st.session_state.playing
    with c4:
        st.write(f"Кадр: **{st.session_state.frame_idx + 1} / {len(event_ids)}**")

    cur_id = event_ids[st.session_state.frame_idx]
    disable_360 = cur_id in ctx["bad_ids"]
    fig = render_event(ctx, cur_id, disable_360=disable_360)
    st.pyplot(fig)

    if st.session_state.playing:
        time.sleep(1.0 / float(fps))
        st.session_state.frame_idx = (st.session_state.frame_idx + 1) % len(event_ids)
        st.rerun()


index_path = resolve_index_path()
if index_path is None:
    st.error("Не найден index CSV")
    st.stop()

idx_df = load_index_csv(str(index_path))
if idx_df.empty:
    st.error(f"Индекс пустой: {index_path}")
    st.stop()

if "to_speak" in idx_df.columns:
    idx_df = idx_df[idx_df["to_speak"] == 1].copy()

idx_df = idx_df.sort_values("match_id").reset_index(drop=True)
if idx_df.empty:
    st.warning("TOP-15 матчи не найдены в индексе")
    st.stop()

match_labels = build_match_labels(idx_df)

home_tab, viz_tab = st.tabs(["Главная", "Визуализации матча"])
with home_tab:
    render_home()
with viz_tab:
    render_visual_viewer(idx_df, match_labels)
