from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.patches import Rectangle
from mplsoccer import Pitch

PITCH_L = 120.0
PITCH_W = 80.0


def _marker_box(marker: str, facecolor: str, edgecolor: str = "black", size: int = 10) -> DrawingArea:
    da = DrawingArea(18, 14, 0, 0)
    m = Line2D(
        [9],
        [7],
        marker=marker,
        linestyle="",
        markerfacecolor=facecolor,
        markeredgecolor=edgecolor,
        markersize=size,
    )
    da.add_artist(m)
    return da


def _visible_area_box(fc: str = "gray", ec: str = "white", lw: float = 2.0, ls: str = "-.", alpha: float = 0.2) -> DrawingArea:
    da = DrawingArea(26, 14, 0, 0)
    rect = Rectangle((2, 3), 22, 8, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, alpha=alpha)
    da.add_artist(rect)
    return da


def add_badge(
    ax: Any,
    ts: Any,
    event_id: str,
    team: str,
    player: str,
    etype: str,
    team_label: str = "teammates",
    opponent_label: str = "rivals",
    team_color: str = "tab:blue",
    opponent_color: str = "tab:red",
    actor_color: str = "gold",
) -> None:
    line3 = TextArea(
        f"current_action: {team} | {player} | {etype}",
        textprops=dict(color="white", fontsize=10, family="monospace"),
    )
    actor = _marker_box("o", facecolor=actor_color, size=12)
    actor_txt = TextArea("actor", textprops=dict(color="white", fontsize=10, family="monospace"))
    tm_dot = _marker_box("o", facecolor=team_color, size=9)
    tm_txt = TextArea(team_label, textprops=dict(color="white", fontsize=10, family="monospace"))
    rv_dot = _marker_box("o", facecolor=opponent_color, size=9)
    rv_txt = TextArea(opponent_label, textprops=dict(color="white", fontsize=10, family="monospace"))
    va_box = _visible_area_box()
    va_txt = TextArea("visible area", textprops=dict(color="white", fontsize=10, family="monospace"))

    legend_row = HPacker(
        children=[actor, actor_txt, TextArea("  "), tm_dot, tm_txt, TextArea("  "), rv_dot, rv_txt, TextArea("  "), va_box, va_txt],
        align="center",
        pad=0,
        sep=3,
    )
    box = VPacker(children=[line3, legend_row], align="left", pad=0, sep=4)
    anchored = AnchoredOffsetbox(
        loc="upper left",
        child=box,
        pad=0.3,
        borderpad=0.0,
        frameon=True,
        bbox_to_anchor=(0.0, 1.0),
        bbox_transform=ax.transAxes,
    )
    anchored.patch.set_facecolor("#111827")
    anchored.patch.set_edgecolor("white")
    anchored.patch.set_alpha(0.9)
    ax.add_artist(anchored)


def _fmt_jersey(x: Any) -> str | None:
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    try:
        xf = float(x)
        if xf.is_integer():
            return str(int(xf))
        return str(x)
    except Exception:
        return str(x)


def _num_fontsize(num: str, base: int = 12) -> int:
    if len(num) <= 1:
        return base
    if len(num) == 2:
        return base - 1
    return base - 2


def _rot180_xy(x: Any, y: Any) -> tuple[Any, Any]:
    if pd.isna(x) or pd.isna(y):
        return x, y
    return (PITCH_L - float(x)), (PITCH_W - float(y))


def _rot180_visible_area(va: Any) -> Any:
    if va is None:
        return None
    if isinstance(va, float) and np.isnan(va):
        return None
    if len(va) == 0:
        return va
    pts = np.array(va, dtype=float).reshape(-1, 2)
    pts[:, 0] = PITCH_L - pts[:, 0]
    pts[:, 1] = PITCH_W - pts[:, 1]
    return pts.reshape(-1).tolist()


def _get_frame_xy_cols(fr: pd.DataFrame) -> tuple[str | None, str | None, pd.DataFrame]:
    if fr is None or len(fr) == 0:
        return None, None, fr
    if ("x" in fr.columns) and ("y" in fr.columns):
        return "x", "y", fr
    if "location" in fr.columns:
        fr2 = fr.copy()
        fr2["x"] = fr2["location"].apply(lambda t: t[0] if isinstance(t, (list, tuple)) and len(t) >= 2 else np.nan)
        fr2["y"] = fr2["location"].apply(lambda t: t[1] if isinstance(t, (list, tuple)) and len(t) >= 2 else np.nan)
        return "x", "y", fr2
    raise KeyError("frames_by_id[event_id] must include x/y or location=[x,y].")


def draw_event_keep_style(
    df_events: pd.DataFrame,
    event_id: str,
    frames_by_id: dict[str, pd.DataFrame] | None = None,
    visible_by_id: dict[str, Any] | None = None,
    facecolor: str = "#0e1117",
    linecolor: str = "#c7d5cc",
    actor_color: str = "gold",
    reference_team: str = "Scotland",
    team_colors: dict[str, str] | None = None,
    show_badge: bool = True,
) -> tuple[Any, Any]:
    """Draw one event with optional 360 layer and action trajectory."""
    if team_colors is None:
        team_colors = {"Scotland": "tab:red", "Germany": "tab:blue"}

    ev_df = df_events.loc[df_events["id"] == event_id]
    if ev_df.empty:
        raise ValueError(f"event_id={event_id} not found in df_events")
    ev = ev_df.iloc[0]

    ts = ev.get("timestamp", "?")
    team = ev.get("team_name", "?")
    pname = ev.get("event_player_name", ev.get("player_name", "?"))
    etype = ev.get("type_name", "?")
    sx = ev.get("event_x", np.nan)
    sy = ev.get("event_y", np.nan)
    ex = ev.get("event_end_x", np.nan)
    ey = ev.get("event_end_y", np.nan)
    event_j = _fmt_jersey(ev.get("event_jersey_number", None))

    teams_in_match = df_events.get("team_name", pd.Series(dtype=object)).dropna().unique().tolist()
    opponent_team = next((t for t in teams_in_match if t != team), None)
    team_color = team_colors.get(str(team), "white")
    opp_color = team_colors.get(str(opponent_team), "white")

    # half-switch: in 2nd half, swap reference_team to the opponent
    ref_team = reference_team
    period = ev.get("period")
    if period == 2:
        ref_team = next((t for t in teams_in_match if str(t) != str(reference_team)), reference_team)

    flip = str(team) != str(ref_team)
    if flip:
        sx, sy = _rot180_xy(sx, sy)
        ex, ey = _rot180_xy(ex, ey)

    fig, ax = plt.subplots(figsize=(13, 8), tight_layout=True)
    fig.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    pitch = Pitch(pitch_type="statsbomb", pitch_color=facecolor, line_color=linecolor)
    pitch.draw(ax=ax)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    pad_x, pad_y = 0, 4
    ax.set_xlim(min(x0, x1) - pad_x, max(x0, x1) + pad_x)
    if ax.yaxis_inverted():
        ax.set_ylim(max(y0, y1) + pad_y, min(y0, y1) - pad_y)
    else:
        ax.set_ylim(min(y0, y1) - pad_y, max(y0, y1) + pad_y)

    fr = frames_by_id.get(event_id) if frames_by_id is not None else None
    has_360 = fr is not None and len(fr) > 0
    if has_360:
        va = visible_by_id.get(event_id) if visible_by_id is not None else None
        if flip:
            va = _rot180_visible_area(va)
        if va:
            poly = np.array(va).reshape(-1, 2)
            pitch.polygon([poly], ax=ax, color="gray", ec="white", lw=3, linestyle="-.", alpha=0.2, zorder=1)

        fx, fy, fr_local = _get_frame_xy_cols(fr)
        if flip:
            fr_local = fr_local.copy()
            fr_local[fx] = PITCH_L - fr_local[fx].astype(float)
            fr_local[fy] = PITCH_W - fr_local[fy].astype(float)

        for _, r in fr_local.iterrows():
            if bool(r.get("actor", False)):
                continue
            teammate = bool(r.get("teammate", False))
            keeper = bool(r.get("keeper", False))
            player_team = team if teammate else opponent_team
            color = team_colors.get(str(player_team).strip(), "white")
            marker = "D" if keeper else "o"
            size = 90 if keeper else 70
            ax.scatter(r[fx], r[fy], c=color, s=size, marker=marker, edgecolors="black", linewidths=0.8, zorder=3)

    if pd.notna(sx) and pd.notna(sy):
        ax.scatter(sx, sy, c=actor_color, s=320, marker="o", edgecolors="black", linewidths=1.2, zorder=6)
        if event_j is not None:
            ax.text(
                sx,
                sy,
                event_j,
                ha="center",
                va="center",
                color="black",
                fontsize=_num_fontsize(event_j, base=12),
                fontweight="bold",
                zorder=7,
                path_effects=[pe.withStroke(linewidth=2.0, foreground="white")],
            )

    has_end = pd.notna(sx) and pd.notna(sy) and pd.notna(ex) and pd.notna(ey)
    if etype in ("Pass", "Shot") and has_end:
        pitch.lines(xstart=sx, ystart=sy, xend=ex, yend=ey, ax=ax, comet=True, color="white", zorder=5)
    elif etype == "Carry" and has_end:
        pitch.arrows(sx, sy, ex, ey, ax=ax, color="white", lw=2.5, linestyle="--", zorder=5)

    if show_badge:
        add_badge(
            ax,
            ts,
            event_id,
            team,
            pname,
            etype,
            team_label=str(team),
            opponent_label=str(opponent_team) if opponent_team else "opponent",
            team_color=team_color,
            opponent_color=opp_color,
            actor_color=actor_color,
        )

    fig.subplots_adjust(top=0.93)
    fig.text(0.48, 0.95, f"time: {ts}", ha="left", va="top", color="white", fontsize=12, family="monospace")

    if not has_360:
        ax.text(
            0.02,
            0.86,
            "360: NOT AVAILABLE",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="white",
            fontsize=10,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#111827", ec="white", alpha=0.9),
        )

    return fig, ax


def save_event_figure(fig: Any, event_id: str, out_dir: str = "event_images", dpi: int = 200) -> str:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    path = out_path / f"{event_id}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    return str(path)
