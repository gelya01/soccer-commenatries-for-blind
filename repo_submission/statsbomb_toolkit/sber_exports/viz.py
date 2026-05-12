"""
Visualization utilities for payload events and pass-direction diagnostics.
Auto-exported from sber_chain_eval.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import pandas as pd
import unicodedata
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker
from matplotlib.patches import Rectangle
from mplsoccer import Pitch

HAS_PITCH = True
PITCH_L = 120.0
PITCH_W = 80.0

DEFAULT_TEAM_COLORS = {
    "Scotland": "#1f77b4",
    "Germany": "#111111",
    "Spain": "#c62828",
    "England": "#f2f2f2",
    "France": "#1e3a8a",
    "Italy": "#1d4ed8",
    "Portugal": "#b91c1c",
    "Netherlands": "#ea580c",
    "Belgium": "#7f1d1d",
    "Croatia": "#dc2626",
    "Switzerland": "#ef4444",
    "Denmark": "#b91c1c",
    "Austria": "#dc2626",
    "Turkey": "#b91c1c",
    "Georgia": "#2563eb",
    "Romania": "#facc15",
    "Slovenia": "#16a34a",
    "Slovakia": "#2563eb",
    "Serbia": "#b91c1c",
    "Ukraine": "#2563eb",
    "Poland": "#ef4444",
    "Albania": "#b91c1c",
    "Hungary": "#b91c1c",
    "Czechia": "#dc2626",
}


def resolve_team_colors(teams: list[str], custom_colors: dict[str, str] | None = None):
    custom_colors = custom_colors or {}
    out = {}
    for i, t in enumerate(teams):
        out[t] = custom_colors.get(t) or DEFAULT_TEAM_COLORS.get(t) or (["#ef4444", "#3b82f6", "#22c55e", "#f59e0b"][i % 4])
    return out


def _norm_name(name):
    if not isinstance(name, str):
        return None
    s = name.strip()
    if not s:
        return None
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.casefold()


def _surname(name):
    if not isinstance(name, str):
        return None
    toks = name.strip().split()
    return toks[-1] if toks else None


def _safe_int(x):
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None

def _marker_box(marker: str, facecolor: str, edgecolor: str = 'black', size: int = 10) -> DrawingArea:
    da = DrawingArea(18, 14, 0, 0)
    m = Line2D([9], [7], marker=marker, linestyle='', markerfacecolor=facecolor, markeredgecolor=edgecolor, markersize=size)
    da.add_artist(m)
    return da


def _visible_area_box(fc: str = 'gray', ec: str = 'white', lw: float = 2.0, ls: str = '-.', alpha: float = 0.2) -> DrawingArea:
    da = DrawingArea(26, 14, 0, 0)
    rect = Rectangle((2, 3), 22, 8, facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, alpha=alpha)
    da.add_artist(rect)
    return da


def _fmt_jersey(x):
    if x is None:
        return None
    try:
        xf = float(x)
        if xf != xf:
            return None
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


def _event_end_xy_from_payload_event(ej: dict):
    et = ((ej.get('type') or {}).get('name'))
    if et == 'Pass':
        end = ((ej.get('pass') or {}).get('end_location'))
        if isinstance(end, (list, tuple)) and len(end) >= 2:
            return float(end[0]), float(end[1])
    if et == 'Carry':
        end = ((ej.get('carry') or {}).get('end_location'))
        if isinstance(end, (list, tuple)) and len(end) >= 2:
            return float(end[0]), float(end[1])
    if et == 'Shot':
        end = ((ej.get('shot') or {}).get('end_location'))
        if isinstance(end, (list, tuple)) and len(end) >= 2:
            return float(end[0]), float(end[1])
    return None, None


def _actor_loc_from_sb360(sb):
    ff = (sb or {}).get('freeze_frame') or []
    for p in ff:
        if isinstance(p, dict) and p.get('actor') is True:
            loc = p.get('location')
            if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                return float(loc[0]), float(loc[1])
    return None


def _is_finite_xy(x, y):
    try:
        return x is not None and y is not None and np.isfinite(float(x)) and np.isfinite(float(y))
    except Exception:
        return False


def _add_badge(ax, team, player, etype, ts, event_id):
    line = TextArea(f'current_action: {team} | {player} | {etype}', textprops=dict(color='white', fontsize=10, family='monospace'))
    actor = _marker_box('o', facecolor='gold', size=12)
    actor_txt = TextArea('actor', textprops=dict(color='white', fontsize=10, family='monospace'))
    tm_dot = _marker_box('o', facecolor='tab:red', size=9)
    tm_txt = TextArea('teammates', textprops=dict(color='white', fontsize=10, family='monospace'))
    rv_dot = _marker_box('o', facecolor='tab:blue', size=9)
    rv_txt = TextArea('opponents', textprops=dict(color='white', fontsize=10, family='monospace'))
    va_box = _visible_area_box()
    va_txt = TextArea('visible area', textprops=dict(color='white', fontsize=10, family='monospace'))

    row = HPacker(children=[actor, actor_txt, TextArea('  '), tm_dot, tm_txt, TextArea('  '), rv_dot, rv_txt, TextArea('  '), va_box, va_txt], align='center', pad=0, sep=3)
    box = VPacker(children=[line, row], align='left', pad=0, sep=4)

    anchored = AnchoredOffsetbox(loc='upper left', child=box, pad=0.3, borderpad=0.0, frameon=True, bbox_to_anchor=(0.0, 1.0), bbox_transform=ax.transAxes)
    anchored.patch.set_facecolor('#111827')
    anchored.patch.set_edgecolor('white')
    anchored.patch.set_alpha(0.9)
    ax.add_artist(anchored)

    ax.figure.text(0.76, 0.965, f'time: {ts}', ha='left', va='top', color='white', fontsize=11, family='monospace')
    ax.figure.text(0.76, 0.935, f'id: {event_id}', ha='left', va='top', color='white', fontsize=9, family='monospace')


def add_badge(
    ax,
    ts,
    event_id,
    team,
    player,
    etype,
    team_label="teammates",
    opponent_label="rivals",
    team_color="tab:blue",
    opponent_color="tab:red",
    actor_color="gold",
):
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
        children=[
            actor,
            actor_txt,
            TextArea("  "),
            tm_dot,
            tm_txt,
            TextArea("  "),
            rv_dot,
            rv_txt,
            TextArea("  "),
            va_box,
            va_txt,
        ],
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


def draw_payload_event_fullstyle(
    event_item,
    facecolor="#0e1117",
    linecolor="#c7d5cc",
    team_colors: dict[str, str] | None = None,
    show_badge: bool = True,
):
    ej = event_item.get('event_json') or {}
    sb = event_item.get('sb360_json')

    ts = ej.get('timestamp', '?')
    eid = event_item.get('event_id', '?')
    team = ((ej.get('team') or {}).get('name', '?'))
    player = ((ej.get('player') or {}).get('name', '?'))
    etype = ((ej.get('type') or {}).get('name', '?'))

    loc = ej.get('location') or [None, None]
    sx = float(loc[0]) if isinstance(loc, (list, tuple)) and len(loc) >= 2 else None
    sy = float(loc[1]) if isinstance(loc, (list, tuple)) and len(loc) >= 2 else None
    ex, ey = _event_end_xy_from_payload_event(ej)

    # actor берем из 360, если есть; иначе из event.location
    actor_xy = _actor_loc_from_sb360(sb)
    if actor_xy is not None:
        sx, sy = actor_xy

    fig, ax = plt.subplots(figsize=(13, 8), tight_layout=True)
    fig.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    pitch = Pitch(pitch_type='statsbomb', pitch_color=facecolor, line_color=linecolor)
    pitch.draw(ax=ax)

    # palette
    team_colors = team_colors or resolve_team_colors([team] if team else [])
    team_color = team_colors.get(team, "#ef4444")
    opp_candidates = [v for k, v in team_colors.items() if k != team]
    opp_color = opp_candidates[0] if opp_candidates else "#3b82f6"

    # 360 overlay
    if sb is not None:
        va = sb.get('visible_area')
        if isinstance(va, list) and len(va) >= 6:
            poly = np.array(va, dtype=float).reshape(-1, 2)
            pitch.polygon([poly], ax=ax, color='gray', ec='white', lw=2.5, linestyle='-.', alpha=0.22, zorder=1)

        ff = sb.get('freeze_frame') or []
        for p in ff:
            if not isinstance(p, dict):
                continue
            if bool(p.get('actor', False)):
                continue
            loc2 = p.get('location')
            if not (isinstance(loc2, (list, tuple)) and len(loc2) >= 2):
                continue
            x, y = float(loc2[0]), float(loc2[1])
            teammate = bool(p.get('teammate', False))
            keeper = bool(p.get('keeper', False))
            color = team_color if teammate else opp_color
            marker = 'o'
            size = 92 if keeper else 70
            ax.scatter(x, y, c=color, s=size, marker=marker, edgecolors='black', linewidths=0.8, zorder=3)

    # actor point
    if _is_finite_xy(sx, sy):
        ax.scatter(sx, sy, c='gold', s=320, marker='o', edgecolors='black', linewidths=1.2, zorder=6)
        jersey = _fmt_jersey(((ej.get('player') or {}).get('jersey_number')))
        if jersey:
            ax.text(sx, sy, jersey, ha='center', va='center', color='black', fontsize=_num_fontsize(jersey, base=12), fontweight='bold', zorder=7, path_effects=[pe.withStroke(linewidth=2.0, foreground='white')])

    # action trajectory
    if _is_finite_xy(sx, sy) and _is_finite_xy(ex, ey):
        if etype in ('Pass', 'Shot'):
            pitch.lines(xstart=sx, ystart=sy, xend=ex, yend=ey, ax=ax, comet=True, color='white', zorder=5)
        elif etype == 'Carry':
            pitch.arrows(sx, sy, ex, ey, ax=ax, color='white', lw=2.5, linestyle='--', zorder=5)

    if show_badge:
        _add_badge(ax, team, player, etype, ts, eid)
    return fig, ax


def collect_payload_events(prepared, period, limit=4, require_360=None):
    out = []
    for ch in prepared.get("payloads", []):
        for e in ch.get("events", []):
            ej = e.get("event_json") or {}
            if ej.get("period") != period:
                continue
            has360 = e.get("sb360_json") is not None
            if require_360 is True and not has360:
                continue
            if require_360 is False and has360:
                continue
            out.append(e)
            if len(out) >= limit:
                return out
    return out


def show_match_payload_visuals(prepared, cfg, n_with360=3, n_without360=3):
    team_colors = (cfg or {}).get("team_colors", {})

    def _draw_many(events, title):
        print(title, "| n=", len(events))
        if not events:
            print("  (ничего не найдено для этого среза)")
            return
        for e in events:
            fig, ax = draw_payload_event_fullstyle(e, team_colors=team_colors, show_badge=True)
            try:
                from IPython.display import display

                display(fig)
            except Exception:
                pass
            plt.show()
            plt.close(fig)

    _draw_many(collect_payload_events(prepared, 1, n_with360, True), "=== PERIOD 1 | with 360 ===")
    _draw_many(collect_payload_events(prepared, 2, n_with360, True), "=== PERIOD 2 | with 360 ===")
    _draw_many(collect_payload_events(prepared, 1, n_without360, False), "=== PERIOD 1 | no 360 ===")
    _draw_many(collect_payload_events(prepared, 2, n_without360, False), "=== PERIOD 2 | no 360 ===")


def _iter_pass_events(chains):
    rows = []
    for ci, ch in enumerate(chains, start=1):
        for ei, item in enumerate(ch.get('events', []), start=1):
            ej = item.get('event_json') or {}
            et = ((ej.get('type') or {}).get('name'))
            if et != 'Pass':
                continue

            loc = ej.get('location') or [None, None]
            end = ((ej.get('pass') or {}).get('end_location')) or [None, None]
            sx, sy = (loc + [None, None])[:2]
            ex, ey = (end + [None, None])[:2]
            if sx is None or sy is None or ex is None or ey is None:
                continue

            d = item.get('derived') or {}
            sig = d.get('episode_signals') or {}
            mv = d.get('movement') or {}
            ori = d.get('orientation') or {}
            rows.append({
                'chain_idx': ci,
                'event_pos': ei,
                'event_id': item.get('event_id'),
                'timestamp': ej.get('timestamp'),
                'period': ej.get('period'),
                'team': ((ej.get('team') or {}).get('name')),
                'player': ((ej.get('player') or {}).get('name')),
                'recipient': (((ej.get('pass') or {}).get('recipient') or {}).get('name')),
                'sx': float(sx), 'sy': float(sy), 'ex': float(ex), 'ey': float(ey),
                'compass': sig.get('pass_direction_compass') or mv.get('compass') or 'unknown',
                'target_rel': sig.get('pass_target_rel'),
                'target_abs': sig.get('pass_target_abs'),
                'pass_style_ru': sig.get('pass_style_ru'),
                'pass_length': sig.get('pass_length'),
                'forward_delta': mv.get('forward_delta'),
                'lateral_delta': mv.get('lateral_delta'),
                'own_goal_x': ori.get('own_goal_x'),
                'opp_goal_x': ori.get('opp_goal_x'),
                'sb360_is_null': item.get('sb360_json') is None,
                'sb360_json': item.get('sb360_json'),
            })
    return pd.DataFrame(rows)


def _arrow_color(compass):
    cmap = {
        'forward_left': '#22c55e',
        'forward_right': '#16a34a',
        'forward': '#65a30d',
        'backward_left': '#ef4444',
        'backward_right': '#dc2626',
        'backward': '#b91c1c',
        'left': '#0ea5e9',
        'right': '#0284c7',
        'short_or_static': '#a1a1aa',
        'unknown': '#a3a3a3',
    }
    return cmap.get(compass, '#a3a3a3')


def plot_pass_row(r, title_prefix=''):
    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)

    if HAS_PITCH:
        pitch = Pitch(pitch_type='statsbomb', pitch_color='#0b1220', line_color='#d1d5db')
        pitch.draw(ax=ax)
        pitch.arrows(r.sx, r.sy, r.ex, r.ey, ax=ax, color=_arrow_color(r.compass), width=2, headwidth=4, headlength=5, zorder=4)
        ax.scatter([r.sx], [r.sy], c='gold', s=180, edgecolors='black', linewidths=0.8, zorder=5)
        ax.scatter([r.ex], [r.ey], c='#93c5fd', s=130, edgecolors='black', linewidths=0.8, zorder=5)
    else:
        ax.set_facecolor('#0b1220')
        ax.set_xlim(0, 120)
        ax.set_ylim(80, 0)
        ax.grid(alpha=0.2)
        ax.arrow(r.sx, r.sy, r.ex-r.sx, r.ey-r.sy, color=_arrow_color(r.compass), width=0.3, length_includes_head=True)
        ax.scatter([r.sx], [r.sy], c='gold', s=180)
        ax.scatter([r.ex], [r.ey], c='#93c5fd', s=130)

    head = f"{title_prefix} chain={int(r.chain_idx)} | {r.timestamp} | {r.player} → {r.recipient if pd.notna(r.recipient) else '—'}"
    sub = f"compass={r.compass} | style={r.pass_style_ru} | target_rel={r.target_rel} | target_abs={r.target_abs} | fwd={r.forward_delta:.2f} | lat={r.lateral_delta:.2f}"
    ax.set_title(head + "\n" + sub, fontsize=11, color='white' if HAS_PITCH else 'black')
    plt.show()


# ---------------------------------------------------------------------------
# Half-switch style visualization (adapted from half_switch_test flow)
# ---------------------------------------------------------------------------
def _hs_rot180_xy(x, y):
    if x is None or y is None:
        return x, y
    return PITCH_L - float(x), PITCH_W - float(y)


def _hs_rot180_visible_area(va):
    if va is None:
        return None
    if isinstance(va, float) and np.isnan(va):
        return None
    if not isinstance(va, (list, tuple)) or len(va) == 0:
        return va
    pts = np.array(va, dtype=float).reshape(-1, 2)
    pts[:, 0] = PITCH_L - pts[:, 0]
    pts[:, 1] = PITCH_W - pts[:, 1]
    return pts.reshape(-1).tolist()


def extract_starting_xi_jersey_maps(events_list):
    """
    Build jersey maps using Starting XI events only.
    """
    jersey_by_id, jersey_by_name = {}, {}
    for e in events_list:
        t = e.get("type") or {}
        type_name = t.get("name") if isinstance(t, dict) else None
        if type_name != "Starting XI":
            continue
        lineup = ((e.get("tactics") or {}).get("lineup") or [])
        for pl in lineup:
            pobj = pl.get("player") or {}
            pid = _safe_int(pobj.get("id"))
            pname = pobj.get("name")
            jn = _safe_int(pl.get("jersey_number"))
            if jn is None:
                continue
            if pid is not None:
                jersey_by_id[pid] = jn
            if isinstance(pname, str) and pname.strip():
                jersey_by_name[str(pname)] = jn
    return jersey_by_id, jersey_by_name


def build_jersey_maps(events_list, fixed_by_id=None, fixed_by_name=None):
    # Base map: Starting XI
    jersey_by_id, jersey_by_name = extract_starting_xi_jersey_maps(events_list)

    # Enrich from regular events too.
    for e in events_list:
        p = e.get("player") or {}
        pid = _safe_int(p.get("id"))
        pname = p.get("name")
        jn = _safe_int(p.get("jersey_number"))
        try:
            if pid is not None and jn is not None:
                jersey_by_id[pid] = jn
        except Exception:
            pass
        if pname and jn is not None:
            try:
                jersey_by_name[str(pname)] = jn
            except Exception:
                pass

    # Add normalized + surname keys for robust matching.
    extra = {}
    for k, v in list(jersey_by_name.items()):
        nk = _norm_name(k)
        if nk:
            extra[nk] = v
        sn = _surname(k)
        if sn:
            extra[sn] = v
            nsn = _norm_name(sn)
            if nsn:
                extra[nsn] = v
    jersey_by_name.update(extra)

    # Manual overrides (optional)
    fixed_by_id = fixed_by_id or {}
    fixed_by_name = fixed_by_name or {}
    for pid, jn in fixed_by_id.items():
        try:
            pid_i = _safe_int(pid)
            jn_i = _safe_int(jn)
            if pid_i is not None and jn_i is not None:
                jersey_by_id[pid_i] = jn_i
        except Exception:
            pass
    for nm, jn in fixed_by_name.items():
        try:
            jersey_by_name[str(nm)] = int(jn)
            nn = _norm_name(str(nm))
            if nn:
                jersey_by_name[nn] = int(jn)
            sn = _surname(str(nm))
            if sn:
                jersey_by_name[sn] = int(jn)
        except Exception:
            pass

    return jersey_by_id, jersey_by_name


def make_viz_events_df(events_list, jersey_by_id=None, jersey_by_name=None):
    jersey_by_id = jersey_by_id or {}
    jersey_by_name = jersey_by_name or {}

    rows = []
    for e in events_list:
        p = e.get("player") or {}
        t = e.get("team") or {}
        et = e.get("type") or {}
        loc = e.get("location") if isinstance(e.get("location"), (list, tuple)) else None

        end_loc = None
        if isinstance(e.get("pass"), dict):
            end_loc = e["pass"].get("end_location")
        if end_loc is None and isinstance(e.get("carry"), dict):
            end_loc = e["carry"].get("end_location")
        if end_loc is None and isinstance(e.get("shot"), dict):
            end_loc = e["shot"].get("end_location")

        pid = p.get("id")
        pname = p.get("name")
        jn = np.nan
        try:
            if pid is not None and int(pid) in jersey_by_id:
                jn = jersey_by_id[int(pid)]
            elif pname in jersey_by_name:
                jn = jersey_by_name[pname]
            elif _norm_name(pname) in jersey_by_name:
                jn = jersey_by_name[_norm_name(pname)]
            elif _surname(pname) in jersey_by_name:
                jn = jersey_by_name[_surname(pname)]
            elif _norm_name(_surname(pname)) in jersey_by_name:
                jn = jersey_by_name[_norm_name(_surname(pname))]
            elif p.get("jersey_number") is not None:
                jn = int(p.get("jersey_number"))
        except Exception:
            pass

        rows.append(
            {
                "id": e.get("id"),
                "index": e.get("index"),
                "period": e.get("period"),
                "timestamp": e.get("timestamp"),
                "team_name": t.get("name"),
                "event_player_name": pname,
                "event_jersey_number": jn,
                "type_name": et.get("name"),
                "event_x": (float(loc[0]) if isinstance(loc, (list, tuple)) and len(loc) >= 2 else np.nan),
                "event_y": (float(loc[1]) if isinstance(loc, (list, tuple)) and len(loc) >= 2 else np.nan),
                "event_end_x": (float(end_loc[0]) if isinstance(end_loc, (list, tuple)) and len(end_loc) >= 2 else np.nan),
                "event_end_y": (float(end_loc[1]) if isinstance(end_loc, (list, tuple)) and len(end_loc) >= 2 else np.nan),
            }
        )

    return pd.DataFrame(rows)


def make_360_indexes(three_sixty_list):
    frames_by_id = {}
    visible_by_id = {}
    for s in three_sixty_list or []:
        eid = s.get("event_uuid") or s.get("id")
        if not eid:
            continue
        visible_by_id[eid] = s.get("visible_area")
        ff = s.get("freeze_frame") or []
        fr_rows = []
        for p in ff:
            loc = p.get("location")
            if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                x, y = float(loc[0]), float(loc[1])
            else:
                x, y = np.nan, np.nan
            fr_rows.append(
                {
                    "x": x,
                    "y": y,
                    "teammate": bool(p.get("teammate", False)),
                    "actor": bool(p.get("actor", False)),
                    "keeper": bool(p.get("keeper", False)),
                }
            )
        frames_by_id[eid] = pd.DataFrame(fr_rows)
    return frames_by_id, visible_by_id


def draw_event_keep_style_half_switch_full(
    df_events,
    event_id,
    frames_by_id,
    visible_by_id,
    *,
    facecolor="#0e1117",
    linecolor="#c7d5cc",
    actor_color="gold",
    reference_team=None,
    ref_by_period=None,
    team_colors=None,
    show_badge=True,
    show_header=True,
    show_ref=False,
    show_actor_source=True,
):
    if team_colors is None:
        all_teams = df_events.get("team_name", pd.Series(dtype=object)).dropna().unique().tolist()
        team_colors = resolve_team_colors(all_teams)

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
    opponent_team = next((t for t in teams_in_match if str(t) != str(team)), None)
    if reference_team is None:
        reference_team = teams_in_match[0] if teams_in_match else team

    team_color = team_colors.get(str(team), "white")
    opp_color = team_colors.get(str(opponent_team), "white")

    period = int(ev.get("period") or 1)
    if isinstance(ref_by_period, dict) and ref_by_period:
        ref = ref_by_period.get(period, reference_team)
    else:
        if period == 2:
            opp_ref = next((t for t in teams_in_match if str(t) != str(reference_team)), reference_team)
            ref = opp_ref
        else:
            ref = reference_team

    flip = str(team) != str(ref)
    if flip:
        sx, sy = _hs_rot180_xy(sx, sy)
        ex, ey = _hs_rot180_xy(ex, ey)

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
    actor_xy_from_360 = None
    if has_360:
        va = visible_by_id.get(event_id) if visible_by_id is not None else None
        if flip:
            va = _hs_rot180_visible_area(va)
        if va:
            poly = np.array(va).reshape(-1, 2)
            pitch.polygon([poly], ax=ax, color="gray", ec="white", lw=3, linestyle="-.", alpha=0.2, zorder=1)

        fr_local = fr
        if flip:
            fr_local = fr_local.copy()
            fr_local["x"] = PITCH_L - fr_local["x"].astype(float)
            fr_local["y"] = PITCH_W - fr_local["y"].astype(float)

        # Prefer actor location from 360 if present.
        actor_rows = fr_local[fr_local.get("actor", False) == True] if isinstance(fr_local, pd.DataFrame) else None
        if actor_rows is not None and len(actor_rows) > 0:
            ax0 = actor_rows.iloc[0].get("x")
            ay0 = actor_rows.iloc[0].get("y")
            if _is_finite_xy(ax0, ay0):
                actor_xy_from_360 = (float(ax0), float(ay0))

        for _, r in fr_local.iterrows():
            if bool(r.get("actor", False)):
                continue
            teammate = bool(r.get("teammate", False))
            keeper = bool(r.get("keeper", False))
            player_team = team if teammate else opponent_team
            color = team_colors.get(str(player_team).strip(), "white")
            marker = "D" if keeper else "o"
            size = 90 if keeper else 70
            ax.scatter(
                r["x"],
                r["y"],
                c=color,
                s=size,
                marker=marker,
                edgecolors="black",
                linewidths=0.8,
                zorder=3,
            )

    # Actor fallback order:
    # 1) 360 actor point
    # 2) event location
    # 3) previous pass end point (for Ball Receipt*)
    actor_source = None
    if actor_xy_from_360 is not None:
        sx, sy = actor_xy_from_360
        actor_source = "freeze_frame_actor"
    elif _is_finite_xy(sx, sy):
        actor_source = "event_location"
    else:
        if etype == "Ball Receipt*" and "index" in ev:
            cur_idx = ev.get("index")
            try:
                cur_idx = float(cur_idx)
            except Exception:
                cur_idx = None
            if cur_idx is not None:
                cand = df_events[
                    (df_events["period"] == ev.get("period"))
                    & (df_events["type_name"] == "Pass")
                    & (df_events["index"] <= cur_idx)
                ].sort_values("index", ascending=False)
                if len(cand) > 0:
                    row = cand.iloc[0]
                    fx, fy = row.get("event_end_x"), row.get("event_end_y")
                    if flip:
                        fx, fy = _hs_rot180_xy(fx, fy)
                    if _is_finite_xy(fx, fy):
                        sx, sy = float(fx), float(fy)
                        actor_source = "prev_pass_end_fallback"

    if _is_finite_xy(sx, sy):
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

    has_end = _is_finite_xy(sx, sy) and _is_finite_xy(ex, ey)
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

    if show_header:
        fig.text(
            0.87,
            0.95,
            f"time: {ts}",
            ha="right",
            va="top",
            color="white",
            fontsize=12,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc="#111827", ec="white", alpha=0.65),
        )

    if show_ref:
        ax.text(
            0.5,
            -0.04,
            f"ref={ref}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            color="white",
            fontsize=9,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="#111827", ec="white", alpha=0.9),
        )
    if show_actor_source and actor_source is not None:
        ax.text(
            0.98,
            0.02,
            f"actor_source={actor_source}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            color="white",
            fontsize=8,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.2", fc="#111827", ec="white", alpha=0.55),
        )
    return fig, ax
