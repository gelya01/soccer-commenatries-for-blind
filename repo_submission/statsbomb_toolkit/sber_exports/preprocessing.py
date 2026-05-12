"""
Preprocessing, standardization, zone logic, feature engineering and payload builders.
Self-contained module extracted and cleaned from sber_chain_eval workflow.
"""

from __future__ import annotations

import copy
import math
from typing import Any

PITCH_L = 120.0
PITCH_W = 80.0

NON_PLAY_EVENT_TYPES = {
    "Starting XI",
    "Half Start",
    "Half End",
    "Tactical Shift",
    "Substitution",
    "Player On",
    "Player Off",
    "Injury Stoppage",
    "Bad Behaviour",
    "Referee Ball Drop",
}

LOW_SIGNAL_EVENT_TYPES = {"Ball Receipt*", "Pressure"}

KEY_ACTION_EVENT_TYPES = {
    "Pass",
    "Carry",
    "Dribble",
    "Shot",
    "Interception",
    "Ball Recovery",
    "Dispossessed",
    "Miscontrol",
    "Foul Won",
    "Foul Committed",
    "Offside",
    "Clearance",
    "Goal Keeper",
}

ALWAYS_HIGH_TYPES = {
    "Shot",
    "Foul Won",
    "Foul Committed",
    "Offside",
    "Interception",
    "Ball Recovery",
    "Dispossessed",
    "Miscontrol",
    "Error",
}

LOW_TYPES = {"Ball Receipt*", "Pressure"}

SET_PLAY_PATTERNS = {
    "From Throw In",
    "From Free Kick",
    "From Corner",
    "From Goal Kick",
    "From Kick Off",
}


def rot180_xy(x: float | None, y: float | None) -> tuple[float | None, float | None]:
    if x is None or y is None:
        return x, y
    return PITCH_L - float(x), PITCH_W - float(y)


def rot180_point_list_xy(loc: Any):
    if not isinstance(loc, (list, tuple)) or len(loc) < 2:
        return loc
    x, y = rot180_xy(float(loc[0]), float(loc[1]))
    out = list(loc)
    out[0] = x
    out[1] = y
    return out


def rot180_visible_area(va: Any):
    if not isinstance(va, list) or len(va) < 6:
        return va
    out = []
    for i in range(0, len(va), 2):
        x = va[i]
        y = va[i + 1] if i + 1 < len(va) else None
        if y is None:
            continue
        xr, yr = rot180_xy(float(x), float(y))
        out.extend([xr, yr])
    return out


def _flip_event_dict_inplace(ev: dict):
    if "location" in ev:
        ev["location"] = rot180_point_list_xy(ev.get("location"))
    if isinstance(ev.get("pass"), dict) and "end_location" in ev["pass"]:
        ev["pass"]["end_location"] = rot180_point_list_xy(ev["pass"].get("end_location"))
    if isinstance(ev.get("carry"), dict) and "end_location" in ev["carry"]:
        ev["carry"]["end_location"] = rot180_point_list_xy(ev["carry"].get("end_location"))
    if isinstance(ev.get("shot"), dict) and "end_location" in ev["shot"]:
        ev["shot"]["end_location"] = rot180_point_list_xy(ev["shot"].get("end_location"))


def should_flip_for_event(ev: dict, ref_by_period: dict[int, str]) -> bool:
    period = int(ev.get("period", 1) or 1)
    reference_team = ref_by_period.get(period, ref_by_period.get(1, "Scotland"))
    team_name = ((ev.get("team") or {}).get("name"))
    return str(team_name) != str(reference_team)


def standardize_events_and_360_by_half(
    events_raw: list[dict],
    three_sixty_raw: list[dict],
    ref_by_period: dict[int, str],
):
    events_std = copy.deepcopy(events_raw)

    # Enrich missing player.jersey_number:
    # 1) collect from any event where jersey is already present
    # 2) extend from Starting XI lineups
    jersey_by_player_id = {}
    for ev in events_std:
        p = ev.get("player")
        if not isinstance(p, dict):
            continue
        player_id = p.get("id")
        jersey = p.get("jersey_number")
        if player_id is not None and jersey is not None:
            jersey_by_player_id[player_id] = jersey

    for ev in events_std:
        if get_type_name(ev) != "Starting XI":
            continue
        lineup = ((ev.get("tactics") or {}).get("lineup")) or []
        for p in lineup:
            player_id = ((p.get("player") or {}).get("id"))
            jersey = p.get("jersey_number")
            if player_id is not None and jersey is not None:
                jersey_by_player_id[player_id] = jersey

    for ev in events_std:
        pl = ev.get("player")
        if not isinstance(pl, dict):
            continue
        if pl.get("jersey_number") is not None:
            continue
        pid = pl.get("id")
        if pid in jersey_by_player_id:
            pl["jersey_number"] = jersey_by_player_id[pid]

    flip_map: dict[str, bool] = {}

    for ev in events_std:
        eid = ev.get("id")
        flip = should_flip_for_event(ev, ref_by_period)
        if eid:
            flip_map[eid] = flip
        if flip:
            _flip_event_dict_inplace(ev)
        period = int(ev.get("period", 1) or 1)
        ev["_flip180"] = bool(flip)
        ev["_reference_team"] = ref_by_period.get(period, ref_by_period.get(1, "Scotland"))

    events_raw_by_id = {e.get("id"): e for e in events_raw if e.get("id")}
    three_sixty_std = copy.deepcopy(three_sixty_raw)
    for s in three_sixty_std:
        eid = s.get("event_uuid") or s.get("id")
        flip = bool(flip_map.get(eid, False))
        if flip:
            if "visible_area" in s:
                s["visible_area"] = rot180_visible_area(s.get("visible_area"))
            ff = s.get("freeze_frame")
            if isinstance(ff, list):
                for p in ff:
                    if isinstance(p, dict) and "location" in p:
                        p["location"] = rot180_point_list_xy(p.get("location"))
        period = int((events_raw_by_id.get(eid) or {}).get("period", 1) or 1)
        s["_flip180"] = bool(flip)
        s["_reference_team"] = ref_by_period.get(period, ref_by_period.get(1, "Scotland"))

    return events_std, three_sixty_std, flip_map


def clean_event_recursive(
    obj: Any,
    *,
    keep_event_id: bool = True,
    keep_id_paths: set[tuple[str, ...]] | None = None,
    drop_keys_global: set[str] | None = None,
    drop_paths: set[tuple[str, ...]] | None = None,
    _path: tuple[str, ...] = (),
):
    keep_id_paths = keep_id_paths or set()
    drop_keys_global = drop_keys_global or set()
    drop_paths = drop_paths or set()

    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            path = _path + (k,)
            if path in drop_paths:
                continue
            if k in drop_keys_global:
                continue
            if k == "id" and not keep_event_id and _path == ():
                continue
            if k == "id" and path not in keep_id_paths and _path != ():
                continue
            out[k] = clean_event_recursive(
                v,
                keep_event_id=keep_event_id,
                keep_id_paths=keep_id_paths,
                drop_keys_global=drop_keys_global,
                drop_paths=drop_paths,
                _path=path,
            )
        return out
    if isinstance(obj, list):
        return [
            clean_event_recursive(
                x,
                keep_event_id=keep_event_id,
                keep_id_paths=keep_id_paths,
                drop_keys_global=drop_keys_global,
                drop_paths=drop_paths,
                _path=_path,
            )
            for x in obj
        ]
    return obj


def clean_event_for_llm(ev: dict) -> dict:
    return clean_event_recursive(
        ev,
        keep_event_id=True,
        keep_id_paths={("player", "id")},
        drop_keys_global={"possession", "possession_team"},
        drop_paths={("pass", "angle")},
    )


def get_type_name(ev):
    t = ev.get("type") or {}
    return t.get("name") if isinstance(t, dict) else str(t)


def get_play_pattern(ev):
    p = ev.get("play_pattern") or {}
    return p.get("name") if isinstance(p, dict) else str(p)


def get_loc(ev):
    loc = ev.get("location")
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return None


def event_location(ev):
    return get_loc(ev)


def actor_location(sb360):
    if not isinstance(sb360, dict):
        return None
    ff = sb360.get("freeze_frame") or []
    for p in ff:
        if isinstance(p, dict) and p.get("actor") is True:
            loc = p.get("location")
            if isinstance(loc, (list, tuple)) and len(loc) >= 2:
                return float(loc[0]), float(loc[1])
    return None


def dist(a, b):
    if a is None or b is None:
        return None
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def get_end_loc(ev):
    et = get_type_name(ev)
    if et == "Pass":
        loc = (ev.get("pass") or {}).get("end_location")
    elif et == "Carry":
        loc = (ev.get("carry") or {}).get("end_location")
    elif et == "Shot":
        loc = (ev.get("shot") or {}).get("end_location")
    else:
        loc = None
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        return float(loc[0]), float(loc[1])
    return None


def zone_label_abs(loc):
    if loc is None:
        return "unknown"
    x, y = float(loc[0]), float(loc[1])

    if (x <= 2 or x >= 118) and (y <= 2 or y >= 78):
        return "corner_zone"
    if y <= 1:
        return "throw_in_top"
    if y >= 79:
        return "throw_in_bottom"
    if x <= 6 and 30 <= y <= 50:
        return "left_goal_area"
    if x >= 114 and 30 <= y <= 50:
        return "right_goal_area"
    if x <= 18 and 18 <= y <= 62:
        return "left_box"
    if x >= 102 and 18 <= y <= 62:
        return "right_box"
    if 18 < x <= 24 and 18 <= y <= 62:
        return "pre_box_left"
    if 96 <= x < 102 and 18 <= y <= 62:
        return "pre_box_right"
    if 12 <= x <= 24 and 18 <= y < 26:
        return "left_box_corner_top"
    if 12 <= x <= 24 and 54 < y <= 62:
        return "left_box_corner_bottom"
    if 96 <= x <= 108 and 18 <= y < 26:
        return "right_box_corner_top"
    if 96 <= x <= 108 and 54 < y <= 62:
        return "right_box_corner_bottom"
    if (x - 60.0) ** 2 + (y - 40.0) ** 2 <= 10.0**2:
        return "center_circle_zone"

    half = "left_half" if x < 60 else "right_half"
    if y < 18:
        lane = "top_flank"
    elif y > 62:
        lane = "bottom_flank"
    else:
        lane = "center_lane"
    return f"{half}:{lane}"


def zone_label_team(abs_zone, own_goal_x):
    left_is_own = own_goal_x == 0.0
    remap = {
        "left_goal_area": "own_goal_area" if left_is_own else "opponent_goal_area",
        "right_goal_area": "opponent_goal_area" if left_is_own else "own_goal_area",
        "left_box": "own_box" if left_is_own else "opponent_box",
        "right_box": "opponent_box" if left_is_own else "own_box",
        "pre_box_left": "pre_own_box" if left_is_own else "pre_opponent_box",
        "pre_box_right": "pre_opponent_box" if left_is_own else "pre_own_box",
        "left_box_corner_top": "own_box_corner_top" if left_is_own else "opponent_box_corner_top",
        "left_box_corner_bottom": "own_box_corner_bottom" if left_is_own else "opponent_box_corner_bottom",
        "right_box_corner_top": "opponent_box_corner_top" if left_is_own else "own_box_corner_top",
        "right_box_corner_bottom": "opponent_box_corner_bottom" if left_is_own else "own_box_corner_bottom",
    }
    return remap.get(abs_zone, abs_zone)


def _own_opp_goal_x_for_event(ev: dict, ref_by_period: dict[int, str]):
    period = int(ev.get("period") or 1)
    team = (ev.get("team") or {}).get("name")
    ref = ev.get("_reference_team") or ref_by_period.get(period, ref_by_period.get(1, "Scotland"))
    team_is_ref = str(team) == str(ref)
    if period == 1:
        own_goal_x = 0.0 if team_is_ref else 120.0
    else:
        own_goal_x = 120.0 if team_is_ref else 0.0
    opp_goal_x = 120.0 - own_goal_x
    return own_goal_x, opp_goal_x, ref


def pass_outcome_name(ev):
    out = (ev.get("pass") or {}).get("outcome") or {}
    return out.get("name") if isinstance(out, dict) else (str(out) if out else None)


def rel_zone(x, own_goal_x, opp_goal_x):
    del opp_goal_x
    if x is None:
        return "unknown"
    if 58 <= x <= 62:
        return "center_line"
    if own_goal_x == 0.0:
        return "own_half" if x < 60 else "opponent_half"
    return "own_half" if x > 60 else "opponent_half"


def lane_zone(y):
    if y is None:
        return "unknown"
    if y < 18:
        return "top_flank"
    if y > 62:
        return "bottom_flank"
    return "center_lane"


def movement_label(dx, dy, attack_sign, min_progress=3.0):
    if dx is None:
        return "unknown"
    fwd = dx * attack_sign
    if abs(fwd) < min_progress and abs(dy or 0) < min_progress:
        return "short_or_static"
    if fwd >= min_progress:
        return "forward"
    if fwd <= -min_progress:
        return "backward"
    return "lateral"


def movement_compass(dx, dy, attack_sign, thr_fwd=3.0, thr_lat=3.0):
    if dx is None or dy is None:
        return "unknown"
    fwd = dx * attack_sign
    lat = -(dy * attack_sign)
    fwd_tag = "forward" if fwd >= thr_fwd else ("backward" if fwd <= -thr_fwd else "neutral")
    lat_tag = "left" if lat >= thr_lat else ("right" if lat <= -thr_lat else "center")
    if fwd_tag == "neutral" and lat_tag == "center":
        return "short_or_static"
    if fwd_tag == "neutral":
        return lat_tag
    if lat_tag == "center":
        return fwd_tag
    return f"{fwd_tag}_{lat_tag}"


def pass_style_ru(ev_clean, pass_len_val):
    p = ev_clean.get("pass") or {}
    h = ((p.get("height") or {}).get("name") or "").strip().lower()
    if "high" in h:
        if pass_len_val is not None and pass_len_val >= 25:
            return "заброс"
        return "верхом"
    if "low" in h or "ground" in h:
        return "низом"
    return "пас"


def _surname(name: str):
    if not isinstance(name, str) or not name.strip():
        return name
    return name.strip().split()[-1]


def to_surname_event(ev_clean: dict) -> dict:
    ev = copy.deepcopy(ev_clean)
    p = ev.get("player")
    if isinstance(p, dict) and isinstance(p.get("name"), str):
        p["name"] = _surname(p["name"])
    ppass = ev.get("pass")
    if isinstance(ppass, dict):
        rec = ppass.get("recipient")
        if isinstance(rec, dict) and isinstance(rec.get("name"), str):
            rec["name"] = _surname(rec["name"])
    return ev


def should_skip_derive(ev_clean: dict):
    et = get_type_name(ev_clean)
    if et in NON_PLAY_EVENT_TYPES:
        return True, "non_play_event"
    loc = get_loc(ev_clean)
    if loc is None and et not in {"Goal Keeper", "Shot"}:
        return True, "no_location"
    return False, None


def event_features(ev: dict, ref_by_period: dict[int, str]):
    et = get_type_name(ev)
    pp = get_play_pattern(ev)
    loc = get_loc(ev)
    end_loc = get_end_loc(ev)
    own_goal_x, opp_goal_x, reference_team = _own_opp_goal_x_for_event(ev, ref_by_period)
    forward_sign = 1.0 if opp_goal_x > own_goal_x else -1.0
    start_abs = zone_label_abs(loc)
    end_abs = zone_label_abs(end_loc) if end_loc else None
    f = {
        "event_type": et,
        "play_pattern": pp,
        "is_set_play": int(pp in SET_PLAY_PATTERNS),
        "is_always_high": int(et in ALWAYS_HIGH_TYPES),
        "is_low": int(et in LOW_TYPES),
        "start_zone_abs": start_abs,
        "end_zone_abs": end_abs,
        "start_zone_team": zone_label_team(start_abs, own_goal_x),
        "end_zone_team": zone_label_team(end_abs, own_goal_x) if end_abs else None,
        "has_end_loc": int(end_loc is not None),
        "forward_dx_abs": None,
        "progress_to_opp_goal": None,
        "dist": None,
        "pass_outcome": pass_outcome_name(ev),
        "own_goal_x": own_goal_x,
        "opp_goal_x": opp_goal_x,
        "reference_team": reference_team,
    }
    if loc and end_loc:
        dx = end_loc[0] - loc[0]
        dy = end_loc[1] - loc[1]
        prog = dx * forward_sign
        f["forward_dx_abs"] = round(dx, 3)
        f["progress_to_opp_goal"] = round(prog, 3)
        f["dist"] = round(math.hypot(dx, dy), 3)
    f["entered_opponent_box"] = int(f["end_zone_team"] == "opponent_box" and f["start_zone_team"] != "opponent_box")
    f["left_own_box"] = int(f["start_zone_team"] == "own_box" and f["end_zone_team"] != "own_box")
    f["negative_outcome"] = int((f["pass_outcome"] or "") in {"Incomplete", "Out", "Pass Offside", "Unknown"})
    return f


def chain_features(chain_ids, events_by_id: dict[str, dict], ref_by_period: dict[int, str]):
    feats = [event_features(events_by_id[eid], ref_by_period) for eid in chain_ids if eid in events_by_id]
    n = len(feats)
    high = sum(x["is_always_high"] for x in feats)
    setp = sum(x["is_set_play"] for x in feats)
    low = sum(x["is_low"] for x in feats)
    entered_box = sum(x["entered_opponent_box"] for x in feats)
    return {
        "events_n": n,
        "high_events_n": high,
        "set_play_events_n": setp,
        "low_events_n": low,
        "entered_opponent_box_n": entered_box,
        "importance_score_rule": int(3 * high + 2 * entered_box + setp - 0.5 * low),
    }


def build_chain_payload_v4(
    chain_ids: list[str],
    *,
    events_by_id: dict[str, dict],
    events_clean_by_id: dict[str, dict],
    sb360_by_id: dict[str, dict] | None,
    bad_ids: set[str] | None,
    ref_by_period: dict[int, str],
):
    sb360_by_id = sb360_by_id or {}
    bad_ids = bad_ids or set()
    items = []

    for eid in chain_ids:
        ev_std = events_by_id.get(eid)
        ev_clean_raw = events_clean_by_id.get(eid)
        if not ev_std or not ev_clean_raw:
            continue
        ev_clean = to_surname_event(ev_clean_raw)
        et = get_type_name(ev_clean)
        sb = None if (eid in bad_ids) else sb360_by_id.get(eid)
        skip, skip_reason = should_skip_derive(ev_clean)

        base_item = {
            "event_id": eid,
            "event_json": ev_clean,
            "sb360_json": sb,
            "derived": {
                "quality_flags": {
                    "sb360_is_bad_filtered": int(eid in bad_ids),
                    "skip_derive": int(skip),
                    "skip_reason": skip_reason,
                },
                "event_semantics": {
                    "event_type": et,
                    "is_non_play": int(et in NON_PLAY_EVENT_TYPES),
                    "is_low_signal": int(et in LOW_SIGNAL_EVENT_TYPES),
                    "is_key_action": int(et in KEY_ACTION_EVENT_TYPES),
                },
            },
        }

        if skip:
            items.append(base_item)
            continue

        period = int(ev_clean.get("period", 1) or 1)
        team = ((ev_clean.get("team") or {}).get("name"))
        own_goal_x, opp_goal_x, _ = _own_opp_goal_x_for_event(ev_clean, ref_by_period)
        attack_sign = +1 if opp_goal_x > own_goal_x else -1

        loc = get_loc(ev_clean)
        end_loc = get_end_loc(ev_clean)
        sx = loc[0] if loc else None
        sy = loc[1] if loc else None
        ex = end_loc[0] if end_loc else None
        ey = end_loc[1] if end_loc else None
        dx = (ex - sx) if (sx is not None and ex is not None) else None
        dy = (ey - sy) if (sy is not None and ey is not None) else None

        pass_obj = ev_clean.get("pass") or {}
        pass_len = pass_obj.get("length") if isinstance(pass_obj, dict) else None
        try:
            pass_len_val = float(pass_len) if pass_len is not None else None
        except Exception:
            pass_len_val = None

        start_rel = rel_zone(sx, own_goal_x, opp_goal_x)
        end_rel = rel_zone(ex, own_goal_x, opp_goal_x) if ex is not None else None
        start_lane = lane_zone(sy)
        end_lane = lane_zone(ey) if ey is not None else None
        zone_transition = f"{start_rel}->{end_rel}" if end_rel is not None else None

        base_item["derived"].update(
            {
                "orientation": {
                    "period": period,
                    "team": team,
                    "own_goal_x": own_goal_x,
                    "opp_goal_x": opp_goal_x,
                    "attack_sign": attack_sign,
                },
                "zones": {
                    "start_abs": zone_label_abs(loc),
                    "end_abs": zone_label_abs(end_loc) if end_loc else None,
                    "start_rel": start_rel,
                    "end_rel": end_rel,
                    "start_lane": start_lane,
                    "end_lane": end_lane,
                    "zone_transition": zone_transition,
                },
                "movement": {
                    "dx": dx,
                    "dy": dy,
                    "forward_delta": (dx * attack_sign) if dx is not None else None,
                    "lateral_delta": (-(dy * attack_sign)) if dy is not None else None,
                    "label": movement_label(dx, dy, attack_sign),
                    "compass": movement_compass(dx, dy, attack_sign),
                },
                "episode_signals": {
                    "is_long_pass": int(pass_len_val is not None and pass_len_val >= 25.0),
                    "pass_length": pass_len_val,
                    "pass_style_ru": pass_style_ru(ev_clean, pass_len_val) if et == "Pass" else None,
                    "pass_height_name": (((pass_obj.get("height") or {}).get("name")) if isinstance(pass_obj, dict) else None),
                    "pass_recipient": (((pass_obj.get("recipient") or {}).get("name")) if isinstance(pass_obj, dict) else None),
                    "pass_direction_compass": movement_compass(dx, dy, attack_sign) if et == "Pass" else None,
                    "pass_target_abs": zone_label_abs(end_loc) if (et == "Pass" and end_loc) else None,
                    "pass_target_rel": rel_zone(ex, own_goal_x, opp_goal_x) if (et == "Pass" and ex is not None) else None,
                    "switched_flank": int(start_lane != "unknown" and end_lane != "unknown" and start_lane != end_lane),
                    "entered_opponent_half": int(start_rel != "opponent_half" and end_rel == "opponent_half"),
                    "entered_opponent_box": int(zone_label_team(zone_label_abs(end_loc), own_goal_x) == "opponent_box")
                    if end_loc
                    else 0,
                },
            }
        )
        items.append(base_item)

    return {
        "chain_event_ids": chain_ids,
        "chain_features": chain_features(chain_ids, events_by_id, ref_by_period),
        "events": items,
    }
