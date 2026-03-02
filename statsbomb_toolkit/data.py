from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

PITCH_L = 120.0
PITCH_W = 80.0


def build_freeze_df(
    events: pd.DataFrame,
    frames: pd.DataFrame,
    visible: pd.DataFrame,
    df_lineup: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge events + 360 frames into one DataFrame for plotting."""
    e = events[
        [
            "id",
            "timestamp",
            "team_name",
            "player_id",
            "player_name",
            "type_name",
            "x",
            "y",
            "end_x",
            "end_y",
        ]
    ].copy()

    e = e.rename(
        columns={
            "x": "event_x",
            "y": "event_y",
            "end_x": "event_end_x",
            "end_y": "event_end_y",
        }
    )

    df_ff = frames.merge(visible, on="id", how="left").merge(e, on="id", how="left")
    if df_lineup is not None and not df_lineup.empty:
        df_ff = df_ff.merge(df_lineup[["player_id", "jersey_number"]], on="player_id", how="left")

    if "timestamp" in df_ff.columns:
        df_ff["timestring"] = df_ff["timestamp"].astype(str).str.slice(0, 5)
    return df_ff


def build_events_df(
    events: pd.DataFrame,
    df_lineup: pd.DataFrame,
    frames: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Prepare compact events table used by visual and LLM pipeline."""
    df_ev = events[
        [
            "id",
            "index",
            "period",
            "timestamp",
            "team_name",
            "player_id",
            "player_name",
            "type_name",
            "x",
            "y",
            "end_x",
            "end_y",
        ]
    ].copy()

    df_ev = df_ev.rename(
        columns={
            "player_id": "event_player_id",
            "player_name": "event_player_name",
            "x": "event_x",
            "y": "event_y",
            "end_x": "event_end_x",
            "end_y": "event_end_y",
        }
    )

    df_ev = df_ev.merge(
        df_lineup[["player_id", "jersey_number"]].rename(
            columns={"player_id": "event_player_id", "jersey_number": "event_jersey_number"}
        ),
        on="event_player_id",
        how="left",
    )

    if frames is not None and len(frames):
        ids_with_360 = set(frames["id"].unique())
        df_ev["has_360"] = df_ev["id"].isin(ids_with_360)
    else:
        df_ev["has_360"] = False

    df_ev["timestring"] = df_ev["timestamp"].astype(str).str.slice(0, 5)
    return df_ev


def build_360_index(frames: pd.DataFrame, visible: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Index 360 data by event id for O(1) lookup."""
    frames_by_id = {eid: g.copy() for eid, g in frames.groupby("id")}
    visible_by_id = dict(zip(visible["id"], visible["visible_area"]))
    return frames_by_id, visible_by_id


def rot180_xy(x: Any, y: Any, L: float = PITCH_L, W: float = PITCH_W) -> tuple[Any, Any]:
    if x is None or y is None:
        return x, y
    try:
        return (L - float(x)), (W - float(y))
    except Exception:
        return x, y


def rot180_point_list_xy(pt: Any, L: float = PITCH_L, W: float = PITCH_W) -> Any:
    if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
        return pt
    x, y = pt
    x2, y2 = rot180_xy(x, y, L, W)
    return [x2, y2]


def rot180_visible_area(va: Any, L: float = PITCH_L, W: float = PITCH_W) -> Any:
    if va is None:
        return None
    if not isinstance(va, (list, tuple)) or len(va) == 0:
        return va
    arr = np.array(va, dtype=float).reshape(-1, 2)
    arr[:, 0] = L - arr[:, 0]
    arr[:, 1] = W - arr[:, 1]
    return arr.reshape(-1).tolist()


def _is_xy_pair(v: Any) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(
        isinstance(t, (int, float, np.number)) for t in v
    )


def flip_event_dict_inplace(d: dict[str, Any], L: float = PITCH_L, W: float = PITCH_W) -> None:
    """Flip every location-like coordinate in event dict by 180 degrees."""
    if not isinstance(d, dict):
        return

    for k, v in list(d.items()):
        if k == "visible_area":
            d[k] = rot180_visible_area(v, L, W)
            continue

        if k == "location" and _is_xy_pair(v):
            d[k] = rot180_point_list_xy(v, L, W)
            continue

        if isinstance(k, str) and k.endswith("_location") and _is_xy_pair(v):
            d[k] = rot180_point_list_xy(v, L, W)
            continue

        if isinstance(v, dict):
            flip_event_dict_inplace(v, L, W)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    flip_event_dict_inplace(item, L, W)


def standardize_events_and_360(
    events_raw: list[dict[str, Any]],
    three_sixty_raw: list[dict[str, Any]] | None,
    reference_team: str = "Scotland",
    L: float = PITCH_L,
    W: float = PITCH_W,
    add_debug_flip_flag: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, bool]]:
    """
    Flip events and 360 so one team is always in the same attack direction.
    """
    events_std = copy.deepcopy(events_raw)
    flip_map: dict[str, bool] = {}

    for e in events_std:
        eid = e.get("id")
        team_name = (e.get("team", {}) or {}).get("name")
        flip = str(team_name) != str(reference_team)
        if eid is not None:
            flip_map[eid] = flip
        if flip:
            flip_event_dict_inplace(e, L, W)
        if add_debug_flip_flag:
            e["_flip180"] = bool(flip)
            e["_reference_team"] = reference_team

    three_sixty_std = None
    if three_sixty_raw is not None:
        three_sixty_std = copy.deepcopy(three_sixty_raw)
        for s in three_sixty_std:
            eid = s.get("event_uuid") or s.get("id")
            flip = bool(flip_map.get(eid, False))
            if flip:
                if "visible_area" in s:
                    s["visible_area"] = rot180_visible_area(s.get("visible_area"), L, W)
                ff = s.get("freeze_frame")
                if isinstance(ff, list):
                    for p in ff:
                        if isinstance(p, dict) and "location" in p:
                            p["location"] = rot180_point_list_xy(p["location"], L, W)
            if add_debug_flip_flag:
                s["_flip180"] = bool(flip)
                s["_reference_team"] = reference_team

    return events_std, three_sixty_std, flip_map


def clean_event_recursive(
    x: Any,
    *,
    keep_event_id: bool = True,
    keep_id_paths: set[tuple[str, ...]] | None = None,
    drop_keys_global: set[str] | None = None,
    drop_paths: set[tuple[str, ...]] | None = None,
    _path: tuple[str, ...] = (),
) -> Any:
    """
    Remove noisy fields before prompt construction.
    """
    if keep_id_paths is None:
        keep_id_paths = {("player", "id")}
    if drop_keys_global is None:
        drop_keys_global = {"possession", "possession_team"}
    if drop_paths is None:
        drop_paths = {("pass", "angle")}

    if isinstance(x, dict):
        out: dict[str, Any] = {}
        for k, v in x.items():
            cur_path = _path + (k,)
            if k in drop_keys_global:
                continue
            if cur_path in drop_paths:
                continue

            if k == "id":
                if keep_event_id and _path == ():
                    out[k] = v
                elif cur_path in keep_id_paths:
                    out[k] = v
                continue

            out[k] = clean_event_recursive(
                v,
                keep_event_id=False,
                keep_id_paths=keep_id_paths,
                drop_keys_global=drop_keys_global,
                drop_paths=drop_paths,
                _path=cur_path,
            )
        return out

    if isinstance(x, list):
        return [
            clean_event_recursive(
                v,
                keep_event_id=False,
                keep_id_paths=keep_id_paths,
                drop_keys_global=drop_keys_global,
                drop_paths=drop_paths,
                _path=_path,
            )
            for v in x
        ]

    return x
