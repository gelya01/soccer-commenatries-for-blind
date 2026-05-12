"""
Related-events graph, chain building, optional chunking and scheduling helpers.
Self-contained module extracted and cleaned from sber_chain_eval workflow.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict

import numpy as np
import pandas as pd

STOP_TYPES = {
    "Shot",
    "Offside",
    "Foul Committed",
    "Foul Won",
    "Interception",
    "Ball Recovery",
    "Dispossessed",
    "Miscontrol",
    "Clearance",
    "Goal Keeper",
}

SCHED_MAX_SILENCE_SEC = 12.0
SCHED_MIN_GAP_SEC = 2.5
SCHED_MIN_IMPORTANCE = 2


def build_related_graph(events_std: list[dict]):
    events_by_id = {e.get("id"): e for e in events_std if e.get("id")}
    outgoing = defaultdict(set)
    incoming = defaultdict(set)
    for e in events_std:
        eid = e.get("id")
        if not eid:
            continue
        rel = e.get("related_events") or []
        for rid in rel:
            if not rid:
                continue
            outgoing[eid].add(rid)
            incoming[rid].add(eid)
    neighbors = defaultdict(set)
    for eid in events_by_id:
        neighbors[eid] |= outgoing.get(eid, set())
        neighbors[eid] |= incoming.get(eid, set())
    return events_by_id, neighbors


def expand_chain(root_id: str, used: set[str], neighbors: dict[str, set[str]], allowed_ids: set[str]) -> set[str]:
    group = set()
    stack = [root_id]
    while stack:
        cur = stack.pop()
        if not cur or cur in group:
            continue
        if cur in used and cur != root_id:
            continue
        group.add(cur)
        for nb in neighbors.get(cur, set()):
            if nb in group or nb in used:
                continue
            if nb in allowed_ids:
                stack.append(nb)
    return group


def build_nonoverlap_chains(events_std: list[dict], neighbors: dict[str, set[str]]):
    events_by_id = {e.get("id"): e for e in events_std if e.get("id")}
    events_sorted = sorted(events_std, key=lambda e: (e.get("index", 10**9), e.get("id") or ""))
    used = set()
    chains = []
    allowed_ids = set(events_by_id.keys())
    for e in events_sorted:
        eid = e.get("id")
        if not eid or eid in used:
            continue
        group = expand_chain(eid, used, neighbors, allowed_ids)
        chain = sorted(group, key=lambda x: events_by_id.get(x, {}).get("index", 10**9))
        used.update(chain)
        chains.append(chain)
    return chains


def _event_type(item):
    return ((item.get("event_json") or {}).get("type") or {}).get("name")


def _to_seconds(ts):
    if not ts or not isinstance(ts, str):
        return None
    m = re.match(r"^(\d+):(\d+):(\d+)(?:\.(\d+))?$", ts.strip())
    if not m:
        return None
    hh, mm, ss, ms = m.groups()
    base = int(hh) * 3600 + int(mm) * 60 + int(ss)
    frac = float(f"0.{ms}") if ms else 0.0
    return base + frac


def split_chain_payload(payload, mode="related_time_cap", max_events=8, max_duration_sec=12.0, gap_sec=3.5):
    events = payload.get("events") or []
    if not events:
        return []
    if mode == "related_only":
        return [payload]

    chunks, cur = [], []
    t0, prev_t = None, None

    def flush():
        nonlocal cur, t0, prev_t
        if not cur:
            return
        p = copy.deepcopy(payload)
        p["events"] = cur
        p["chunk_size"] = len(cur)
        chunks.append(p)
        cur, t0, prev_t = [], None, None

    for ev in events:
        ts = (ev.get("event_json") or {}).get("timestamp")
        t = _to_seconds(ts)
        et = _event_type(ev)

        if not cur:
            cur = [ev]
            t0 = t
            prev_t = t
            continue

        dur = (t - t0) if (t is not None and t0 is not None) else None
        gap = (t - prev_t) if (t is not None and prev_t is not None) else None

        need_split = False
        if len(cur) >= max_events:
            need_split = True
        if dur is not None and dur > max_duration_sec:
            need_split = True
        if gap is not None and gap > gap_sec:
            need_split = True
        if mode == "fixed_time_window" and dur is not None and dur > (max_duration_sec * 0.75):
            need_split = True

        if need_split:
            flush()
            cur = [ev]
            t0 = t
            prev_t = t
        else:
            cur.append(ev)
            prev_t = t

        if et in STOP_TYPES and len(cur) >= 2:
            flush()

    flush()
    return chunks


def _chain_profile(chain):
    types = [_event_type(e) for e in chain.get("events", [])]
    has_critical = any(t in {"Shot", "Goal Keeper", "Offside", "Foul Committed", "Foul Won"} for t in types)
    has_transition = any(t in {"Interception", "Ball Recovery", "Dispossessed", "Miscontrol", "Clearance"} for t in types)
    low_only = all(t in {"Ball Receipt*", "Pressure"} for t in types) if types else False
    if has_critical:
        return "critical"
    if has_transition:
        return "transition"
    if low_only:
        return "low_signal"
    return "build_up"


def timing_budget_for_chain(chain, wpm_fast=184.0):
    events = chain.get("events", [])
    t_start = (events[0].get("event_json") or {}).get("timestamp") if events else ""
    t_end = (events[-1].get("event_json") or {}).get("timestamp") if events else ""
    s0 = _to_seconds(t_start)
    s1 = _to_seconds(t_end)
    window_sec = max((s1 - s0), 0.0) if (s0 is not None and s1 is not None) else 0.0

    profile = _chain_profile(chain)
    if profile == "critical":
        speech_share, min_sec, max_sec = 0.85, 3.5, 11.0
    elif profile == "transition":
        speech_share, min_sec, max_sec = 0.75, 3.0, 9.0
    elif profile == "build_up":
        speech_share, min_sec, max_sec = 0.60, 2.5, 8.0
    else:
        speech_share, min_sec, max_sec = 0.50, 2.0, 6.0

    target_sec = window_sec * speech_share
    target_sec = max(min_sec, min(max_sec, target_sec))
    target_words = target_sec * wpm_fast / 60.0
    min_words = max(4, int(target_words * 0.85))
    max_words = max(min_words + 2, int(target_words * 1.15 + 0.999))
    return {
        "profile": profile,
        "window_sec": round(window_sec, 2),
        "target_sec": round(target_sec, 2),
        "wpm_fast": float(wpm_fast),
        "min_words": int(min_words),
        "max_words": int(max_words),
    }


def build_eval_chains(
    payloads,
    *,
    use_chain_as_is=True,
    mode="related_time_cap",
    max_chains=250,
    include_timing_budget=False,
    wpm_fast=184.0,
    max_events=8,
    max_duration_sec=12.0,
    max_gap_sec=3.5,
):
    out = []
    if use_chain_as_is:
        for p in payloads:
            e = copy.deepcopy(p)
            if include_timing_budget:
                e["timing_budget"] = timing_budget_for_chain(e, wpm_fast=wpm_fast)
            out.append(e)
            if len(out) >= max_chains:
                break
        return out

    for p in payloads:
        for ch in split_chain_payload(
            p,
            mode=mode,
            max_events=max_events,
            max_duration_sec=max_duration_sec,
            gap_sec=max_gap_sec,
        ):
            e = copy.deepcopy(ch)
            if include_timing_budget:
                e["timing_budget"] = timing_budget_for_chain(e, wpm_fast=wpm_fast)
            out.append(e)
            if len(out) >= max_chains:
                return out
    return out


def _chain_time_bounds_sec(chain):
    events = chain.get("events") or []
    if not events:
        return None, None
    t0 = _to_seconds(((events[0].get("event_json") or {}).get("timestamp")))
    t1 = _to_seconds(((events[-1].get("event_json") or {}).get("timestamp")))
    return t0, t1


def _chain_priority(chain):
    cf = chain.get("chain_features") or {}
    high = int(cf.get("high_events_n", 0) or 0)
    entered_box = int(cf.get("entered_opponent_box_n", 0) or 0)
    setp = int(cf.get("set_play_events_n", 0) or 0)
    score = 0
    if high > 0:
        score += 2
    if entered_box > 0:
        score += 1
    if setp > 0:
        score += 1
    return min(score, 4)


def build_schedule(
    eval_chains,
    *,
    max_silence_sec=SCHED_MAX_SILENCE_SEC,
    min_gap_sec=SCHED_MIN_GAP_SEC,
    min_importance=SCHED_MIN_IMPORTANCE,
):
    rows = []
    last_comment_end = None
    for idx, ch in enumerate(eval_chains, start=1):
        t_start, t_end = _chain_time_bounds_sec(ch)
        tb = ch.get("timing_budget") or {}
        target_sec = float(tb.get("target_sec", 0.0) or 0.0)
        if t_start is None:
            t_start = 0.0
        if t_end is None:
            t_end = t_start
        priority = _chain_priority(ch)
        silence_sec = 999.0 if last_comment_end is None else max(0.0, t_start - last_comment_end)
        force_by_silence = int(silence_sec >= max_silence_sec)
        event_worthy = int(priority >= min_importance)
        gap_ok = int((last_comment_end is None) or (t_start - last_comment_end >= min_gap_sec))
        should_comment = int(gap_ok and (event_worthy or force_by_silence))
        comment_kind = "none"
        if should_comment:
            comment_kind = "event" if event_worthy else "heartbeat"
            est_dur = max(target_sec, 2.0)
            last_comment_end = t_start + est_dur

        rows.append(
            {
                "chain_idx": idx,
                "t_start_sec": round(t_start, 3),
                "t_end_sec": round(t_end, 3),
                "priority": priority,
                "silence_sec": round(silence_sec, 3),
                "force_by_silence": force_by_silence,
                "event_worthy": event_worthy,
                "gap_ok": gap_ok,
                "should_comment": should_comment,
                "comment_kind": comment_kind,
                "target_sec": target_sec,
            }
        )
    return pd.DataFrame(rows)

