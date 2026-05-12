#!/usr/bin/env python3
"""
Export LLM-ready payloads for all grouping strategies.

Input:
  outputs/euro2024_all/processed_json/index_all.csv
  standardized JSON artifacts produced by euro2024_top15_clean_pipeline.ipynb

Output:
  one JSON file per strategy with only selected episodes:
    soft_label_3 >= MIN_LLM_SOFT_LABEL
  plus CSV/JSON summaries.

Run example:
  python scripts/export_llm_strategy_payloads.py \
    --processed-dir outputs/euro2024_all/processed_json \
    --out-dir outputs/llm_strategy_payloads \
    --sb360-mode summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statsbomb_toolkit.sber_exports import preprocessing

STOP_TYPES = {
    "Shot", "Offside", "Foul Committed", "Foul Won", "Interception", "Ball Recovery",
    "Dispossessed", "Miscontrol", "Clearance", "Goal Keeper", "Block", "Error",
    "Own Goal For", "Own Goal Against",
}

DEFAULT_STRATEGIES = [
    "event_only",
    "related_only",
    "related_cap",
    "related_stop_cap",
    "time_window",
    "possession_cap",
]

DEFAULT_SPEAK_MATCH_IDS = {
    "3930158",  # Germany - Scotland, opening match
    "3943043",  # Spain - England, final
    "3942752",  # Spain - France, semi-final
    "3942819",  # Netherlands - England, semi-final
    "3942226",  # Spain - Germany, quarter-final
    "3942349",  # Portugal - France, quarter-final
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_artifact_path(raw_path: str, processed_dir: Path) -> Path:
    """Allow index CSV with absolute paths from another machine."""
    p = Path(raw_path)
    if p.exists():
        return p
    alt = processed_dir / p.name
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Cannot resolve artifact path: {raw_path} (also tried {alt})")


def to_seconds(ts):
    if not isinstance(ts, str):
        return None
    m = re.match(r"^(\d+):(\d+):(\d+)(?:\.(\d+))?$", ts.strip())
    if not m:
        return None
    hh, mm, ss, ms = m.groups()
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + (float(f"0.{ms}") if ms else 0.0)


def event_type(ev):
    t = ev.get("type") or {}
    return t.get("name") if isinstance(t, dict) else str(t)


def event_time(ev):
    return to_seconds(ev.get("timestamp"))


def sorted_events(events_std):
    return sorted(events_std, key=lambda e: (e.get("period") or 0, e.get("index", 10**9), e.get("id") or ""))


def build_related_graph(events_std):
    events_by_id = {e.get("id"): e for e in events_std if e.get("id")}
    outgoing = defaultdict(set)
    incoming = defaultdict(set)
    for e in events_std:
        eid = e.get("id")
        if not eid:
            continue
        for rid in e.get("related_events") or []:
            if rid:
                outgoing[eid].add(rid)
                incoming[rid].add(eid)
    neighbors = {eid: set() for eid in events_by_id}
    for eid in events_by_id:
        neighbors[eid] |= outgoing.get(eid, set())
        neighbors[eid] |= incoming.get(eid, set())
    return events_by_id, neighbors


def split_chain_by_caps(chain, events_by_id, *, max_events=6, max_duration_sec=10.0, max_gap_sec=2.5, stop_types=False):
    chunks = []
    cur = []
    t0 = None
    prev_t = None

    def flush():
        nonlocal cur, t0, prev_t
        if cur:
            chunks.append(cur)
        cur, t0, prev_t = [], None, None

    for eid in chain:
        ev = events_by_id[eid]
        t = event_time(ev)
        et = event_type(ev)

        if not cur:
            cur = [eid]
            t0 = t
            prev_t = t
            continue

        need_cut = False
        if len(cur) >= max_events:
            need_cut = True
        if t is not None and t0 is not None and (t - t0) > max_duration_sec:
            need_cut = True
        if t is not None and prev_t is not None and (t - prev_t) > max_gap_sec:
            need_cut = True

        if need_cut:
            flush()
            cur = [eid]
            t0 = t
            prev_t = t
        else:
            cur.append(eid)
            prev_t = t

        if stop_types and et in STOP_TYPES and len(cur) >= 2:
            flush()

    flush()
    return chunks


def chains_event_only(events_std, **_):
    events_by_id = {e.get("id"): e for e in events_std if e.get("id")}
    return [[e.get("id")] for e in sorted_events(events_std) if e.get("id")], events_by_id


def chains_related_components(events_std, *, cap=False, stop_types=False, max_events=6, max_duration_sec=10.0, max_gap_sec=2.5, **_):
    events_by_id, neighbors = build_related_graph(events_std)
    allowed = set(events_by_id)
    used = set()
    chains = []

    def expand(root_id):
        group = set()
        stack = [root_id]
        while stack:
            cur = stack.pop()
            if not cur or cur in group or cur in used:
                continue
            group.add(cur)
            for nb in neighbors.get(cur, set()):
                if nb in allowed and nb not in group and nb not in used:
                    stack.append(nb)
        return group

    for e in sorted_events(events_std):
        eid = e.get("id")
        if not eid or eid in used:
            continue
        group = expand(eid)
        chain = sorted(group, key=lambda x: events_by_id.get(x, {}).get("index", 10**9))
        used.update(chain)
        if cap:
            chains.extend(split_chain_by_caps(chain, events_by_id, max_events=max_events, max_duration_sec=max_duration_sec, max_gap_sec=max_gap_sec, stop_types=stop_types))
        else:
            chains.append(chain)
    return chains, events_by_id


def chains_time_window(events_std, *, time_window_sec=8.0, max_events=6, **_):
    events = [e for e in sorted_events(events_std) if e.get("id")]
    events_by_id = {e.get("id"): e for e in events}
    chains = []
    cur = []
    t0 = None
    cur_period = None
    for e in events:
        eid = e.get("id")
        t = event_time(e)
        period = e.get("period")
        need_cut = False
        if cur:
            if period != cur_period:
                need_cut = True
            if t is not None and t0 is not None and (t - t0) > time_window_sec:
                need_cut = True
            if len(cur) >= max_events:
                need_cut = True
        if need_cut:
            chains.append(cur)
            cur = []
            t0 = None
        if not cur:
            t0 = t
            cur_period = period
        cur.append(eid)
    if cur:
        chains.append(cur)
    return chains, events_by_id


def chains_possession_cap(events_std, *, max_events=6, max_duration_sec=10.0, max_gap_sec=2.5, **_):
    events = [e for e in sorted_events(events_std) if e.get("id")]
    events_by_id = {e.get("id"): e for e in events}
    chains = []
    cur = []
    cur_key = None
    for e in events:
        key = (e.get("period"), e.get("possession"))
        eid = e.get("id")
        if cur and key != cur_key:
            chains.extend(split_chain_by_caps(cur, events_by_id, max_events=max_events, max_duration_sec=max_duration_sec, max_gap_sec=max_gap_sec, stop_types=False))
            cur = []
        cur_key = key
        cur.append(eid)
    if cur:
        chains.extend(split_chain_by_caps(cur, events_by_id, max_events=max_events, max_duration_sec=max_duration_sec, max_gap_sec=max_gap_sec, stop_types=False))
    return chains, events_by_id


def get_strategy_fn(strategy):
    if strategy == "event_only":
        return chains_event_only
    if strategy == "related_only":
        return lambda evs, **kw: chains_related_components(evs, cap=False, stop_types=False, **kw)
    if strategy == "related_cap":
        return lambda evs, **kw: chains_related_components(evs, cap=True, stop_types=False, **kw)
    if strategy == "related_stop_cap":
        return lambda evs, **kw: chains_related_components(evs, cap=True, stop_types=True, **kw)
    if strategy == "time_window":
        return chains_time_window
    if strategy == "possession_cap":
        return chains_possession_cap
    raise ValueError(f"Unknown strategy: {strategy}")


def build_maps(events_std, events_std_clean, sb360_std):
    events_by_id = {e.get("id"): e for e in events_std if e.get("id")}
    clean_by_id = {e.get("id"): e for e in events_std_clean if e.get("id")}
    sb360_by_id = {}
    for s in sb360_std:
        eid = s.get("event_uuid") or s.get("id")
        if eid:
            sb360_by_id[eid] = s
    return events_by_id, clean_by_id, sb360_by_id


def summarize_360(sb):
    if not sb:
        return None
    ff = sb.get("freeze_frame") or []
    va = sb.get("visible_area") or []
    return {
        "freeze_frame_n": len(ff),
        "teammates_n": sum(1 for r in ff if r.get("teammate") is True),
        "opponents_n": sum(1 for r in ff if r.get("teammate") is False),
        "keepers_n": sum(1 for r in ff if r.get("keeper") is True),
        "visible_area_points_n": int(len(va) / 2) if isinstance(va, list) else 0,
    }


def compact_payload_for_llm(payload, sb360_mode="summary"):
    events = []
    for item in payload.get("events", []):
        out = {
            "event_json": item.get("event_json"),
            "derived": item.get("derived"),
        }
        if sb360_mode == "summary":
            out["sb360_summary"] = summarize_360(item.get("sb360_json"))
        elif sb360_mode == "full":
            out["sb360_json"] = item.get("sb360_json")
        else:
            out["sb360_json"] = None
        events.append(out)
    return {
        "match_id": payload.get("match_id"),
        "strategy": payload.get("strategy"),
        "chain_event_ids": payload.get("chain_event_ids"),
        "chain_features": payload.get("chain_features"),
        "selection": payload.get("selection"),
        "events": events,
    }


def chain_duration_sec(payload):
    events = payload.get("events") or []
    if not events:
        return 0.0
    t0 = to_seconds(((events[0].get("event_json") or {}).get("timestamp")))
    t1 = to_seconds(((events[-1].get("event_json") or {}).get("timestamp")))
    return max(0.0, t1 - t0) if t0 is not None and t1 is not None else 0.0


def chain_features_from_payload(payload, chain_id=0):
    events = payload.get("events") or []
    types = []
    first_type = None
    last_type = None
    high_event_n = 0
    set_play_n = 0
    long_pass_n = 0
    entered_opp_half_n = 0
    entered_opp_box_n = 0
    forward_moves_n = 0
    switched_flank_n = 0
    bad_pass_outcome_n = 0
    low_signal_n = 0
    class_shot_n = 0

    high_types = {
        "Shot", "Foul Won", "Foul Committed", "Offside", "Interception", "Ball Recovery",
        "Dispossessed", "Miscontrol", "Error", "Goal Keeper", "Clearance", "Block",
    }
    set_play_patterns = {"From Throw In", "From Free Kick", "From Corner", "From Goal Kick", "From Kick Off"}

    for item in events:
        ej = item.get("event_json") or {}
        derived = item.get("derived") or {}
        et = ((ej.get("type") or {}).get("name"))
        if et:
            types.append(et)
        es = derived.get("event_semantics") or {}
        sig = derived.get("episode_signals") or {}
        move = derived.get("movement") or {}
        pp = ((ej.get("play_pattern") or {}).get("name"))
        pass_out = (((ej.get("pass") or {}).get("outcome") or {}).get("name")) if isinstance(ej.get("pass"), dict) else None

        high_event_n += int(et in high_types or es.get("is_key_action") == 1)
        set_play_n += int(pp in set_play_patterns)
        long_pass_n += int(sig.get("is_long_pass") == 1)
        entered_opp_half_n += int(sig.get("entered_opponent_half") == 1)
        entered_opp_box_n += int(sig.get("entered_opponent_box") == 1)
        forward_moves_n += int((move.get("forward_delta") or 0) > 0)
        switched_flank_n += int(sig.get("switched_flank") == 1)
        bad_pass_outcome_n += int(pass_out in {"Incomplete", "Out", "Pass Offside", "Unknown"})
        low_signal_n += int(es.get("is_low_signal") == 1 or et in {"Ball Receipt*", "Pressure"})
        class_shot_n += int(et == "Shot")

    if types:
        first_type = types[0]
        last_type = types[-1]

    return {
        "chain_id": chain_id,
        "events_n": len(events),
        "duration_sec": chain_duration_sec(payload),
        "first_type": first_type,
        "last_type": last_type,
        "high_event_n": high_event_n,
        "set_play_n": set_play_n,
        "long_pass_n": long_pass_n,
        "entered_opp_half_n": entered_opp_half_n,
        "entered_opp_box_n": entered_opp_box_n,
        "forward_moves_n": forward_moves_n,
        "switched_flank_n": switched_flank_n,
        "bad_pass_outcome_n": bad_pass_outcome_n,
        "low_signal_n": low_signal_n,
        "class_shot_n": class_shot_n,
    }


def soft_label_rule_5(r):
    score = 0.0
    score += 3.0 * r.get("class_shot_n", 0)
    score += 2.5 * r.get("entered_opp_box_n", 0)
    score += 2.0 * r.get("high_event_n", 0)
    score += 1.5 * r.get("long_pass_n", 0)
    score += 1.2 * r.get("forward_moves_n", 0)
    score += 1.2 * r.get("switched_flank_n", 0)
    score += 1.0 * r.get("bad_pass_outcome_n", 0)
    score += 0.8 * r.get("set_play_n", 0)
    score -= 0.35 * r.get("low_signal_n", 0)
    if r.get("events_n", 0) == 1 and r.get("low_signal_n", 0) == 1:
        score -= 2
    if r.get("duration_sec", 0) > 15 and r.get("class_shot_n", 0) == 0:
        score -= 0.5
    if score <= 0:
        return 0
    if score <= 1.5:
        return 1
    if score <= 3.5:
        return 2
    if score <= 6.0:
        return 3
    return 4


def map_5_to_3(lbl5):
    if lbl5 <= 1:
        return 0
    if lbl5 <= 3:
        return 1
    return 2


def selection_metadata_from_features(payload, payload_idx):
    feats = chain_features_from_payload(payload, payload_idx)
    soft5 = soft_label_rule_5(feats)
    soft3 = map_5_to_3(soft5)
    mode_map = {0: "skip", 1: "brief", 2: "must"}
    return {
        "soft_label_5": int(soft5),
        "soft_label_3": int(soft3),
        "commenting_mode": mode_map[int(soft3)],
    }


def load_manifest(index_path: Path, processed_dir: Path, match_ids: set[str] | None):
    rows = []
    with index_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = str(row["match_id"])
            if match_ids is not None and mid not in match_ids:
                continue
            fixed = {"match_id": mid}
            for key in ["events_std", "events_std_clean", "sb360_std", "bad_ids", "meta"]:
                fixed[key] = str(resolve_artifact_path(row[key], processed_dir))
            rows.append(fixed)
    return rows


def export_payloads(args):
    processed_dir = Path(args.processed_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.index).resolve() if args.index else processed_dir / "index_all.csv"
    match_ids = None if args.match_ids == "all" else set(args.match_ids.split(","))
    strategies = args.strategies.split(",") if args.strategies else DEFAULT_STRATEGIES

    manifest = load_manifest(index_path, processed_dir, match_ids)
    print("matches:", len(manifest))
    print("strategies:", strategies)
    print("sb360_mode:", args.sb360_mode)

    export_rows = []
    for strategy in strategies:
        strategy_payloads_all = []
        selected_payloads = []
        local_global_idx = 0

        for row in manifest:
            mid = row["match_id"]
            events_std = load_json(Path(row["events_std"]))
            events_std_clean = load_json(Path(row["events_std_clean"]))
            sb360_std = load_json(Path(row["sb360_std"]))
            bad_ids = set(load_json(Path(row["bad_ids"])))
            meta = load_json(Path(row["meta"]))
            ref_by_period = {int(k): v for k, v in (meta.get("ref_by_period") or {}).items()}

            chains, events_by_id_from_strategy = get_strategy_fn(strategy)(
                events_std,
                max_events=args.max_events,
                max_duration_sec=args.max_duration_sec,
                max_gap_sec=args.max_gap_sec,
                time_window_sec=args.time_window_sec,
            )
            events_by_id, clean_by_id, sb360_by_id = build_maps(events_std, events_std_clean, sb360_std)
            # events_by_id_from_strategy should match events_by_id; keep events_by_id from clean maps for safety.
            for local_idx, chain_ids in enumerate(chains, start=1):
                local_global_idx += 1
                payload = preprocessing.build_chain_payload_v4(
                    chain_ids,
                    events_by_id=events_by_id,
                    events_clean_by_id=clean_by_id,
                    sb360_by_id=sb360_by_id,
                    bad_ids=bad_ids,
                    ref_by_period=ref_by_period,
                )
                payload["match_id"] = mid
                payload["strategy"] = strategy
                payload["local_chain_idx"] = local_idx
                payload["stage"] = meta.get("stage") or "EURO 2024"
                strategy_payloads_all.append(payload)

                selection = selection_metadata_from_features(payload, local_global_idx)
                if selection["soft_label_3"] < args.min_soft_label:
                    continue
                payload["selection"] = selection
                selected_payloads.append(compact_payload_for_llm(payload, args.sb360_mode))

        out_json = out_dir / f"llm_payloads_speak_set_{strategy}_{args.sb360_mode}.json"
        out_json.write_text(json.dumps(selected_payloads, ensure_ascii=False, indent=2), encoding="utf-8")
        brief_n = sum(1 for p in selected_payloads if p.get("selection", {}).get("soft_label_3") == 1)
        must_n = sum(1 for p in selected_payloads if p.get("selection", {}).get("soft_label_3") == 2)
        export_rows.append({
            "strategy": strategy,
            "all_strategy_payloads": len(strategy_payloads_all),
            "llm_payloads_total": len(selected_payloads),
            "brief_n": brief_n,
            "must_n": must_n,
            "output_json": str(out_json),
        })
        print(strategy, "all:", len(strategy_payloads_all), "selected:", len(selected_payloads), "->", out_json)

    index_csv = out_dir / f"llm_payloads_speak_set_index_{args.sb360_mode}.csv"
    with index_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(export_rows[0].keys()))
        writer.writeheader()
        writer.writerows(export_rows)
    (out_dir / f"llm_payloads_speak_set_index_{args.sb360_mode}.json").write_text(
        json.dumps(export_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("saved index:", index_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="outputs/euro2024_all/processed_json")
    parser.add_argument("--index", default=None, help="CSV manifest. Default: <processed-dir>/index_all.csv")
    parser.add_argument("--out-dir", default="outputs/llm_strategy_payloads")
    parser.add_argument("--match-ids", default=",".join(sorted(DEFAULT_SPEAK_MATCH_IDS)), help="Comma-separated match ids or 'all'")
    parser.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES))
    parser.add_argument("--sb360-mode", choices=["none", "summary", "full"], default="summary")
    parser.add_argument("--min-soft-label", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=6)
    parser.add_argument("--max-duration-sec", type=float, default=10.0)
    parser.add_argument("--max-gap-sec", type=float, default=2.5)
    parser.add_argument("--time-window-sec", type=float, default=8.0)
    args = parser.parse_args()
    export_payloads(args)


if __name__ == "__main__":
    main()
