"""
High-level orchestration over preprocessing + chaining modules.
Use this module in notebooks to avoid wiring every step manually.
"""

from __future__ import annotations

import pandas as pd

from . import chaining, preprocessing


def build_maps(events_std, sb360_std):
    events_by_id = {e.get("id"): e for e in events_std if e.get("id")}
    events_clean_by_id = {e.get("id"): preprocessing.clean_event_for_llm(e) for e in events_std if e.get("id")}
    sb360_by_id = {}
    for s in sb360_std:
        eid = s.get("event_uuid") or s.get("id")
        if eid:
            sb360_by_id[eid] = s
    return events_by_id, events_clean_by_id, sb360_by_id


def build_chain_payloads(chains, *, events_by_id, events_clean_by_id, sb360_by_id, bad_ids, ref_by_period):
    payloads = []
    for chain_ids in chains:
        payloads.append(
            preprocessing.build_chain_payload_v4(
                chain_ids,
                events_by_id=events_by_id,
                events_clean_by_id=events_clean_by_id,
                sb360_by_id=sb360_by_id,
                bad_ids=set(bad_ids or []),
                ref_by_period=ref_by_period,
            )
        )
    return payloads


def prepare_match_payloads(events_raw, sb360_raw, *, ref_by_period, bad_ids=None):
    """
    End-to-end prep:
    1) standardize by half/team
    2) related-events graph
    3) non-overlap chains
    4) chain payloads for LLM
    """
    events_std, sb360_std, flip_map = preprocessing.standardize_events_and_360_by_half(
        events_raw, sb360_raw, ref_by_period
    )
    events_by_id, neighbors = chaining.build_related_graph(events_std)
    chains = chaining.build_nonoverlap_chains(events_std, neighbors)
    events_by_id, events_clean_by_id, sb360_by_id = build_maps(events_std, sb360_std)
    payloads = build_chain_payloads(
        chains,
        events_by_id=events_by_id,
        events_clean_by_id=events_clean_by_id,
        sb360_by_id=sb360_by_id,
        bad_ids=set(bad_ids or []),
        ref_by_period=ref_by_period,
    )
    return {
        "events_std": events_std,
        "sb360_std": sb360_std,
        "flip_map": flip_map,
        "events_by_id": events_by_id,
        "neighbors": neighbors,
        "chains": chains,
        "payloads": payloads,
    }


def build_bad_ids(
    events_raw,
    sb360_raw,
    min_freeze_players=3,
    include_tiny_freeze=False,
    include_non_play=True,
    include_low_signal=True,
    include_missing_360=False,
    include_actor_mismatch=True,
    max_actor_event_dist=12.0,
):
    """
    Rule-based bad events:
    - non-play and low-signal types
    - no 360
    - too small freeze_frame
    """
    sb_by_id = {}
    for s in sb360_raw:
        eid = s.get("event_uuid") or s.get("id")
        if eid:
            sb_by_id[eid] = s

    bad = set()
    for ev in events_raw:
        eid = ev.get("id")
        et = preprocessing.get_type_name(ev)

        if include_non_play and et in preprocessing.NON_PLAY_EVENT_TYPES:
            if eid:
                bad.add(eid)
            continue
        if include_low_signal and et in preprocessing.LOW_SIGNAL_EVENT_TYPES:
            if eid:
                bad.add(eid)
            continue

        sb = sb_by_id.get(eid)
        if not isinstance(sb, dict):
            if include_missing_360 and eid:
                bad.add(eid)
            continue

        ff = sb.get("freeze_frame") or []
        if include_tiny_freeze and len(ff) < min_freeze_players and eid:
            bad.add(eid)
            continue

        if include_actor_mismatch:
            ev_loc = preprocessing.event_location(ev)
            actor_loc = preprocessing.actor_location(sb)
            d = preprocessing.dist(ev_loc, actor_loc)
            if d is not None and d > float(max_actor_event_dist):
                if eid:
                    bad.add(eid)

    return bad


def bad_ids_breakdown(
    events_raw,
    sb360_raw,
    min_freeze_players=3,
    include_tiny_freeze=False,
    include_non_play=True,
    include_low_signal=True,
    include_missing_360=False,
    include_actor_mismatch=True,
    max_actor_event_dist=12.0,
):
    sb_by_id = {}
    for s in sb360_raw:
        eid = s.get("event_uuid") or s.get("id")
        if eid:
            sb_by_id[eid] = s

    reasons = {
        "non_play": 0,
        "low_signal": 0,
        "missing_360": 0,
        "tiny_freeze_frame": 0,
        "actor_mismatch_dist": 0,
    }
    bad = set()

    for ev in events_raw:
        eid = ev.get("id")
        et = preprocessing.get_type_name(ev)

        if include_non_play and et in preprocessing.NON_PLAY_EVENT_TYPES:
            reasons["non_play"] += 1
            if eid:
                bad.add(eid)
            continue
        if include_low_signal and et in preprocessing.LOW_SIGNAL_EVENT_TYPES:
            reasons["low_signal"] += 1
            if eid:
                bad.add(eid)
            continue

        sb = sb_by_id.get(eid)
        if not isinstance(sb, dict):
            reasons["missing_360"] += 1
            if include_missing_360 and eid:
                bad.add(eid)
            continue

        ff = sb.get("freeze_frame") or []
        if len(ff) < min_freeze_players:
            reasons["tiny_freeze_frame"] += 1
            if include_tiny_freeze and eid:
                bad.add(eid)
                continue

        if include_actor_mismatch:
            ev_loc = preprocessing.event_location(ev)
            actor_loc = preprocessing.actor_location(sb)
            d = preprocessing.dist(ev_loc, actor_loc)
            if d is not None and d > float(max_actor_event_dist):
                reasons["actor_mismatch_dist"] += 1
                if eid:
                    bad.add(eid)

    return {"bad_ids": bad, "reasons": reasons}


def run_match_pipeline(
    *,
    match_id: int | str,
    events_raw: list,
    sb360_raw: list,
    ref_by_period: dict[int, str],
    include_timing_budget=False,
    use_chain_as_is=True,
    max_chains=100000,
    min_freeze_players=3,
    include_tiny_freeze_bad=False,
    include_non_play_bad=True,
    include_low_signal_bad=True,
    include_missing_360_bad=False,
    include_actor_mismatch_bad=True,
    max_actor_event_dist=12.0,
):
    bad_ids = build_bad_ids(
        events_raw,
        sb360_raw,
        min_freeze_players=min_freeze_players,
        include_tiny_freeze=include_tiny_freeze_bad,
        include_non_play=include_non_play_bad,
        include_low_signal=include_low_signal_bad,
        include_missing_360=include_missing_360_bad,
        include_actor_mismatch=include_actor_mismatch_bad,
        max_actor_event_dist=max_actor_event_dist,
    )
    prepared = prepare_match_payloads(
        events_raw=events_raw,
        sb360_raw=sb360_raw,
        ref_by_period=ref_by_period,
        bad_ids=bad_ids,
    )
    eval_chains = chaining.build_eval_chains(
        prepared["payloads"],
        use_chain_as_is=use_chain_as_is,
        include_timing_budget=include_timing_budget,
        max_chains=max_chains,
    )
    return prepared, eval_chains, bad_ids


def summarize_prepared(prepared, bad_ids):
    events = [e for ch in prepared["payloads"] for e in ch.get("events", [])]
    orient_rows = []
    for e in events:
        ej = e.get("event_json") or {}
        ori = (e.get("derived") or {}).get("orientation") or {}
        if ori:
            orient_rows.append(
                {
                    "period": ej.get("period"),
                    "team": (ej.get("team") or {}).get("name"),
                    "own_goal_x": ori.get("own_goal_x"),
                    "opp_goal_x": ori.get("opp_goal_x"),
                    "attack_sign": ori.get("attack_sign"),
                }
            )

    orient_df = pd.DataFrame(orient_rows)
    if not orient_df.empty:
        orient_df = orient_df.drop_duplicates().sort_values(["period", "team"]).reset_index(drop=True)

    summary = {
        "chains_n": len(prepared.get("chains", [])),
        "payloads_n": len(prepared.get("payloads", [])),
        "events_in_payload_n": len(events),
        "bad_ids_n": len(bad_ids),
        "without_sb360_n": sum(1 for e in events if e.get("sb360_json") is None),
    }
    return summary, orient_df


def payloads_to_event_features(eval_chains, match_id):
    rows = []
    for ci, ch in enumerate(eval_chains, start=1):
        cf = ch.get("chain_features", {})
        for ei, e in enumerate(ch.get("events", []), start=1):
            ej = e.get("event_json") or {}
            d = e.get("derived") or {}
            rows.append(
                {
                    "match_id": match_id,
                    "chain_idx": ci,
                    "event_pos": ei,
                    "event_id": e.get("event_id"),
                    "timestamp": ej.get("timestamp"),
                    "period": ej.get("period"),
                    "team": (ej.get("team") or {}).get("name"),
                    "player": (ej.get("player") or {}).get("name"),
                    "type": (ej.get("type") or {}).get("name"),
                    "sb360_is_null": int(e.get("sb360_json") is None),
                    "skip_derive": (d.get("quality_flags") or {}).get("skip_derive"),
                    "start_rel": (d.get("zones") or {}).get("start_rel"),
                    "end_rel": (d.get("zones") or {}).get("end_rel"),
                    "start_lane": (d.get("zones") or {}).get("start_lane"),
                    "end_lane": (d.get("zones") or {}).get("end_lane"),
                    "zone_transition": (d.get("zones") or {}).get("zone_transition"),
                    "move_label": (d.get("movement") or {}).get("label"),
                    "move_compass": (d.get("movement") or {}).get("compass"),
                    "forward_delta": (d.get("movement") or {}).get("forward_delta"),
                    "lateral_delta": (d.get("movement") or {}).get("lateral_delta"),
                    "is_long_pass": (d.get("episode_signals") or {}).get("is_long_pass"),
                    "switched_flank": (d.get("episode_signals") or {}).get("switched_flank"),
                    "entered_opponent_half": (d.get("episode_signals") or {}).get("entered_opponent_half"),
                    "entered_opponent_box": (d.get("episode_signals") or {}).get("entered_opponent_box"),
                    "chain_events_n": cf.get("events_n"),
                    "chain_high_n": cf.get("high_events_n"),
                    "chain_setplay_n": cf.get("set_play_events_n"),
                    "chain_importance_rule": cf.get("importance_score_rule"),
                }
            )
    return pd.DataFrame(rows)
