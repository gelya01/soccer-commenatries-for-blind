"""
No-GT quality metrics for commentary outputs.
Auto-exported from sber_chain_eval.ipynb
"""

import re
import unicodedata

PRONOUNS_RU = {
    'он','она','они','его','ее','её','их','ему','ей','им','ними','него','неё','нам','вам','мне','ты','я','мы'
}


ACTION_WORDS = {
    'пас','передач','удар','бьет','бьёт','ведет','ведёт','перехват','отбор','потер',
    'углов','штрафн','аут','офсайд','сейв','блок','вбрасыван','гол','заброс','навес'
}


def _count_words_ru(text):
    return len(re.findall(r'[A-Za-zА-Яа-яЁё]+', text or ''))


def _surname(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        return ''
    return name.strip().split()[-1]


def _norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ''
    s = s.strip().lower().replace('ё', 'е')
    s = ''.join(ch for ch in unicodedata.normalize('NFKD', s) if not unicodedata.combining(ch))
    s = re.sub(r'[^a-zа-я0-9]+', '', s)
    return s


def _contains_any(text_low: str, stems: set[str]) -> int:
    return int(any(st in text_low for st in stems))


def _collect_chain_facts(chain):
    names_raw = set()
    tss = []
    type_set = set()

    has = {
        'pass': 0,
        'carry': 0,
        'shot': 0,
        'interception': 0,
        'recovery': 0,
        'dispossessed': 0,
        'miscontrol': 0,
        'foul': 0,
        'offside': 0,
        'corner': 0,
        'throw_in': 0,
        'goal_kick': 0,
    }

    outcomes = {
        'out': 0,
        'pass_offside': 0,
        'incomplete': 0,
        'goal': 0,
    }

    for e in chain.get('events', []):
        ev = e.get('event_json') or {}
        tp = ((ev.get('type') or {}).get('name') or '').strip()
        if tp:
            type_set.add(tp)

        pnm = ((ev.get('player') or {}).get('name'))
        if isinstance(pnm, str) and pnm.strip():
            names_raw.add(_surname(pnm))

        rec = ((ev.get('pass') or {}).get('recipient') or {}).get('name')
        if isinstance(rec, str) and rec.strip():
            names_raw.add(_surname(rec))

        ts = ev.get('timestamp')
        if ts:
            tss.append(ts)

        if tp == 'Pass':
            has['pass'] = 1
        elif tp == 'Carry':
            has['carry'] = 1
        elif tp == 'Shot':
            has['shot'] = 1
        elif tp == 'Interception':
            has['interception'] = 1
        elif tp == 'Ball Recovery':
            has['recovery'] = 1
        elif tp == 'Dispossessed':
            has['dispossessed'] = 1
        elif tp == 'Miscontrol':
            has['miscontrol'] = 1
        elif tp in {'Foul Committed', 'Foul Won'}:
            has['foul'] = 1
        elif tp == 'Offside':
            has['offside'] = 1

        pp = ((ev.get('play_pattern') or {}).get('name') or '').lower()
        if 'throw in' in pp:
            has['throw_in'] = 1
        if 'corner' in pp:
            has['corner'] = 1
        if 'goal kick' in pp:
            has['goal_kick'] = 1

        ptype = ((ev.get('pass') or {}).get('type') or {}).get('name')
        if isinstance(ptype, str):
            pl = ptype.lower()
            if 'corner' in pl:
                has['corner'] = 1
            if 'goal kick' in pl:
                has['goal_kick'] = 1
            if 'throw' in pl:
                has['throw_in'] = 1

        pout = ((ev.get('pass') or {}).get('outcome') or {}).get('name')
        if isinstance(pout, str):
            ol = pout.lower()
            if 'out' in ol:
                outcomes['out'] = 1
            if 'offside' in ol:
                outcomes['pass_offside'] = 1
            if 'incomplete' in ol:
                outcomes['incomplete'] = 1

        sout = ((ev.get('shot') or {}).get('outcome') or {}).get('name')
        if isinstance(sout, str) and 'goal' in sout.lower():
            outcomes['goal'] = 1

    known_names_norm = {_norm_name(n) for n in names_raw if n}
    known_names_norm = {x for x in known_names_norm if x}

    return {
        'names_norm': known_names_norm,
        't_start': tss[0] if tss else '',
        't_end': tss[-1] if tss else '',
        'has': has,
        'outcomes': outcomes,
        'type_set': type_set,
    }


def _norm_type_list(v):
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        if not isinstance(x, str):
            continue
        y = x.strip()
        if y:
            out.append(y)
    return out


def _norm_player_list(v):
    if not isinstance(v, list):
        return []
    out = []
    for x in v:
        if not isinstance(x, str):
            continue
        sx = x.strip()
        if not sx:
            continue
        out.append(_norm_name(_surname(sx)))
    return [x for x in out if x]


def evaluate_one_output(chain, parsed_obj, wpm_fast_default=184.0):
    facts = _collect_chain_facts(chain)

    json_valid = int(isinstance(parsed_obj, dict))
    required_fields = int(json_valid and all(k in parsed_obj for k in ['t_start', 't_end', 'commentary', 'event_types_commented', 'players_commented_en']))

    commentary = str(parsed_obj.get('commentary', '') or '') if json_valid else ''
    clow = commentary.lower()

    action_presence = int(any(w in clow for w in ACTION_WORDS))

    sent_count = len([s for s in re.split(r'[.!?]+', commentary) if s.strip()])
    one_three_sent = int(1 <= sent_count <= 3)

    wc = _count_words_ru(commentary)
    tb = chain.get('timing_budget') or {}
    min_w = int(tb.get('min_words', 0))
    max_w = int(tb.get('max_words', 10**9))
    if tb:
        length_ok = int(one_three_sent == 1 and min_w <= wc <= max_w)
    else:
        length_ok = int(one_three_sent == 1 and wc >= 4)

    toks = re.findall(r'[а-яa-zё]+', clow)
    pron_cnt = sum(t in PRONOUNS_RU for t in toks)
    pronouns_absent = int(pron_cnt == 0)
    low_pronoun_ratio = int((pron_cnt / max(len(toks), 1)) <= 0.15)
    pronoun_soft_ok = int((pron_cnt == 0) or (pron_cnt <= 1 and sent_count >= 2))

    time_bounds_match = 0
    if json_valid:
        time_bounds_match = int(
            str(parsed_obj.get('t_start', '')) == str(facts['t_start']) and
            str(parsed_obj.get('t_end', '')) == str(facts['t_end'])
        )

    # EN player list from model (avoids RU/EN ambiguity in free text)
    players_en = _norm_player_list(parsed_obj.get('players_commented_en', []) if json_valid else [])
    mentioned_players_ok = int(len(players_en) > 0 and any(p in facts['names_norm'] for p in players_en)) if facts['names_norm'] else 0

    unknown_players = [p for p in players_en if p not in facts['names_norm']]
    entity_hallucination_free = int(len(unknown_players) == 0)
    factual_name_consistency = int(len(unknown_players) <= 1)

    # fallback: still detect surname in RU text (diagnostic only)
    surname_mentioned = mentioned_players_ok

    # Event type alignment (new)
    types_pred = _norm_type_list(parsed_obj.get('event_types_commented', []) if json_valid else [])
    types_ok = int(len(types_pred) > 0 and all(t in facts['type_set'] for t in types_pred))

    naming_compliance = int(mentioned_players_ok == 1 and pronoun_soft_ok == 1)

    mention_offside = _contains_any(clow, {'офсайд'})
    mention_out = _contains_any(clow, {'аут', 'за боков', 'за лини'})
    mention_corner = _contains_any(clow, {'углов'})
    mention_goal_kick = _contains_any(clow, {'удар от ворот'})
    mention_goal = _contains_any(clow, {'гол'})

    has_offside_fact = int(facts['has']['offside'] or facts['outcomes']['pass_offside'])
    has_out_fact = int(facts['outcomes']['out'])
    has_corner_fact = int(facts['has']['corner'])
    has_goal_kick_fact = int(facts['has']['goal_kick'])
    has_goal_fact = int(facts['outcomes']['goal'])

    contradiction_count = 0
    if mention_offside and not has_offside_fact:
        contradiction_count += 1
    if mention_out and not has_out_fact:
        contradiction_count += 1
    if mention_corner and not has_corner_fact:
        contradiction_count += 1
    if mention_goal_kick and not has_goal_kick_fact:
        contradiction_count += 1
    if mention_goal and not has_goal_fact:
        contradiction_count += 1

    outcome_contradiction_free = int(contradiction_count == 0)

    has_high = int(any(v == 1 for k, v in facts['has'].items() if k in {
        'shot', 'interception', 'recovery', 'dispossessed', 'miscontrol', 'foul', 'offside'
    }))
    saliency_hit = int((not has_high) or action_presence == 1)

    est_tts_sec = (wc / float(tb.get('wpm_fast', wpm_fast_default))) * 60.0 if tb else 0.0
    target_sec = float(tb.get('target_sec', 0.0)) if tb else 0.0
    timing_fit = int((target_sec <= 0.0) or (0.75 * target_sec <= est_tts_sec <= 1.25 * target_sec))

    faithfulness_core = round((
        0.25 * entity_hallucination_free +
        0.25 * outcome_contradiction_free +
        0.20 * naming_compliance +
        0.10 * factual_name_consistency +
        0.10 * time_bounds_match +
        0.10 * types_ok
    ), 4)

    return {
        'json_valid': json_valid,
        'required_fields': required_fields,
        'surname_mentioned': surname_mentioned,
        'pronouns_absent': pronouns_absent,
        'pronoun_soft_ok': pronoun_soft_ok,
        'naming_compliance': naming_compliance,
        'action_presence': action_presence,
        'length_ok': length_ok,
        'one_three_sent': one_three_sent,
        'low_pronoun_ratio': low_pronoun_ratio,
        'time_bounds_match': time_bounds_match,
        'factual_name_consistency': factual_name_consistency,
        'entity_hallucination_free': entity_hallucination_free,
        'outcome_contradiction_free': outcome_contradiction_free,
        'event_types_alignment': types_ok,
        'faithfulness_core': faithfulness_core,
        'saliency_hit': saliency_hit,
        'timing_fit': timing_fit,
        'word_count': wc,
        'commentary_len': len(commentary),
        'est_tts_sec': round(est_tts_sec, 2),
        'target_sec': target_sec,
        'hallucinated_names_n': len(unknown_players),
        'contradiction_count': contradiction_count,
        'players_listed_n': len(players_en),
        'event_types_listed_n': len(types_pred),
    }
