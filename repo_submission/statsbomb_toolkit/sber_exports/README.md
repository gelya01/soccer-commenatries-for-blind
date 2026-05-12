# sber_exports

Набор самодостаточных модулей, вынесенных из `sber_chain_eval.ipynb`.

## Что внутри

- `preprocessing.py`
  - стандартизация координат по таймам/команде;
  - очистка событий;
  - зоны поля;
  - derived-признаки;
  - сборка payload по цепочке.
- `chaining.py`
  - related-events граф;
  - непересекающиеся цепочки;
  - chunking (опционально);
  - timing budget (опционально);
  - scheduler по тишине/приоритету.
- `prompting.py`
  - системный промпт;
  - формирование prompt payload;
  - валидация формы payload.
- `metrics.py`
  - no-GT метрики качества генерации.
- `viz.py`
  - визуализация событий и передач (в стиле notebook).
- `pipeline.py`
  - end-to-end orchestration поверх модулей выше.

## Быстрый старт

```python
from statsbomb_toolkit.sber_exports import pipeline, chaining, prompting, metrics, viz

REF_BY_PERIOD = {1: "Scotland", 2: "Germany"}
BAD_IDS = set()  # при наличии bad-events подставь сюда

prepared = pipeline.prepare_match_payloads(
    events_raw=events_raw,
    sb360_raw=sb360_raw,
    ref_by_period=REF_BY_PERIOD,
    bad_ids=BAD_IDS,
)

chains = prepared["chains"]
payloads = prepared["payloads"]

# Если хочешь использовать цепочки "как есть":
eval_chains = chaining.build_eval_chains(
    payloads,
    use_chain_as_is=True,
    include_timing_budget=False,
)

prompt = prompting.make_chain_prompt(eval_chains[0])
```

## Важно

- В `make_chain_prompt` передаётся только JSON payload, а системный промпт берётся из `prompting.CHAIN_PROMPT_SBER_V2`.
- В `build_eval_chains(..., include_timing_budget=False)` timing budget в payload не добавляется.
