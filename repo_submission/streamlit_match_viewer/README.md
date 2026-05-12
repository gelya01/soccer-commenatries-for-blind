# Streamlit Match Viewer

Визуализация матчей Euro 2024 из сохранённых `standardized` JSON.

## Запуск

Из корня проекта:

```bash
streamlit run streamlit_match_viewer/app.py
```

По умолчанию используется индекс:

`outputs/euro2024_all/processed_json/index_to_speak_top15.csv`

## Что умеет

- выбор матча;
- автопроигрывание кадров как видео (по событиям);
- фильтр по тайму;
- фильтр `only bad_ids`;
- опция отключать 360 для `bad_ids`;
- нижняя плашка `actor_source` скрыта.
