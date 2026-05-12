```python
from mplsoccer import Sbopen
from statsbomb_toolkit import (
    DEFAULT_PROMPT_TEMPLATE,
    GigaChatClient,
    build_360_index,
    build_episode_catalog,
    build_episode_payloads,
    build_events_df,
    clean_event_recursive,
    draw_event_keep_style,
    run_batch_commentary,
    save_event_figure,
)

parser = Sbopen(dataframe=True)
events, related, freeze, tactics = parser.event(3930158)
frames, visible = parser.frame(3930158)
lineup = parser.lineup(3930158)[["player_id", "jersey_number"]]

df_events = build_events_df(events, lineup, frames=frames)
frames_by_id, visible_by_id = build_360_index(frames, visible)

event_id = df_events.iloc[4]["id"]
fig, ax = draw_event_keep_style(df_events, event_id, frames_by_id, visible_by_id)
image_path = save_event_figure(fig, event_id)

event_raw = events.iloc[4].to_dict()
event_clean = clean_event_recursive(event_raw)

# export GIGACHAT_BASIC_AUTH="<auth key from GigaChat cabinet>"
# If your key requires Basic scheme, use: GigaChatClient(auth_scheme="Basic")
client = GigaChatClient(
    ca_bundle="russian_trusted_root_ca.pem",
    auth_scheme="Bearer",
    # x_client_id="optional-stable-client-id",  # keep same value for upload + generation
)
file_id = client.upload_file(image_path)
result = client.generate_commentary(event_clean, sb360_json=None, image_attachment_id=file_id)
print(result.action, result.commentary)
```

```python
# 1) Просмотр кандидатов (для ручного отбора эпизодов)
catalog = build_episode_catalog(
    df_events,
    target_types=("Pass", "Carry", "Dribble"),
    require_360=True,
    context_window=3,
)
display(catalog.head(20))

# 2) Выбрали несколько event_id вручную
selected_ids = catalog["event_id"].head(5).tolist()

# 3) Собрали payload с контекстом вокруг эпизода
payloads = build_episode_payloads(
    df_events=df_events,
    events_raw=events_std,
    three_sixty_raw=three_sixty_std,
    event_ids=selected_ids,
    context_window=3,
    same_possession=True,
    context_allowed_types={"Pass", "Carry", "Dribble"},
)

# 4) Меняешь промпт как строку и запускаешь batch
prompt_template = DEFAULT_PROMPT_TEMPLATE + "\\nСтиль: кратко, как радио-комментарий."

batch_df = run_batch_commentary(
    client=client,
    payloads=payloads,
    prompt_template=prompt_template,
    extra_instructions="Пиши 2-3 предложения.",
    temperature=0.2,
)
display(batch_df[["event_id", "action", "commentary"]])
```
