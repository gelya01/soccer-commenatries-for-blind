from __future__ import annotations

import json
import mimetypes
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
DEFAULT_API_BASE = "https://gigachat.devices.sberbank.ru/api/v1"
DEFAULT_SCOPE = "GIGACHAT_API_PERS"
DEFAULT_SYSTEM_PROMPT = (
    'Отвечай строго JSON-объектом вида {"timestamp": "...", "action":"...","commentary":"..."}. '
    "Никакого текста вне JSON."
)


@dataclass
class CommentaryResult:
    timestamp: str
    action: str
    commentary: str
    raw_content: str
    raw_response: dict[str, Any]


class GigaChatAPIError(RuntimeError):
    pass


def make_prompt(event_json: dict[str, Any], sb360_json: dict[str, Any] | None) -> str:
    return f"""Ты — тифлокомментатор футбольного матча (audio description для незрячих).
Твоя задача: по одному событию StatsBomb (event_json) и, если есть, по StatsBomb360 (sb360_json + freeze-frame картинка) дать КОРОТКОЕ, нейтральное и проверяемое описание того, что происходит.

ОБЩИЙ СТИЛЬ:
- 1 короткое предложение (максимум 2). Без художественности.
- Без оценок (“опасно”, “красиво”, “остро”, “почти гол”), если это не следует напрямую из outcome.
- Без “мы/они”. Пиши нейтрально: фамилия игрока + действие.
- Не придумывай, кто владеет мячом дальше, если это не видно из outcome/типа события.
- Команду называй только если без неё непонятно (обычно достаточно фамилии).

РАКУРС / КАК СЛУШАТЕЛЬ “ВИДИТ” ПОЛЕ:
- Слушатель “смотрит” с нижней трибуны (со стороны ближней бровки), ракурс постоянный.
- y=80 — ближняя бровка (под нашей трибуной). y=0 — дальняя бровка.
- Команды меняются воротами во 2 тайме. Поэтому “свои/чужие” определяются по period:
  period=1: ворота Шотландии слева (x≈0), ворота Германии справа (x≈120).
  period=2: ворота Германии слева (x≈0), ворота Шотландии справа (x≈120).
- “Своя половина” = ближе к своим воротам команды события в этом тайме.
- “Чужая половина” = ближе к воротам соперника в этом тайме.
- Если используешь “левый/правый фланг”, это ЛЕВО/ПРАВО на карте при фиксированном ракурсе (не “по атаке”).
  Но предпочтительнее: “у ближней бровки / у дальней бровки / в центре”.

КООРДИНАТЫ (StatsBomb 120×80):
- x ∈ [0..120] (слева направо по карте), y ∈ [0..80] (сверху вниз по карте).
- Говори про место ТОЛЬКО если координаты есть.
- Разрешённые футбольные формулировки места:
  1) “на своей половине / на чужой половине” (относительно команды события и тайма)
  2) “у центральной линии” (если x близко к 60)
  3) “у ближней бровки” (если y ближе к 80) / “у дальней бровки” (если y ближе к 0)
  4) “в центре” (если y примерно посередине)
  5) “у линии штрафной” (если близко к границе штрафной)
  6) “в штрафной” (если внутри штрафной)
  7) “у линии ворот” (если очень близко к x≈0 или x≈120)
  8) “у углового флажка” (если одновременно близко к воротам и к одной из бровок)
- Если координаты не дают уверенности — не называй зону.

ПРАВИЛА ДЕЙСТВИЯ (главное — что говорить):
1) PASS (передача):
- Всегда: “<Фамилия> — пас <кому>” если recipient есть.
- Если recipient нет: “<Фамилия> — пас”.
- Если передача не удалась: обязательно добавь исход:
  * если outcome/причина указывает на неточность: “неточно”
  * если out/за линию: “в аут” или “за лицевую” (если это следует из полей)
  * если Pass Offside: “офсайд”
  * если перехват явно следует (например есть связанное событие Interception): “перехват”
- Не говори “перевод”, “обостряющий”, “вразрез”, если это не кодируется явно.

2) BALL RECEIPT (приём):
- “<Фамилия> принимает” или “не принимает” (если incomplete).

3) CARRY (ведение):
- “<Фамилия> ведёт мяч” + направление только если есть start/end:
  “вперёд/назад/к ближней бровке/к дальней бровке/в центр” — по изменению координат.

4) INTERCEPTION / BALL RECOVERY / DISPOSSESSED / MISCONTROL / PRESSURE:
- Говори ровно тип события: “перехват”, “отбор”, “потеря”, “накрыли”, “прессинг” — только если это и есть type_name/outcome.

5) SHOT (удар):
- “<Фамилия> бьёт” + outcome, если он есть: “мимо”, “в створ”, “заблокирован”, “гол”.

6) SET PIECES:
- Если из полей ясно, что это стандарт: “угловой”, “штрафной”, “аут”, “пенальти”.
- Не придумывай стандарт, если он не следует из данных.

SB360 (если есть):
- Используй freeze_frame ТОЛЬКО для очень простого факта:
  * “под давлением” — если рядом с игроком есть как минимум один соперник близко
  * “рядом есть партнёр” — если рядом есть партнёр близко
- Не называй игроков из freeze_frame (их имён там нет).
- visible_area можно не описывать.

ФОРМАТ ОТВЕТА:
Верни СТРОГО валидный JSON без markdown и без лишнего текста:
{{"timestamp":"...","action":"...","commentary":"..."}}

- timestamp бери из event_json["timestamp"] (или minute/second, если timestamp нет).
- action — короткий ярлык: "пас", "приём", "ведение", "перехват", "отбор", "потеря", "удар", "угловой", "аут", "штрафной", "пенальти".
- commentary — 1 короткое предложение (максимум 2), строго по данным.

ВХОДНЫЕ ДАННЫЕ (StatsBomb):
event_json: {event_json}
sb360_json: {sb360_json}
"""


class GigaChatClient:
    def __init__(
        self,
        auth_key: str | None = None,
        *,
        ca_bundle: str | None = "russian_trusted_root_ca.pem",
        verify_ssl_certs: bool = True,
        oauth_url: str = DEFAULT_OAUTH_URL,
        api_base: str = DEFAULT_API_BASE,
        scope: str = DEFAULT_SCOPE,
        model: str = "GigaChat-2",
        timeout: int = 60,
        token_refresh_margin_seconds: int = 30,
        auth_scheme: Literal["Bearer", "Basic"] = "Bearer",
        x_client_id: str | None = None,
    ) -> None:
        self.auth_key = auth_key or os.getenv("GIGACHAT_BASIC_AUTH")
        if not self.auth_key:
            raise ValueError("Set auth_key or env GIGACHAT_BASIC_AUTH.")
        if not self.auth_key.strip():
            raise ValueError("auth_key is empty.")

        self.oauth_url = oauth_url
        self.api_base = api_base.rstrip("/")
        self.scope = scope
        self.model = model
        self.timeout = timeout
        self.token_refresh_margin_seconds = token_refresh_margin_seconds
        self.auth_scheme = auth_scheme
        self.x_client_id = x_client_id

        if verify_ssl_certs:
            self.verify: bool | str = (
                str(ca_bundle) if ca_bundle and Path(ca_bundle).exists() else True
            )
        else:
            self.verify = False

        self._access_token: str | None = None
        self._token_expires_at: datetime | None = None
        self._session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            status=3,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
            backoff_factor=1.0,
        )
        self._session.mount("https://", HTTPAdapter(max_retries=retry))
        self._default_headers = {"Accept": "application/json"}

    @staticmethod
    def _normalize_auth_key(raw: str) -> tuple[str, str]:
        val = raw.strip()
        low = val.lower()
        if low.startswith("basic "):
            return "Basic", val.split(" ", 1)[1].strip()
        if low.startswith("bearer "):
            return "Bearer", val.split(" ", 1)[1].strip()
        return "", val

    def _oauth_authorization_value(self, scheme_override: str | None = None) -> str:
        declared, token = self._normalize_auth_key(self.auth_key)
        if declared:
            return f"{declared} {token}"
        scheme = scheme_override or self.auth_scheme
        return f"{scheme} {token}"

    def _oauth_headers(self, scheme_override: str | None = None) -> dict[str, str]:
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid.uuid4()),
            "Authorization": self._oauth_authorization_value(
                scheme_override=scheme_override
            ),
        }
        if self.x_client_id:
            headers["X-Client-ID"] = self.x_client_id
        return headers

    def _token_is_valid(self) -> bool:
        if not self._access_token:
            return False
        if self._token_expires_at is None:
            return True
        return datetime.now(timezone.utc) < self._token_expires_at

    def _parse_token_expiry(self, payload: dict[str, Any]) -> datetime:
        now = datetime.now(timezone.utc)
        margin = timedelta(seconds=max(self.token_refresh_margin_seconds, 0))

        expires_in = payload.get("expires_in")
        if isinstance(expires_in, (int, float)):
            return now + timedelta(seconds=max(float(expires_in), 0)) - margin

        expires_at = payload.get("expires_at")
        if isinstance(expires_at, (int, float)):
            ts = float(expires_at)
            if ts > 10_000_000_000:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc) - margin

        # Fallback for GigaChat OAuth token lifecycle (~30 minutes).
        return now + timedelta(minutes=29) - margin

    @staticmethod
    def _response_payload(resp: requests.Response) -> Any:
        try:
            return resp.json()
        except Exception:
            return resp.text

    def _raise_api_error(self, resp: requests.Response, context: str) -> None:
        payload = self._response_payload(resp)
        if isinstance(payload, dict):
            message = (
                payload.get("message")
                or payload.get("error_description")
                or payload.get("error")
                or str(payload)
            )
        else:
            message = str(payload)
        raise GigaChatAPIError(f"{context} failed: HTTP {resp.status_code}: {message}")

    def _authorized_headers(
        self, access_token: str, content_type: str | None = "application/json"
    ) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {access_token}", **self._default_headers}
        if content_type:
            headers["Content-Type"] = content_type
        if self.x_client_id:
            headers["X-Client-ID"] = self.x_client_id
        return headers

    def _post_with_token_refresh(
        self,
        *,
        url: str,
        headers: dict[str, str],
        json_body: Any = None,
        form_data: Any = None,
        files: Any = None,
        context: str,
    ) -> requests.Response:
        resp = self._session.post(
            url,
            headers=headers,
            json=json_body,
            data=form_data,
            files=files,
            timeout=self.timeout,
            verify=self.verify,
        )
        if resp.status_code == 401:
            new_token = self.get_access_token(force_refresh=True)
            retry_headers = dict(headers)
            retry_headers["Authorization"] = f"Bearer {new_token}"
            resp = self._session.post(
                url,
                headers=retry_headers,
                json=json_body,
                data=form_data,
                files=files,
                timeout=self.timeout,
                verify=self.verify,
            )
        if resp.status_code >= 400:
            self._raise_api_error(resp, context)
        return resp

    def get_access_token(self, force_refresh: bool = False) -> str:
        if not force_refresh and self._token_is_valid():
            return self._access_token

        declared, _ = self._normalize_auth_key(self.auth_key)
        oauth_try_schemes: list[str | None]
        if declared:
            oauth_try_schemes = [None]
        else:
            alt = "Basic" if self.auth_scheme == "Bearer" else "Bearer"
            oauth_try_schemes = [self.auth_scheme, alt]

        resp: requests.Response | None = None
        for i, scheme in enumerate(oauth_try_schemes):
            resp = self._session.post(
                self.oauth_url,
                headers=self._oauth_headers(scheme_override=scheme),
                data=f"scope={self.scope}",
                timeout=self.timeout,
                verify=self.verify,
            )
            if resp.status_code < 400:
                break
            is_last_try = i == (len(oauth_try_schemes) - 1)
            if is_last_try or resp.status_code not in (400, 401, 403):
                self._raise_api_error(resp, "OAuth")

        if resp is None:
            raise GigaChatAPIError("OAuth failed: no response.")
        payload = resp.json()
        token = payload.get("access_token")
        if not token:
            raise RuntimeError(f"OAuth response has no access_token: {payload}")
        self._access_token = token
        self._token_expires_at = self._parse_token_expiry(payload)
        return token

    def upload_file(self, file_path: str, purpose: str = "general") -> str:
        """
        Upload file to /files. For model usage docs require purpose='general'.
        """
        if purpose != "general":
            raise ValueError("GigaChat docs require purpose='general' for generation.")

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        token = self.get_access_token()
        url = f"{self.api_base}/files"
        headers = self._authorized_headers(token, content_type=None)
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        with open(path, "rb") as f:
            resp = self._post_with_token_refresh(
                url=url,
                headers=headers,
                files={"file": (path.name, f, mime)},
                form_data={"purpose": purpose},
                context="File upload",
            )
        payload = resp.json()
        file_id = payload.get("id")
        if not file_id:
            raise RuntimeError(f"Upload response has no file id: {payload}")
        return file_id

    def list_files(self) -> dict[str, Any]:
        token = self.get_access_token()
        url = f"{self.api_base}/files"
        resp = self._session.get(
            url,
            headers=self._authorized_headers(token, content_type=None),
            timeout=self.timeout,
            verify=self.verify,
        )
        if resp.status_code == 401:
            token = self.get_access_token(force_refresh=True)
            resp = self._session.get(
                url,
                headers=self._authorized_headers(token, content_type=None),
                timeout=self.timeout,
                verify=self.verify,
            )
        if resp.status_code >= 400:
            self._raise_api_error(resp, "List files")
        return resp.json()

    def list_models(self) -> dict[str, Any]:
        """
        List available models for the current account.
        API: GET /models
        """
        token = self.get_access_token()
        url = f"{self.api_base}/models"
        resp = self._session.get(
            url,
            headers=self._authorized_headers(token, content_type=None),
            timeout=self.timeout,
            verify=self.verify,
        )
        if resp.status_code == 401:
            token = self.get_access_token(force_refresh=True)
            resp = self._session.get(
                url,
                headers=self._authorized_headers(token, content_type=None),
                timeout=self.timeout,
                verify=self.verify,
            )
        if resp.status_code >= 400:
            self._raise_api_error(resp, "List models")
        return resp.json()

    def delete_file(self, file_id: str) -> dict[str, Any]:
        token = self.get_access_token()
        url = f"{self.api_base}/files/{file_id}/delete"
        resp = self._post_with_token_refresh(
            url=url,
            headers=self._authorized_headers(token, content_type=None),
            context="Delete file",
        )
        return self._response_payload(resp)

    def chat_completion(
        self,
        prompt: str | None = None,
        *,
        messages: list[dict[str, Any]] | None = None,
        attachments: list[str] | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.1,
        function_call: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        if messages is None:
            if prompt is None:
                raise ValueError("Provide either prompt or messages.")
            user_msg: dict[str, Any] = {"role": "user", "content": prompt}
            if attachments:
                user_msg["attachments"] = attachments
            messages = [{"role": "system", "content": system_prompt}, user_msg]
        elif attachments:
            user_msgs = [m for m in messages if m.get("role") == "user"]
            if not user_msgs:
                raise ValueError(
                    "attachments were provided but no user message in messages."
                )
            user_msgs[-1]["attachments"] = attachments

        token = self.get_access_token()
        url = f"{self.api_base}/chat/completions"
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if function_call is not None:
            payload["function_call"] = function_call

        resp = self._post_with_token_refresh(
            url=url,
            headers=self._authorized_headers(token),
            json_body=payload,
            context="Chat completion",
        )
        return resp.json()

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not m:
                raise
            return json.loads(m.group(0))

    @staticmethod
    def _event_action_from_event_json(event_json: dict[str, Any]) -> str:
        type_obj = event_json.get("type")
        if isinstance(type_obj, dict):
            name = type_obj.get("name")
            if isinstance(name, str):
                return name
        type_name = event_json.get("type_name")
        if isinstance(type_name, str):
            return type_name
        return ""

    @staticmethod
    def append_commentary_result_to_txt(
        result: CommentaryResult,
        txt_path: str,
        *,
        event_id: str | None = None,
    ) -> None:
        path = Path(txt_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "event_id": event_id,
            "timestamp": result.timestamp,
            "action": result.action,
            "commentary": result.commentary,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def generate_commentary(
        self,
        event_json: dict[str, Any],
        sb360_json: dict[str, Any] | None,
        *,
        image_attachment_id: str | None = None,
        temperature: float = 0.1,
        model: str | None = None,
    ) -> CommentaryResult:
        prompt = make_prompt(event_json, sb360_json)
        response = self.chat_completion(
            prompt,
            attachments=[image_attachment_id] if image_attachment_id else None,
            temperature=temperature,
            model=model,
        )
        text = response["choices"][0]["message"]["content"]
        parsed = self._extract_json(text)
        timestamp = str(parsed.get("timestamp") or event_json.get("timestamp") or "")
        action = self._event_action_from_event_json(event_json) or str(
            parsed.get("action", "")
        )
        return CommentaryResult(
            timestamp=timestamp,
            action=action,
            commentary=str(parsed.get("commentary", "")),
            raw_content=text,
            raw_response=response,
        )
