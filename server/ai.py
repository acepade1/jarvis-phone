"""AI 공급자 연결: 키 자동 배정, 스트리밍, 실패 시 다음 AI로 전환(쿨다운 포함)."""
import json
import os
import time

import httpx

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "groq": "openai/gpt-oss-120b",   # llama-3.3-70b 는 Groq 무료 플랜에서 빠짐(Enterprise 전용)
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-5-5",
}
ENV_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
}
NAMES = {"gemini": "Gemini", "groq": "Groq", "openai": "ChatGPT", "claude": "Claude"}
# 돈이 드는 AI 는 사용자의 허락이 있을 때만 쓴다 (무료 AI 가 모두 막혔을 때 묻거나, 사용자가 직접 지목했을 때)
PAID = {"openai", "claude"}

# Groq 무료 플랜에서 쓸 수 있는 모델들 (문서 기준). 앞 모델이 막혀 있으면 다음 모델로 자동 전환
GROQ_ALTS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
_groq_ok: dict = {"model": None}

_transport = None  # 테스트에서 httpx.MockTransport 를 넣는 자리
_cooldown: dict[str, float] = {}


class ProviderError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status}: {message}")
        self.status = status
        self.message = message


# 위키백과 등은 봇 이름을 밝히지 않으면 403 으로 막는다
BOT_UA = "JarvisServer/1.0 (personal voice assistant; https://github.com/acepade1/jarvis-phone)"


def client(timeout: float = 60.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=_transport,
        timeout=httpx.Timeout(timeout, connect=10.0),
        headers={"User-Agent": BOT_UA},
    )


def available() -> dict[str, str]:
    """환경변수에 들어 있는 키 중 쓸 수 있는 것. {공급자: 키}"""
    out = {}
    for prov, env in ENV_KEYS.items():
        k = os.environ.get(env, "").strip()
        if k:
            out[prov] = k
    return out


def model_for(prov: str) -> str:
    return os.environ.get(f"{prov.upper()}_MODEL", "").strip() or DEFAULT_MODELS[prov]


_paid_count = {"day": "", "n": 0}


def paid_limit() -> int:
    try:
        return int(os.environ.get("PAID_DAILY_LIMIT", "30"))
    except ValueError:
        return 30


def paid_budget_ok() -> bool:
    today = time.strftime("%Y-%m-%d", time.gmtime())
    if _paid_count["day"] != today:
        _paid_count.update(day=today, n=0)
    return _paid_count["n"] < paid_limit()


def count_paid() -> None:
    _paid_count["n"] += 1


def paid_available() -> list[str]:
    keys = available()
    return [p for p in ("claude", "openai") if p in keys]


def chain(fast: bool = False, allow_paid: bool = False, prefer: str | None = None) -> list[str]:
    keys = available()
    order = ["groq", "gemini", "openai", "claude"] if fast else ["gemini", "groq", "openai", "claude"]
    allp = [p for p in order if p in keys and (allow_paid or p not in PAID)]
    if prefer in allp:  # 사용자가 "클로드로" 처럼 지목하면 그 AI 를 먼저
        allp.remove(prefer)
        allp.insert(0, prefer)
    now = time.time()
    ok = [p for p in allp if _cooldown.get(p, 0) <= now]
    return ok or allp


def mark_failed(prov: str, err: Exception) -> None:
    msg = str(err).lower()
    hard = any(w in msg for w in ("429", "quota", "rate", "limit", "exceed", "401", "403", "invalid", "api key", "permission"))
    _cooldown[prov] = time.time() + (120 if hard else 20)


def reset_cooldowns() -> None:
    _cooldown.clear()


async def _sse(resp: httpx.Response):
    async for line in resp.aiter_lines():
        line = line.strip()
        if line.startswith("data:"):
            d = line[5:].strip()
            if d and d != "[DONE]":
                try:
                    yield json.loads(d)
                except ValueError:
                    pass


async def _fail(resp: httpx.Response) -> ProviderError:
    body = (await resp.aread()).decode("utf-8", "replace")
    try:
        j = json.loads(body)
        msg = (j.get("error") or {}).get("message") if isinstance(j.get("error"), dict) else str(j)[:200]
    except ValueError:
        msg = body[:200]
    return ProviderError(resp.status_code, msg or "오류")


def groq_extra(model: str) -> dict:
    """추론(reasoning) 모델은 생각에 쓰는 토큰이 한도를 먹으므로 최소로 줄인다."""
    env = os.environ.get("GROQ_REASONING", "").strip()
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": env or "low"}
    if model.startswith("qwen/"):
        return {"reasoning_effort": env or "none"}
    return {}


async def stream(prov: str, system: str, messages: list[dict], max_tokens: int = 400):
    """텍스트 조각을 하나씩 내보내는 비동기 제너레이터. (Groq 는 모델이 막혀 있으면 다음 모델로)"""
    if prov != "groq":
        async for ch in _stream(prov, system, messages, max_tokens, model_for(prov)):
            yield ch
        return
    models = [m for m in dict.fromkeys([_groq_ok["model"], model_for("groq"), *GROQ_ALTS]) if m]
    last: ProviderError | None = None
    for i, m in enumerate(models):
        got = False
        try:
            async for ch in _stream(prov, system, messages, max_tokens, m):
                got = True
                yield ch
            _groq_ok["model"] = m
            return
        except ProviderError as e:
            last = e
            bad_model = e.status in (400, 403, 404) and "model" in e.message.lower()
            if got or not bad_model or i == len(models) - 1:
                raise
    if last:
        raise last


async def _stream(prov: str, system: str, messages: list[dict], max_tokens: int, model: str):
    key = available()[prov]
    async with client() as c:
        if prov == "gemini":
            gc: dict = {"maxOutputTokens": max_tokens}
            if "2.5-flash" in model:
                gc["thinkingConfig"] = {"thinkingBudget": 0}  # 속도 우선
            body = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [
                    {"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]}
                    for m in messages
                ],
                "generationConfig": gc,
            }
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:streamGenerateContent?alt=sse"
            async with c.stream("POST", url, json=body, headers={"x-goog-api-key": key}) as r:
                if r.status_code != 200:
                    raise await _fail(r)
                async for j in _sse(r):
                    parts = (((j.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
                    for p in parts:
                        if p.get("text"):
                            yield p["text"]
        elif prov in ("groq", "openai"):
            url = (
                "https://api.groq.com/openai/v1/chat/completions"
                if prov == "groq"
                else "https://api.openai.com/v1/chat/completions"
            )
            body = {
                "model": model,
                "stream": True,
                "max_completion_tokens": max(max_tokens, 800) if prov == "groq" else max_tokens,
                "messages": [{"role": "system", "content": system}, *messages],
            }
            if prov == "groq":
                body.update(groq_extra(model))
            async with c.stream("POST", url, json=body, headers={"Authorization": f"Bearer {key}"}) as r:
                if r.status_code != 200:
                    raise await _fail(r)
                async for j in _sse(r):
                    d = ((j.get("choices") or [{}])[0].get("delta")) or {}
                    if d.get("content"):
                        yield d["content"]
        else:  # claude
            body = {"model": model, "max_tokens": max_tokens, "stream": True, "system": system, "messages": messages}
            hdr = {"x-api-key": key, "anthropic-version": "2023-06-01"}
            async with c.stream("POST", "https://api.anthropic.com/v1/messages", json=body, headers=hdr) as r:
                if r.status_code != 200:
                    raise await _fail(r)
                async for j in _sse(r):
                    if j.get("type") == "content_block_delta" and (j.get("delta") or {}).get("text"):
                        yield j["delta"]["text"]
