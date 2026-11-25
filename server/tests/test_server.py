import asyncio
import json
import re
import os

import httpx
import pytest
from fastapi.testclient import TestClient

os.environ["JARVIS_PASSWORD"] = "pw-test"
os.environ["JARVIS_SECRET"] = "unit-test-secret"
os.environ["GEMINI_API_KEY"] = "AIzaTEST"
os.environ["GROQ_API_KEY"] = "gsk_TEST"
for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(k, None)

from server import ai, main, tools  # noqa: E402

CALLS = {"gemini": 0, "groq": 0, "claude": 0, "openrouter": 0, "forecast": 0}
CAPTURE = []   # groq 로 간 요청 본문들
SCRIPT = {"gemini": [], "groq": [], "openrouter": []}
OR_CAPTURE = []   # openrouter 로 간 요청 (본문, 헤더)
OR_IDS = ["nvidia/nemotron-3.5-super-vl:free", "google/gemma-4-31b-it:free", "liquid/lfm-3-1.7b:free",
          "google/gemma-4-26b-a4b-it:free", "openrouter/free", "nvidia/nemotron-3.5-lightning:free", "google/lyria-3-pro-preview"]
OR_LIST_FAIL = {"on": False}  # 호출 순서대로 돌려줄 응답 (문자열 or ("err", status, msg))


def _chunks(text):
    return [text[i:i + 7] for i in range(0, len(text), 7)]


def sse_gemini(text):
    return "".join("data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": c}]}}]}) + "\n\n" for c in _chunks(text))


def sse_openai(text):
    return "".join("data: " + json.dumps({"choices": [{"delta": {"content": c}}]}) + "\n\n" for c in _chunks(text)) + "data: [DONE]\n\n"


DDG_HTML = (
    '<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">테스트 제목</a>'
    '<a class="result__snippet">테스트 요약 문장</a></div>'
)


def sse_claude(text):
    nl = chr(10)
    return "".join("data: " + json.dumps({"type": "content_block_delta", "delta": {"type": "text_delta", "text": c}}) + nl + nl
                   for c in _chunks(text))


def handler(request: httpx.Request):
    url = str(request.url)
    if "openrouter.ai/api/v1/models" in url:
        if OR_LIST_FAIL["on"]:
            return httpx.Response(500, text="down")
        return httpx.Response(200, json={"data": [{"id": i} for i in OR_IDS]})
    if "openrouter.ai/api/v1/chat/completions" in url:
        CALLS["openrouter"] += 1
        OR_CAPTURE.append({"body": json.loads(request.content), "headers": dict(request.headers)})
        item = SCRIPT["openrouter"].pop(0) if SCRIPT["openrouter"] else "오픈라우터 기본 응답이에요."
        if isinstance(item, tuple):
            return httpx.Response(item[1], json={"error": {"message": item[2] if len(item) > 2 else "error"}})
        nl = chr(10)
        if isinstance(item, dict):   # 200 으로 시작한 뒤 스트림 도중 오류
            err = "data: " + json.dumps({"error": {"code": 429, "message": "rate limited upstream"}}) + nl + nl
            return httpx.Response(200, content=err.encode(), headers={"content-type": "text/event-stream"})
        return httpx.Response(200, content=sse_openai(item).encode(), headers={"content-type": "text/event-stream"})
    if "api.anthropic.com" in url:
        CALLS["claude"] += 1
        return httpx.Response(200, content=sse_claude("클로드가 답했어요.").encode(), headers={"content-type": "text/event-stream"})
    if "generativelanguage" in url or "api.groq.com" in url:
        prov = "gemini" if "generativelanguage" in url else "groq"
        CALLS[prov] += 1
        if prov == "groq":
            CAPTURE.append(json.loads(request.content))
        item = SCRIPT[prov].pop(0) if SCRIPT[prov] else "기본 응답이에요."
        if isinstance(item, tuple):
            return httpx.Response(item[1], json={"error": {"message": item[2] if len(item) > 2 else "error"}})
        body = sse_gemini(item) if prov == "gemini" else sse_openai(item)
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})
    if "api.open-meteo.com/v1/forecast" in url:
        CALLS["forecast"] += 1
        return httpx.Response(200, json={
            "current": {"temperature_2m": 12.5, "apparent_temperature": 10.1},
            "daily": {"weather_code": [0, 61], "temperature_2m_max": [18, 15], "temperature_2m_min": [9, 8],
                      "precipitation_probability_max": [0, 80]}})
    if "duckduckgo.com/html" in url:
        return httpx.Response(200, text=DDG_HTML)
    return httpx.Response(404, text="not mocked: " + url)


@pytest.fixture(autouse=True)
def setup():
    ai._transport = httpx.MockTransport(handler)
    ai.reset_cooldowns()
    main.TASKS.clear()
    main._login_hits.clear()
    main._chat_hits.clear()
    for k in CALLS:
        CALLS[k] = 0
    SCRIPT["gemini"].clear()
    SCRIPT["groq"].clear()
    ai._paid_count.update(day="", n=0)
    ai._groq_ok["model"] = None
    ai._or_cache.update(t=0.0, models=None)
    OR_CAPTURE.clear()
    OR_LIST_FAIL["on"] = False
    SCRIPT["openrouter"].clear()
    CAPTURE.clear()
    yield
    ai._transport = None


@pytest.fixture
def client():
    with TestClient(main.app) as c:
        yield c


def login(client):
    r = client.post("/api/login", json={"password": "pw-test"})
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


def events(client, h, body, path="/api/chat", method="POST"):
    out = []
    with client.stream(method, path, json=body if method == "POST" else None, headers=h) as r:
        assert r.status_code == 200, r.read()
        for line in r.iter_lines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:]))
    return out


def chat_body(text, rid=None, **kw):
    b = {"messages": [{"role": "user", "content": text}], **kw}
    if rid:
        b["request_id"] = rid
    return b


# ---------------- 인증 ----------------
def test_health_public(client):
    r = client.get("/api/health").json()
    assert r["ok"] and "Gemini" in r["providers"] and "Groq" in r["providers"]


def test_login_and_auth_required(client):
    assert client.post("/api/chat", json=chat_body("안녕")).status_code == 401
    assert client.post("/api/login", json={"password": "wrong"}).status_code == 401
    assert client.post("/api/chat", json=chat_body("안녕"), headers={"Authorization": "Bearer 1.abc"}).status_code == 401
    h = login(client)
    assert client.get("/api/tasks", headers=h).status_code == 200


def test_login_rate_limit(client):
    codes = [client.post("/api/login", json={"password": "x"}).status_code for _ in range(10)]
    assert codes[:8] == [401] * 8 and 429 in codes[8:]


def test_expired_token_rejected():
    tok, _ = main.make_token(ttl=-10)
    assert not main.check_token(tok)


# ---------------- 대화 ----------------
def test_plain_chat_streams(client):
    SCRIPT["gemini"].append("안녕하세요, 자비스입니다.")
    ev = events(client, login(client), chat_body("안녕"))
    deltas = "".join(e["x"] for e in ev if e["t"] == "delta")
    assert deltas == "안녕하세요, 자비스입니다."
    assert ev[-1] == {"t": "final", "x": "안녕하세요, 자비스입니다."}


def test_tool_flow_weather(client):
    SCRIPT["gemini"].append('{"tool":"get_weather","args":{"day":1}}')
    SCRIPT["groq"].append("내일 해운대는 비가 와요.")   # 도구 결과를 요약할 때는 가장 빠른 AI(Groq)가 먼저
    ev = events(client, login(client), chat_body("내일 날씨", city="부산 해운대구"))
    assert CALLS["forecast"] == 1
    assert not any("tool" in e["x"] for e in ev if e["t"] == "delta"), "도구 JSON 이 말로 새어나가면 안 됨"
    assert ev[-1]["t"] == "final" and "비가" in ev[-1]["x"]
    assert any(e["t"] == "status" and "get_weather" in e["x"] for e in ev)


def test_fast_ai_used_after_tool(client):
    SCRIPT["gemini"].append('{"tool":"get_weather","args":{}}')
    SCRIPT["groq"].append("그록이 요약했어요.")
    ev = events(client, login(client), chat_body("오늘 날씨"))
    assert ev[-1]["x"] == "그록이 요약했어요." and CALLS["gemini"] == 1 and CALLS["groq"] == 1


def test_fallback_and_cooldown(client):
    h = login(client)
    SCRIPT["gemini"].append(("err", 429, "quota exceeded"))
    SCRIPT["groq"].append("그록 응답이에요.")
    ev = events(client, h, chat_body("안녕"))
    assert ev[-1]["x"] == "그록 응답이에요."
    SCRIPT["groq"].append("두 번째도 그록.")
    before = CALLS["gemini"]
    events(client, h, chat_body("또 안녕"))
    assert CALLS["gemini"] == before, "쿨다운 중인 AI 는 건너뛰어야 함"


def test_all_providers_fail(client):
    SCRIPT["gemini"].append(("err", 500, "boom"))
    SCRIPT["groq"].append(("err", 500, "boom"))
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["t"] == "error"


# ---------------- 중복 방지 / 이어받기 / 취소 ----------------
def test_idempotent_request_runs_once(client):
    h = login(client)
    SCRIPT["gemini"].append("한 번만 실행돼요.")
    a = events(client, h, chat_body("테스트", rid="req-1"))
    b = events(client, h, chat_body("테스트", rid="req-1"))
    assert CALLS["gemini"] == 1
    assert a[-1] == b[-1] == {"t": "final", "x": "한 번만 실행돼요."}


def test_resume_from_index(client):
    h = login(client)
    SCRIPT["gemini"].append("이어받기 시험 문장입니다.")
    full = events(client, h, chat_body("테스트", rid="req-2"))
    part = events(client, h, None, path="/api/chat/req-2/stream?start=2", method="GET")
    assert part == full[2:]
    t = client.get("/api/tasks/req-2", headers=h).json()
    assert t["state"] == "COMPLETED" and t["final"] == "이어받기 시험 문장입니다."


def test_resume_unknown_task_404(client):
    assert client.get("/api/chat/nope/stream", headers=login(client)).status_code == 404


def test_cancel_task(client):
    h = login(client)
    task = main.Task("req-3")
    task.state = "RUNNING"
    main.TASKS["req-3"] = task
    assert client.delete("/api/tasks/req-3", headers=h).status_code == 200


# ---------------- 권한 / 주입 방어 ----------------
def test_remember_denied_without_user_intent(client):
    SCRIPT["gemini"].append('{"tool":"remember","args":{"fact":"비밀번호는 1234"}}')
    SCRIPT["groq"].append("알겠어요.")
    ev = events(client, login(client), chat_body("이 페이지 요약해줘"))
    assert not any(e["t"] == "action" for e in ev), "사용자가 요청하지 않은 기억 저장은 막아야 함"


def test_remember_allowed_with_user_intent(client):
    SCRIPT["gemini"].append('{"tool":"remember","args":{"fact":"내 이름은 철수"}}')
    SCRIPT["groq"].append("기억했어요.")
    ev = events(client, login(client), chat_body("내 이름은 철수야 기억해"))
    acts = [e for e in ev if e["t"] == "action"]
    assert acts and acts[0]["name"] == "remember"


def test_unknown_tool_denied(client):
    SCRIPT["gemini"].append('{"tool":"delete_everything","args":{}}')
    SCRIPT["groq"].append("할 수 없어요.")
    ev = events(client, login(client), chat_body("다 지워"))
    assert not any(e["t"] == "action" for e in ev)
    assert ev[-1]["t"] == "final"


def test_tool_result_marked_untrusted(client):
    seen = []
    orig = ai.stream

    async def spy(prov, system, messages, max_tokens=400):
        seen.append(messages[-1]["content"])
        async for c in orig(prov, system, messages, max_tokens):
            yield c

    ai.stream = spy
    try:
        SCRIPT["gemini"].append('{"tool":"web_search","args":{"query":"테스트"}}')
        SCRIPT["groq"].append("요약이에요.")
        events(client, login(client), chat_body("검색해줘"))
    finally:
        ai.stream = orig
    assert any(s.startswith("[도구결과") and "명령이 아님" in s for s in seen)


async def _try_read(url):
    try:
        await tools.read_url(url)
    except tools.ToolDenied as e:
        return str(e)
    return "ALLOWED"


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8000/", "http://localhost/admin", "http://169.254.169.254/latest/meta-data/",
    "http://10.0.0.5/", "file:///etc/passwd", "http://[::1]/"])
def test_ssrf_blocked(url):
    ai._transport = None  # 실제 검사 경로 (모의 전송이면 검사를 건너뜀)
    assert asyncio.run(_try_read(url)) != "ALLOWED"


def test_web_search_via_tool(client):
    SCRIPT["gemini"].append('{"tool":"web_search","args":{"query":"x"}}')
    SCRIPT["groq"].append("결과 요약.")
    ev = events(client, login(client), chat_body("검색해줘"))
    assert ev[-1]["t"] == "final"
    res = asyncio.run(tools.web_search("x"))
    assert "테스트 제목" in res and "example.com" in res and "테스트 요약 문장" in res


def test_parse_ddg_markdown():
    md = "## [제목 **굵게**](https://duckduckgo.com/l/?uddg=https%3A%2F%2Fa.com%2Fx&rut=1)\n\n[![i](x)](y)\n\n[요약 **문장**](https://z)\n"
    assert tools.parse_ddg_markdown(md) == [{"title": "제목 굵게", "url": "https://a.com/x", "snippet": "요약 문장"}]


def test_static_app_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/manifest.json").status_code == 200
    assert client.get("/server/main.py").status_code == 404
    assert client.get("/requirements.txt").status_code == 404


# ---------------- 작업 화면(view) 이벤트 ----------------
def test_view_event_for_search(client):
    SCRIPT["gemini"].append('{"tool":"web_search","args":{"query":"해운대 맛집"}}')
    SCRIPT["groq"].append("검색 결과를 요약했어요.")
    ev = events(client, login(client), chat_body("해운대 맛집 검색해줘"))
    views = [e for e in ev if e["t"] == "view"]
    assert len(views) == 1
    v = views[0]["view"]
    assert v["kind"] == "search" and v["title"] == "해운대 맛집"
    assert v["results"][0]["title"] == "테스트 제목" and v["results"][0]["host"] == "example.com"
    assert ev.index(views[0]) < ev.index(ev[-1]), "view 는 최종 답변보다 먼저 와야 함"


def test_view_event_for_weather(client):
    SCRIPT["gemini"].append('{"tool":"get_weather","args":{}}')
    SCRIPT["groq"].append("맑아요.")
    ev = events(client, login(client), chat_body("날씨 알려줘"))
    v = next(e for e in ev if e["t"] == "view")["view"]
    assert v["kind"] == "weather" and "맑음" in v["text"]


def test_no_view_for_client_actions(client):
    SCRIPT["gemini"].append('{"tool":"set_timer","args":{"seconds":5}}')
    SCRIPT["groq"].append("맞췄어요.")
    ev = events(client, login(client), chat_body("5초 타이머"))
    assert not any(e["t"] == "view" for e in ev) and any(e["t"] == "action" for e in ev)


# ---------------- 유료 AI(Claude)는 허락 후에만 ----------------
@pytest.fixture
def with_claude(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-TEST")


def test_paid_not_used_by_default(client, with_claude):
    SCRIPT["gemini"].append("무료 제미나이 답이에요.")
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["x"] == "무료 제미나이 답이에요." and CALLS["claude"] == 0


def test_free_fail_asks_approval_and_does_not_call_paid(client, with_claude):
    SCRIPT["gemini"].append(("err", 500, "boom"))
    SCRIPT["groq"].append(("err", 500, "boom"))
    h = login(client)
    ev = events(client, h, chat_body("안녕", rid="ap-1"))
    assert ev[-1]["t"] == "approval" and "Claude" in ev[-1]["x"] and ev[-1]["providers"] == ["claude"]
    assert CALLS["claude"] == 0, "허락 전에는 유료 AI 를 호출하면 안 됨"
    assert client.get("/api/tasks/ap-1", headers=h).json()["state"] == "WAITING_APPROVAL"


def test_approved_request_uses_claude(client, with_claude):
    h = login(client)
    SCRIPT["gemini"].append(("err", 500, "boom"))
    SCRIPT["groq"].append(("err", 500, "boom"))
    ask = events(client, h, chat_body("안녕", rid="ap-2"))           # 무료 AI 실패 → 허락 요청
    assert ask[-1]["t"] == "approval" and CALLS["claude"] == 0
    ev = events(client, h, chat_body("안녕", rid="ap-2b", allow_paid=True))   # 사용자가 허락 → 같은 질문을 다시
    assert ev[-1]["x"] == "클로드가 답했어요." and CALLS["claude"] == 1


def test_approval_does_not_force_paid_when_free_works(client, with_claude):
    SCRIPT["gemini"].append("무료가 답해요.")
    ev = events(client, login(client), chat_body("안녕", allow_paid=True))
    assert ev[-1]["x"] == "무료가 답해요." and CALLS["claude"] == 0


def test_explicit_mention_counts_as_permission(client, with_claude):
    ev = events(client, login(client), chat_body("클로드로 설명해줘"))
    assert ev[-1]["x"] == "클로드가 답했어요." and CALLS["claude"] == 1 and CALLS["gemini"] == 0


def test_paid_daily_limit(client, with_claude, monkeypatch):
    monkeypatch.setenv("PAID_DAILY_LIMIT", "1")
    h = login(client)
    first = events(client, h, chat_body("클로드로 첫 번째", rid="lim-1"))
    assert first[-1]["x"] == "클로드가 답했어요."
    SCRIPT["gemini"].append("한도가 지나 무료가 답해요.")
    second = events(client, h, chat_body("클로드로 두 번째", rid="lim-2"))
    assert CALLS["claude"] == 1, "한도를 넘으면 유료 AI 를 더 호출하면 안 됨"
    assert second[-1]["x"] == "한도가 지나 무료가 답해요."


def test_without_paid_key_behaviour_unchanged(client):
    SCRIPT["gemini"].append(("err", 500, "boom"))
    SCRIPT["groq"].append(("err", 500, "boom"))
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["t"] == "error"


# ---------------- Groq 무료 모델 / 모델 자동 전환 ----------------
def test_groq_default_is_free_model_with_low_reasoning(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    SCRIPT["groq"].append("그록이 답했어요.")
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["x"] == "그록이 답했어요."
    body = CAPTURE[0]
    assert body["model"] == "openai/gpt-oss-120b", "llama-3.3-70b 는 무료 플랜에서 빠졌으므로 기본값이면 안 됨"
    assert body["reasoning_effort"] == "low" and body["max_completion_tokens"] >= 800


def test_groq_switches_model_when_model_unavailable(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    SCRIPT["groq"].append(("err", 404, "The model `openai/gpt-oss-120b` does not exist or you do not have access to it."))
    SCRIPT["groq"].append("다음 모델이 답했어요.")
    h = login(client)
    ev = events(client, h, chat_body("안녕"))
    assert ev[-1]["x"] == "다음 모델이 답했어요."
    assert [c["model"] for c in CAPTURE] == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]
    # 동작한 모델을 기억해서 다음 요청은 바로 그 모델로
    SCRIPT["groq"].append("또 답했어요.")
    events(client, h, chat_body("또 안녕"))
    assert CAPTURE[-1]["model"] == "openai/gpt-oss-20b" and len(CAPTURE) == 3


def test_groq_non_model_error_does_not_try_other_models(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(main, "RETRY_429_WAIT", 0)
    SCRIPT["groq"] += [("err", 429, "rate limit exceeded"), ("err", 429, "rate limit exceeded")]
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["t"] == "error"
    assert {c["model"] for c in CAPTURE} == {"openai/gpt-oss-120b"}, "한도 오류일 때 다른 모델로 바꾸면 안 됨(모델 문제가 아니므로)"


def test_groq_model_env_override_and_qwen_reasoning(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    SCRIPT["groq"].append("퀜이 답했어요.")
    events(client, login(client), chat_body("안녕"))
    assert CAPTURE[0]["model"] == "qwen/qwen3.8-27b" and CAPTURE[0]["reasoning_effort"] == "none"


def test_request_size_is_capped(client):
    seen = {}
    orig = ai.stream

    async def spy(prov, system, messages, max_tokens=400):
        seen["n"] = len(messages)
        seen["system"] = system
        async for c in orig(prov, system, messages, max_tokens):
            yield c

    ai.stream = spy
    try:
        msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"메시지 {i}"} for i in range(40)]
        msgs[-1] = {"role": "user", "content": "마지막 질문"}
        SCRIPT["gemini"].append("답이에요.")
        events(client, login(client), {"messages": msgs, "memories": [f"기억{i}" for i in range(50)]})
    finally:
        ai.stream = orig
    assert seen["n"] <= 16, "AI 에 보내는 대화는 최근 16개까지만"
    assert "기억49" in seen["system"] and "기억29" not in seen["system"], "기억은 최근 20개까지만"


# ---------------- 분당 한도(429) 한 번 재시도 ----------------
def test_single_provider_429_retries_once(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(main, "RETRY_429_WAIT", 0)
    SCRIPT["groq"].append(("err", 429, "Rate limit reached for model: tokens per minute (TPM)"))
    SCRIPT["groq"].append("다시 시도해서 답했어요.")
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["x"] == "다시 시도해서 답했어요." and CALLS["groq"] == 2
    assert any(e["t"] == "status" and "바빠요" in e["x"] for e in ev)


def test_429_with_two_providers_falls_through_without_waiting(client, monkeypatch):
    monkeypatch.setattr(main, "RETRY_429_WAIT", 30)   # 기다리면 테스트가 멈추므로, 기다리지 않는 것을 증명
    SCRIPT["gemini"].append(("err", 429, "quota"))
    SCRIPT["groq"].append("그록이 바로 답했어요.")
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["x"] == "그록이 바로 답했어요."


def test_single_provider_429_twice_gives_error(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(main, "RETRY_429_WAIT", 0)
    SCRIPT["groq"] += [("err", 429, "rate limit"), ("err", 429, "rate limit")]
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["t"] == "error" and CALLS["groq"] == 2


# ---------------- OpenRouter (무료 모델) ----------------
@pytest.fixture
def only_openrouter(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-TEST")


def test_openrouter_listed_in_health_and_is_free(client, only_openrouter):
    assert "OpenRouter" in client.get("/api/health").json()["providers"]
    assert "openrouter" not in ai.PAID
    assert ai.chain(False) == ["openrouter"]


def test_openrouter_only_answers_with_free_gemma_first(client, only_openrouter):
    SCRIPT["openrouter"].append("오픈라우터가 답했어요.")
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["x"] == "오픈라우터가 답했어요."
    req = OR_CAPTURE[0]
    assert req["body"]["model"] == "google/gemma-4-31b-it:free"
    assert req["body"]["model"].endswith(":free"), "유료 모델은 절대 쓰지 않음"
    assert req["headers"]["authorization"] == "Bearer sk-or-v1-TEST"
    assert req["headers"]["x-title"] == "Jarvis" and req["body"]["max_tokens"] >= 800


def test_openrouter_switches_model_on_error(client, only_openrouter):
    SCRIPT["openrouter"].append(("err", 429, "rate limited"))
    SCRIPT["openrouter"].append("다음 모델이 답했어요.")
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["x"] == "다음 모델이 답했어요."
    assert [c["body"]["model"] for c in OR_CAPTURE] == ["google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free"]


def test_openrouter_midstream_error_switches_model(client, only_openrouter):
    SCRIPT["openrouter"].append({"midstream": True})
    SCRIPT["openrouter"].append("복구해서 답했어요.")
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["x"] == "복구해서 답했어요." and len(OR_CAPTURE) == 2


def test_openrouter_empty_reply_switches_model(client, only_openrouter):
    SCRIPT["openrouter"].append("")
    SCRIPT["openrouter"].append("두 번째 모델 답이에요.")
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["x"] == "두 번째 모델 답이에요." and len(OR_CAPTURE) == 2


def test_openrouter_gives_up_after_three_models(client, only_openrouter):
    SCRIPT["openrouter"] += [("err", 500, "x"), ("err", 500, "x"), ("err", 500, "x"), "도달하면 안 됨"]
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["t"] == "error" and len(OR_CAPTURE) == 3


def test_openrouter_candidates_skip_vision_and_media_models(client, only_openrouter):
    ms = asyncio.run(ai.or_candidates())
    assert ms[:3] == ["google/gemma-4-31b-it:free", "google/gemma-4-26b-a4b-it:free", "openrouter/free"]
    assert not any(re.search(r"vl|lyria|lfm", m) for m in ms)
    assert "nvidia/nemotron-3.5-lightning:free" in ms and len(ms) <= 5


def test_openrouter_static_fallback_when_list_unreadable(client, only_openrouter):
    OR_LIST_FAIL["on"] = True
    assert asyncio.run(ai.or_candidates()) == ai.OR_STATIC


def test_openrouter_model_env_override(client, only_openrouter, monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free")
    assert asyncio.run(ai.or_candidates())[0] == "nvidia/nemotron-3.5-lightning:free"


def test_groq_failure_falls_back_to_openrouter(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-TEST")
    monkeypatch.setattr(main, "RETRY_429_WAIT", 0)
    SCRIPT["groq"].append(("err", 500, "boom"))
    SCRIPT["openrouter"].append("그록이 막혀서 오픈라우터가 답했어요.")
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["x"] == "그록이 막혀서 오픈라우터가 답했어요." and CALLS["groq"] == 1


def test_openrouter_does_not_unlock_paid(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-TEST")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-TEST")
    SCRIPT["openrouter"].append("무료로 답해요.")
    ev = events(client, login(client), chat_body("안녕"))
    assert ev[-1]["x"] == "무료로 답해요." and CALLS["claude"] == 0
