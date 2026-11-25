import asyncio
import json
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

CALLS = {"gemini": 0, "groq": 0, "forecast": 0}
SCRIPT = {"gemini": [], "groq": []}  # 호출 순서대로 돌려줄 응답 (문자열 or ("err", status, msg))


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


def handler(request: httpx.Request):
    url = str(request.url)
    if "generativelanguage" in url or "api.groq.com" in url:
        prov = "gemini" if "generativelanguage" in url else "groq"
        CALLS[prov] += 1
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
