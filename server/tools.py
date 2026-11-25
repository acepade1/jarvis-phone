"""서버 도구. 서버에서 실행하는 도구(읽기 전용)와, 폰에서 실행해야 하는 동작(client action)을 구분한다.

권한 등급
  L0  정보 조회 (검색·날씨·위키·URL 읽기)        → 항상 허용
  L1  폰 안의 가벼운 동작 (타이머·링크·기억 저장)  → 허용하되 기억 변경은 사용자가 직접 요청했을 때만
  L2+ 파일 변경·메일 발송·결제 등                 → 아직 없음 (Phase 3~4, 별도 승인 필요)
"""
import asyncio
import ipaddress
import re
import socket
from urllib.parse import parse_qs, quote, unquote, urlparse

from bs4 import BeautifulSoup

from . import ai

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

TOOL_LEVEL = {
    "get_weather": 0, "web_search": 0, "news": 0, "read_url": 0, "wiki": 0, "youtube": 0,
    "set_timer": 1, "open_link": 1, "maps": 1, "launch_app": 1, "battery": 1, "media": 1,
    "remember": 1, "forget": 1,
}
# 폰에서 실행되는 동작 (서버는 이름과 인자만 전달)
CLIENT_ACTIONS = {"set_timer", "open_link", "maps", "launch_app", "battery", "media", "remember", "forget"}
MEMORY_INTENT = re.compile(r"기억|외워|잊어|잊지|지워|삭제|remember|forget", re.I)

WX = {0: "맑음", 1: "대체로 맑음", 2: "구름 조금", 3: "흐림", 45: "안개", 48: "안개", 51: "이슬비", 53: "이슬비",
      55: "이슬비", 61: "비", 63: "비", 65: "강한 비", 71: "눈", 73: "눈", 75: "강한 눈", 80: "소나기",
      81: "소나기", 82: "강한 소나기", 95: "뇌우", 96: "뇌우", 99: "뇌우"}
GEO = [("해운대", 35.1631, 129.1639, "부산 해운대구"), ("부산", 35.1796, 129.0756, "부산"),
       ("서울", 37.5665, 126.978, "서울"), ("인천", 37.4563, 126.7052, "인천"), ("대구", 35.8714, 128.6014, "대구"),
       ("대전", 36.3504, 127.3845, "대전"), ("광주", 35.1595, 126.8526, "광주"), ("울산", 35.5384, 129.3114, "울산"),
       ("제주", 33.4996, 126.5312, "제주")]


class ToolDenied(Exception):
    pass


# ---------- 안전한 URL 가져오기 (SSRF 방어) ----------
def _is_public_host(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            return False
    return True


async def safe_get(url: str, max_chars: int = 200_000) -> str:
    cur = url
    async with ai.client() as c:
        for _ in range(4):
            u = urlparse(cur)
            if u.scheme not in ("http", "https") or not u.hostname:
                raise ToolDenied("http/https 주소만 읽을 수 있어요")
            if ai._transport is None and not await asyncio.to_thread(_is_public_host, u.hostname):
                raise ToolDenied("내부 주소는 읽을 수 없어요")
            r = await c.get(cur, headers={"User-Agent": BROWSER_UA, "Accept-Language": "ko,en;q=0.8"}, follow_redirects=False)
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                cur = str(httpx_join(cur, r.headers["location"]))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"페이지 읽기 실패({r.status_code})")
            return r.text[:max_chars]
    raise RuntimeError("리다이렉트가 너무 많아요")


def httpx_join(base: str, loc: str):
    import httpx
    return httpx.URL(base).join(loc)


def html_to_text(html: str, limit: int = 3500) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "form"]):
        t.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))
    return text[:limit]


# ---------- 도구 구현 ----------
async def get_weather(city: str = "", day: int = 0, default_city: str = "부산 해운대구") -> str:
    city = (city or default_city).strip()
    hit = next((g for g in GEO if g[0] in city), None)
    async with ai.client() as c:
        if hit:
            lat, lon, name = hit[1], hit[2], hit[3]
        else:
            lat = lon = name = None
            for q in [city, *reversed(city.split())]:
                r = await c.get("https://geocoding-api.open-meteo.com/v1/search",
                                params={"name": q, "count": 1, "language": "ko"})
                res = (r.json().get("results") or [None])[0]
                if res:
                    lat, lon, name = res["latitude"], res["longitude"], res["name"]
                    break
            if lat is None:
                return f"{city} 위치를 찾지 못했어요"
        w = (await c.get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 2,
            "current": "temperature_2m,apparent_temperature",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max"})).json()
    d = 1 if int(day or 0) >= 1 else 0
    dl, cur = w["daily"], w["current"]
    out = (f"{name} {'내일' if d else '오늘'}: {WX.get(dl['weather_code'][d], '정보없음')}, "
           f"최저 {dl['temperature_2m_min'][d]}도 최고 {dl['temperature_2m_max'][d]}도, "
           f"강수확률 {dl['precipitation_probability_max'][d]}%")
    if not d:
        out += f". 현재 {cur['temperature_2m']}도, 체감 {cur['apparent_temperature']}도"
    return out


def parse_ddg_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    out = []
    for res in soup.select(".result"):
        a = res.select_one("a.result__a")
        if not a:
            continue
        href = a.get("href", "")
        if "uddg=" in href:
            href = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
        sn = res.select_one(".result__snippet")
        out.append({"title": a.get_text(" ", strip=True), "url": href,
                    "snippet": sn.get_text(" ", strip=True)[:220] if sn else ""})
    return out


def parse_ddg_markdown(md: str) -> list[dict]:
    """Jina Reader 를 거쳤을 때의 마크다운 형식"""
    out = []
    for block in re.split(r"^## ", md, flags=re.M)[1:]:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        m = re.match(r"\[(.+?)\]\((https?:[^)]+)\)", lines[0]) if lines else None
        if not m:
            continue
        url = m.group(2)
        if "uddg=" in url:
            url = unquote(parse_qs(urlparse(url).query).get("uddg", [url])[0])
        sn = next((l for l in lines[1:] if l.startswith("[") and not l.startswith("[![")), "")
        sn = re.sub(r"\]\(.*$", "", sn.lstrip("[")).replace("**", "")[:220]
        out.append({"title": m.group(1).replace("**", ""), "url": url, "snippet": sn})
    return out


async def web_search_results(query: str, n: int = 4) -> list[dict]:
    results: list[dict] = []
    async with ai.client(timeout=10) as c:
        try:  # 1) DuckDuckGo 직접 (봇으로 의심되면 202/빈 결과 → 바로 다음 경로)
            r = await c.post("https://html.duckduckgo.com/html/", data={"q": query, "kl": "kr-kr"},
                             headers={"User-Agent": BROWSER_UA, "Accept-Language": "ko,en;q=0.8"})
            if r.status_code == 200:
                results = parse_ddg_html(r.text)
        except Exception:
            results = []
        for attempt in range(2):  # 2) Jina Reader 경유 (최대 2번)
            if results:
                break
            try:
                r = await c.get("https://r.jina.ai/https://html.duckduckgo.com/html/?kl=kr-kr&q=" + quote(query),
                                headers={"Accept": "text/plain"})
                if r.status_code == 200:
                    results = parse_ddg_markdown(r.text)
            except Exception:
                results = []
            if not results and attempt == 0:
                await asyncio.sleep(1.0)
    return [{**x, "host": urlparse(x["url"]).hostname or ""} for x in results[:n]]


def format_results(results: list[dict]) -> str:
    if not results:
        return "검색 결과가 없어요"
    return "\n".join(f"{i + 1}. {x['title']} ({x['host']}) — {x['snippet']}" for i, x in enumerate(results))


async def web_search(query: str, n: int = 4) -> str:
    return format_results(await web_search_results(query, n))


async def wiki_data(query: str) -> tuple[str, str] | None:
    async with ai.client(timeout=12) as c:
        r = await c.get("https://ko.wikipedia.org/w/api.php", params={
            "action": "query", "list": "search", "srsearch": query, "format": "json", "srlimit": 1})
        hits = (r.json().get("query") or {}).get("search") or []
        if not hits:
            return None
        title = hits[0]["title"]
        s = (await c.get("https://ko.wikipedia.org/api/rest_v1/page/summary/" + quote(title))).json()
    return title, (s.get("extract") or "")[:600]


async def wiki(query: str, source: str = "wikipedia") -> str:
    if source == "namu":  # 나무위키는 봇 차단 → 검색 결과 요약으로 대체
        return "(나무위키 검색 결과)\n" + await web_search("site:namu.wiki " + query)
    d = await wiki_data(query)
    return f"{d[0]}: {d[1]}" if d else "위키백과에서 찾지 못했어요"


async def read_url(url: str) -> str:
    html = await safe_get(url)
    return html_to_text(html)


async def youtube_search(query: str) -> dict | None:
    """영상 ID와 제목을 찾는다. (서버 IP 에 따라 막힐 수 있어 여러 경로를 시도)"""
    async with ai.client() as c:
        try:
            r = await c.get("https://www.youtube.com/results", params={"search_query": query},
                            headers={"User-Agent": BROWSER_UA, "Accept-Language": "ko,en;q=0.8",
                                     "Cookie": "CONSENT=YES+1; SOCS=CAI"})
            m = re.search(r'"videoRenderer":\{"videoId":"([\w-]{11})".*?"title":\{"runs":\[\{"text":"(.*?)"\}', r.text, re.S)
            if m:
                return {"id": m.group(1), "title": m.group(2).encode().decode("unicode_escape", "ignore") if "\\u" in m.group(2) else m.group(2)}
        except Exception:
            pass
        for host in ("https://api.piped.private.coffee", "https://pipedapi.adminforge.de"):
            try:
                r = await c.get(f"{host}/search", params={"q": query, "filter": "videos"}, timeout=6)
                items = [i for i in r.json().get("items", []) if "v=" in (i.get("url") or "")]
                if items:
                    return {"id": items[0]["url"].split("v=")[1].split("&")[0], "title": items[0].get("title", "")}
            except Exception:
                continue
    return None


# ---------- 실행 진입점 ----------
async def run_tool(name: str, args: dict, ctx: dict) -> tuple[str, dict | None, dict | None]:
    """returns (AI에게 줄 결과 텍스트, 폰에서 실행할 동작 or None, 작업 화면에 보여줄 내용 or None). 권한 검사는 여기서 한다."""
    if name not in TOOL_LEVEL:
        raise ToolDenied(f"알 수 없는 도구: {name}")
    args = args if isinstance(args, dict) else {}
    level = TOOL_LEVEL[name]
    # 기억 변경은 '사용자가 직접 요청했을 때만' (외부 자료에 심어 둔 지시 방어)
    if name in ("remember", "forget") and not MEMORY_INTENT.search(ctx.get("last_user", "")):
        raise ToolDenied("사용자가 요청하지 않은 기억 변경은 허용되지 않아요")
    if name in CLIENT_ACTIONS:
        return "폰에서 실행했어요", {"name": name, "args": args, "level": level}, None
    if name == "get_weather":
        txt = await get_weather(str(args.get("city", "")), args.get("day", 0), ctx.get("city") or "부산 해운대구")
        return txt, None, {"kind": "weather", "title": "날씨", "text": txt}
    if name in ("web_search", "news"):
        q = str(args.get("query", "")) if name == "web_search" else f"{args.get('topic') or '오늘 주요'} 뉴스"
        rs = await web_search_results(q, 4 if name == "web_search" else 5)
        return format_results(rs), None, {"kind": "search", "title": q, "results": rs}
    if name == "read_url":
        url = str(args.get("url", ""))
        txt = await read_url(url)
        return txt, None, {"kind": "page", "title": urlparse(url).hostname or url, "url": url, "text": txt[:600]}
    if name == "wiki":
        q, src = str(args.get("query", "")), str(args.get("source", "wikipedia"))
        if src == "namu":
            rs = await web_search_results("site:namu.wiki " + q, 4)
            return "(나무위키 검색 결과)\n" + format_results(rs), None, {"kind": "search", "title": "나무위키: " + q, "results": rs}
        d = await wiki_data(q)
        if not d:
            return "위키백과에서 찾지 못했어요", None, None
        return f"{d[0]}: {d[1]}", None, {"kind": "wiki", "title": d[0], "text": d[1]}
    if name == "youtube":
        v = await youtube_search(str(args.get("query", "")))
        if not v:
            return "영상을 바로 못 찾았어요. 유튜브 검색 링크를 보여 드릴게요", {
                "name": "open_link", "level": 1,
                "args": {"url": "https://www.youtube.com/results?search_query=" + quote(str(args.get("query", ""))),
                         "label": "유튜브 검색"}}, None
        return f"재생 시작: {v['title']}", {"name": "play_video", "args": v, "level": 1}, None
    raise ToolDenied(name)
