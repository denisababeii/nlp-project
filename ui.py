#Minimal FastAPI frontend for the course overlap assistant.

from __future__ import annotations

import os
import time
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

APP_TITLE = "Course Overlap Assistant"
DEFAULT_BACKEND_URL = os.getenv("COURSE_OVERLAP_BACKEND_URL", "http://127.0.0.1:8001")
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("COURSE_OVERLAP_TIMEOUT_SECONDS", "60"))

app = FastAPI(title=APP_TITLE)


class AnalyzeRequest(BaseModel):
    """Request from the browser to the frontend."""

    backend_url: str = Field(default=DEFAULT_BACKEND_URL)
    question: str = Field(min_length=1)
    mode: Literal["simple", "advanced"] = "simple"
    timeout_seconds: float = Field(default=DEFAULT_TIMEOUT_SECONDS, ge=1.0, le=180.0)


def endpoint_for_mode(mode: str) -> str:
    """Return backend endpoint for frontend mode."""
    return "/analyze-rag" if mode == "advanced" else "/analyze"


async def post_to_backend(request: AnalyzeRequest) -> tuple[dict[str, Any], float, int, str]:
    """Forward the question to the selected backend endpoint.

    Parameters
    ----------
    request : AnalyzeRequest
        Browser request containing backend URL, question and selected mode.

    Returns
    -------
    tuple[dict[str, Any], float, int, str]
        JSON payload, latency in milliseconds, status code and endpoint path.

    Raises
    ------
    HTTPException
        Raised with readable messages for common connection and backend errors.
    """
    endpoint = endpoint_for_mode(request.mode)
    url = f"{request.backend_url.rstrip('/')}{endpoint}"
    start = time.perf_counter()

    try:
        async with httpx.AsyncClient(timeout=request.timeout_seconds) as client:
            response = await client.post(url, json={"question": request.question})
        latency_ms = (time.perf_counter() - start) * 1000.0
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                f"The backend did not respond within {request.timeout_seconds:.1f} seconds. "
                "Check that the backend is running and that the selected mode is not stuck."
            ),
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "The frontend could not reach the backend. Check the backend URL, "
                "port number, and whether uvicorn is running."
            ),
        ) from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=(
                f"The backend returned HTTP {response.status_code}. "
                f"Response body: {response.text[:600]}"
            ),
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="The backend responded, but the response body was not valid JSON.",
        ) from exc

    return payload, latency_ms, response.status_code, endpoint


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """Render the single-page frontend."""
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{APP_TITLE}</title>
  <style>
    :root {{
      --bg: #f7f7f5;
      --panel: #ffffff;
      --text: #171717;
      --muted: #6b6b6b;
      --soft: #eeeeeb;
      --border: #deded8;
      --black: #171717;
      --white: #ffffff;
      --red: rgb(153, 0, 0);
      --red-soft: #fff1f1;
      --green-soft: #ecfdf3;
      --green: #167047;
      --amber-soft: #fff8e6;
      --amber: #8a6100;
      --shadow: 0 24px 80px rgba(0, 0, 0, 0.08);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top, #ffffff 0, var(--bg) 46rem),
        var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    button, textarea, input {{ font: inherit; }}

    .page {{
      min-height: 100vh;
      width: min(960px, calc(100% - 32px));
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      padding: 28px 0 48px;
    }}

    .topbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}

    .brand {{
      display: flex;
      align-items: center;
      gap: 10px;
      color: var(--muted);
      font-size: 14px;
      font-weight: 600;
    }}

    .brand-mark {{
      width: 34px;
      height: 34px;
      border-radius: 999px;
      background: var(--black);
      color: var(--white);
      display: grid;
      place-items: center;
      box-shadow: 0 10px 24px rgba(0, 0, 0, 0.16);
    }}

    .mode-switch {{
      display: flex;
      gap: 4px;
      padding: 4px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 999px;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    }}

    .mode-button {{
      position: relative;
      border: 0;
      border-radius: 999px;
      background: transparent;
      color: var(--muted);
      padding: 9px 14px;
      cursor: pointer;
      transition: 160ms ease;
      font-size: 14px;
      font-weight: 600;
      white-space: nowrap;
    }}

    .mode-button.active {{
      background: var(--black);
      color: var(--white);
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.14);
    }}

    .hero {{
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
      text-align: center;
      padding: 15px 0 24px;
    }}

    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.8);
      color: var(--muted);
      padding: 7px 12px;
      font-size: 13px;
      font-weight: 600;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.04);
    }}

    h1 {{
      max-width: 760px;
      margin: 22px auto 0;
      font-size: clamp(36px, 7vw, 64px);
      line-height: 1.02;
      letter-spacing: -0.055em;
      font-weight: 720;
    }}

    .subtitle {{
      max-width: 620px;
      margin: 18px auto 0;
      color: var(--muted);
      font-size: 17px;
      line-height: 1.7;
    }}

    .composer {{
      width: min(760px, 100%);
      margin: 34px auto 0;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 30px;
      padding: 10px;
      box-shadow: var(--shadow);
    }}

    textarea {{
      width: 100%;
      min-height: 132px;
      resize: vertical;
      border: 0;
      outline: none;
      background: transparent;
      padding: 18px 18px 12px;
      color: var(--text);
      font-size: 16px;
      line-height: 1.65;
    }}

    textarea::placeholder {{ color: #9a9a93; }}

    .composer-footer {{
      border-top: 1px solid var(--soft);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 8px 2px;
    }}

    .endpoint {{
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }}

    .dot {{
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #c7c7c0;
    }}

    .ask-button {{
      width: auto;
      border: 0;
      border-radius: 999px;
      background: var(--black);
      color: var(--white);
      padding: 12px 18px;
      cursor: pointer;
      font-weight: 700;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: 160ms ease;
    }}

    .ask-button:hover {{ transform: translateY(-1px); }}
    .ask-button:disabled {{ opacity: 0.48; cursor: not-allowed; transform: none; }}

    .settings {{
      width: min(760px, 100%);
      margin: 14px auto 0;
      display: grid;
      grid-template-columns: 1fr 150px;
      gap: 10px;
    }}

    .settings input {{
      width: 100%;
      border: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.78);
      border-radius: 999px;
      padding: 11px 14px;
      color: var(--muted);
      outline: none;
    }}

    .results {{
      width: min(760px, 100%);
      margin: 28px auto 0;
      display: grid;
      gap: 14px;
      text-align: left;
    }}

    .card {{
      background: rgba(255, 255, 255, 0.9);
      border: 1px solid var(--border);
      border-radius: 26px;
      padding: 22px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    }}

    .card-title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 14px;
      font-weight: 700;
      margin-bottom: 12px;
    }}

    .answer {{
      white-space: pre-wrap;
      color: #2b2b2b;
      font-size: 16px;
      line-height: 1.72;
      margin: 0;
    }}

    .course-row {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
    }}

    .label {{
      color: #9a9a93;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-size: 12px;
      font-weight: 800;
    }}

    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 9px; }}

    .chip {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      background: #f1f1ee;
      padding: 7px 11px;
      color: #444;
      font-size: 14px;
      font-weight: 650;
    }}

    .rank-item {{
      border: 1px solid var(--soft);
      background: #fafaf8;
      border-radius: 20px;
      padding: 16px;
      margin-top: 10px;
    }}

    .rank-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 14px;
    }}

    .course-code {{
      font-size: 20px;
      font-weight: 780;
      letter-spacing: -0.02em;
    }}

    .similarity {{
      color: var(--muted);
      font-size: 14px;
      margin-top: 2px;
    }}

    .badge {{
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 12px;
      font-weight: 800;
      white-space: nowrap;
    }}

    .badge.low {{ background: var(--green-soft); color: var(--green); }}
    .badge.moderate {{ background: var(--amber-soft); color: var(--amber); }}
    .badge.high {{ background: var(--red-soft); color: var(--red); }}

    .recommendation {{
      margin: 12px 0 0;
      color: #414141;
      line-height: 1.55;
      font-size: 14px;
    }}

    .evidence {{
      margin: 12px 0 0;
      padding-left: 20px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.55;
    }}

    .source-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 10px;
    }}

    .source {{
      border: 1px solid var(--soft);
      background: #fafaf8;
      border-radius: 18px;
      padding: 13px;
    }}

    .source-code {{ font-weight: 800; }}
    .source-title {{ color: var(--muted); font-size: 13px; line-height: 1.45; margin-top: 4px; }}

    .status {{
      border-radius: 22px;
      padding: 16px;
      font-size: 14px;
      line-height: 1.55;
      white-space: pre-wrap;
    }}

    .status.info {{ background: #f1f1ee; color: var(--muted); }}
    .status.error {{ background: var(--red-soft); color: var(--red); border: 1px solid #ffd2d2; }}

    .metrics {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
    }}

    .metric {{
      border: 1px solid var(--soft);
      background: #fafaf8;
      border-radius: 999px;
      padding: 7px 10px;
    }}

    @media (max-width: 680px) {{
      .topbar {{ align-items: flex-start; flex-direction: column; }}
      .settings {{ grid-template-columns: 1fr; }}
      .course-row {{ grid-template-columns: 1fr; }}
      .composer-footer {{ align-items: stretch; flex-direction: column; }}
      .ask-button {{ justify-content: center; width: 100%; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header class="topbar">

      <nav class="mode-switch" aria-label="Analysis mode">
        <button id="simpleMode" class="mode-button active" type="button" onclick="setMode('simple')">Simple</button>
        <button id="advancedMode" class="mode-button" type="button" onclick="setMode('advanced')">Advanced (RAG)</button>
      </nav>
    </header>

    <section class="hero">
      <h1>Are your courses overlapping?</h1>
      <p class="subtitle">
        Enter your completed course codes and the courses you are considering.
        The assistant compares learning objectives and academic prerequisites.
      </p>

      <form class="composer" onsubmit="submitQuestion(event)">
        <textarea id="question" placeholder="Example: I have completed 01002. Should I take 01003 or 01017?"></textarea>
        <div class="composer-footer">
          <div class="endpoint">
            <span class="dot"></span>
            <span id="endpointText">Using /analyze</span>
          </div>
          <button id="askButton" class="ask-button" type="submit">Ask →</button>
        </div>
      </form>

      <div class="settings" style="display: none;">
        <input id="backendUrl" type="url" value="{DEFAULT_BACKEND_URL}" placeholder="Backend URL" required />
        <input id="timeoutSeconds" type="number" value="{DEFAULT_TIMEOUT_SECONDS}" min="1" max="180" placeholder="Timeout (seconds)" required />
      </div>

      <section id="results" class="results"></section>
    </section>
  </main>

<script>
let currentMode = 'simple';
let metrics = {{
  totalRequests: 0,
  successfulRequests: 0,
  failedRequests: 0,
  lastLatencyMs: null
}};

function escapeHtml(value) {{
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}}

function setMode(mode) {{
  currentMode = mode;
  document.getElementById('simpleMode').classList.toggle('active', mode === 'simple');
  document.getElementById('advancedMode').classList.toggle('active', mode === 'advanced');
  document.getElementById('endpointText').textContent = mode === 'advanced' ? 'Using /analyze-rag' : 'Using /analyze';
}}

function similarityPercent(value) {{
  if (typeof value !== 'number') return '—';
  return `${{Math.round(value * 100)}}%`;
}}

function overlapLevel(value) {{
  const level = String(value || 'low').toLowerCase();
  if (level === 'high' || level === 'moderate') return level;
  return 'low';
}}

function renderMetrics(extra) {{
  const parts = [
    `<span class="metric">Endpoint: ${{escapeHtml(extra.endpoint || '—')}}</span>`,
    `<span class="metric">Latency: ${{extra.latency_ms ? extra.latency_ms.toFixed(1) + ' ms' : '—'}}</span>`,
    `<span class="metric">Requests: ${{metrics.totalRequests}}</span>`,
    `<span class="metric">Successful: ${{metrics.successfulRequests}}</span>`,
    `<span class="metric">Failed: ${{metrics.failedRequests}}</span>`
  ];
  return `<div class="metrics">${{parts.join('')}}</div>`;
}}

function renderCourseChips(courses) {{
  if (!Array.isArray(courses) || courses.length === 0) return '<span class="chip">None detected</span>';
  return courses.map((course) => `<span class="chip">${{escapeHtml(course)}}</span>`).join('');
}}

function renderRanking(ranking) {{
  if (!Array.isArray(ranking) || ranking.length === 0) return '';

  const rows = ranking.map((item) => {{
    const level = overlapLevel(item.overlap_level);
    const recommendation = item.rag_recommendation || item.recommendation || '';
    const evidence = Array.isArray(item.evidence) && item.evidence.length > 0
      ? `<ul class="evidence">${{item.evidence.map((point) => `<li>${{escapeHtml(point)}}</li>`).join('')}}</ul>`
      : '';

    return `
      <article class="rank-item">
        <div class="rank-head">
          <div>
            <div class="course-code">${{escapeHtml(item.course_number || 'Unknown course')}}</div>
            <div class="similarity">Similarity: ${{similarityPercent(item.similarity)}}</div>
          </div>
          <span class="badge ${{level}}">${{level}} overlap</span>
        </div>
        <p class="recommendation">${{escapeHtml(recommendation)}}</p>
        ${{evidence}}
      </article>
    `;
  }}).join('');

  return `
    <section class="card">
      <div class="card-title"><span>Ranked overlap</span><span>${{currentMode === 'advanced' ? 'RAG + TF-IDF' : 'TF-IDF'}}</span></div>
      ${{rows}}
    </section>
  `;
}}

function renderSources(sources) {{
  if (currentMode !== 'advanced' || !Array.isArray(sources) || sources.length === 0) return '';

  const sourceCards = sources.map((source) => `
    <div class="source">
      <div class="source-code">${{escapeHtml(source.course_code)}}</div>
      <div class="source-title">${{escapeHtml(source.title || 'Untitled course')}}</div>
    </div>
  `).join('');

  return `
    <section class="card">
      <div class="card-title"><span>Retrieved sources</span><span>${{sources.length}} courses</span></div>
      <div class="source-grid">${{sourceCards}}</div>
    </section>
  `;
}}

function renderResponse(data, meta) {{
  if (data.error) {{
    return `
      <div class="status error"><strong>Course check stopped.</strong>\n${{escapeHtml(data.error)}}</div>
      ${{renderMetrics(meta)}}
    `;
  }}

  const answer = data.answer
    ? `<section class="card"><div class="card-title">Analysis</div><p class="answer">${{escapeHtml(data.answer)}}</p></section>`
    : '';

  const courses = `
    <section class="card">
      <div class="course-row">
        <div>
          <div class="label">Completed</div>
          <div class="chips">${{renderCourseChips(data.completed_courses)}}</div>
        </div>
        <div>
          <div class="label">Compared</div>
          <div class="chips">${{renderCourseChips(data.compared_courses)}}</div>
        </div>
      </div>
    </section>
  `;

  return `
    ${{renderMetrics(meta)}}
    ${{answer}}
    ${{courses}}
    ${{renderRanking(data.ranking)}}
  `;
}}

async function submitQuestion(event) {{
  event.preventDefault();

  const question = document.getElementById('question').value.trim();
  const backendUrl = document.getElementById('backendUrl').value.trim();
  const timeoutSeconds = Number(document.getElementById('timeoutSeconds').value);
  const results = document.getElementById('results');
  const button = document.getElementById('askButton');

  if (!question) {{
    results.innerHTML = '<div class="status error">Please enter a question first.</div>';
    return;
  }}

  button.disabled = true;
  button.textContent = 'Analyzing…';
  results.innerHTML = '<div class="status info">Analyzing your course overlap question…</div>';

  try {{
    const response = await fetch('/api/analyze', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{
        backend_url: backendUrl,
        timeout_seconds: timeoutSeconds,
        question,
        mode: currentMode
      }})
    }});

    const payload = await response.json();
    if (!response.ok) {{
      throw new Error(payload.detail || 'Unknown frontend error.');
    }}

    metrics.totalRequests += 1;
    metrics.successfulRequests += 1;
    metrics.lastLatencyMs = payload.metrics.latency_ms;

    results.innerHTML = renderResponse(payload.upstream, payload.metrics);
  }} catch (error) {{
    metrics.totalRequests += 1;
    metrics.failedRequests += 1;
    results.innerHTML = `<div class="status error"><strong>Unable to complete the request.</strong>\n${{escapeHtml(error.message)}}</div>`;
  }} finally {{
    button.disabled = false;
    button.textContent = 'Ask →';
  }}
}}
</script>
</body>
</html>
"""


@app.post("/api/analyze")
async def api_analyze(request: AnalyzeRequest) -> JSONResponse:
    """Proxy browser requests to the selected course-overlap backend endpoint."""
    payload, latency_ms, status_code, endpoint = await post_to_backend(request)
    return JSONResponse(
        {
            "upstream": payload,
            "metrics": {
                "latency_ms": latency_ms,
                "status_code": status_code,
                "endpoint": endpoint,
                "mode": request.mode,
            },
        }
    )


@app.get("/api/health")
async def api_health(backend_url: str = DEFAULT_BACKEND_URL) -> JSONResponse:
    """Check whether the configured backend health endpoint is reachable."""
    url = f"{backend_url.rstrip('/')}/health"
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
        response.raise_for_status()
        return JSONResponse({"ok": True, "backend": response.json()})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            {"ok": False, "error": f"Could not reach backend health endpoint: {exc}"},
            status_code=502,
        )
