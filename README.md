# Course Overlap Assistant

A FastAPI-based course overlap assistant that helps students check whether courses overlap in content. The app compares course learning objectives and academic prerequisites, and provides ranked recommendations through either a simple analysis endpoint or a RAG-enhanced endpoint.

The project includes:

- a backend API in `main.py`
- a lightweight FastAPI frontend in `ui.py`
- Docker support through `Dockerfile` and `docker-compose.yml`
- tests for the simple and RAG endpoints
- course data loaded from a JSONL file such as `dtu_courses.jsonl`

---

## 1. Prerequisites

You need:

- Docker and Docker Compose
- a CampusAI API key
- the course catalogue JSONL file, for example `dtu_courses.jsonl`
- a `.env` file based on `.env_template`

For local development without Docker, you also need:

- Python 3.13+
- `uv`

---

## 2. Environment setup

Copy the environment template:

```bash
cp .env_template .env
```

Open `.env` and add your own CampusAI credentials.

Example:

```env
CAMPUSAI_API_KEY=your-campusai-api-key-here
CAMPUSAI_API_URL=https://chat.campusai.compute.dtu.dk/api/v1
CAMPUSAI_MODEL=Gemma 3 (Chat)
COURSES_PATH=dtu_courses.jsonl
```

Optional variables:

```env
COURSE_OVERLAP_BACKEND_URL=http://127.0.0.1:8001
COURSE_OVERLAP_TIMEOUT_SECONDS=60
```

`COURSE_OVERLAP_BACKEND_URL` is used by the frontend proxy. In the provided Docker setup, the default `http://127.0.0.1:8001` works because the frontend and backend run inside the same container.

---

## 3. Run with Docker

Build and start the project:

```bash
docker compose up --build
```

This starts:

- backend API on `http://127.0.0.1:8001`
- frontend UI on `http://127.0.0.1:8000`

Open the UI in your browser:

```text
http://127.0.0.1:8000
```

Stop the container:

```bash
docker compose down
```

Run in detached mode:

```bash
docker compose up --build -d
```

View logs:

```bash
docker compose logs -f
```

---

## 4. Run locally without Docker

Install dependencies:

```bash
uv sync
```

Start the backend:

```bash
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8001
```

In another terminal, start the frontend:

```bash
uv run uvicorn ui:app --reload --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

---

## 5. Frontend

The frontend is intentionally simple and dependency-light. It is implemented as a FastAPI app that serves a single HTML page with inline CSS and vanilla JavaScript.

The UI has two modes:

| Mode | Backend endpoint | Description |
|---|---|---|
| Simple | `POST /analyze` | Extracts course codes from the question and returns a deterministic overlap ranking. |
| Advanced (RAG) | `POST /analyze-rag` | Retrieves relevant course context first, then uses the LLM to produce a more explanatory answer with evidence. |

The frontend itself exposes:

### `GET /`

Serves the browser interface.

### `POST /api/analyze`

Frontend proxy endpoint. The browser sends requests here, and the frontend forwards them to either `/analyze` or `/analyze-rag` on the backend depending on the selected mode.

Example request:

```json
{
  "question": "I have completed 01002. Should I take 01003 or 01004?",
  "mode": "advanced",
  "backend_url": "http://127.0.0.1:8001",
  "timeout_seconds": 60
}
```

### `GET /api/health`

Checks whether the backend health endpoint is reachable.

---

## 6. Backend API

The backend expects JSON requests with a single `question` field:

```json
{
  "question": "I have completed 01002. Should I take 01003 or 01004?"
}
```

### `POST /analyze`

Simple overlap analysis.

Under the hood:

1. An LLM extracts two lists from the user question:
   - `completed_courses`
   - `compared_courses`
2. Course codes are normalized.
3. The API validates that all course codes exist in the course database.
4. It checks hard conflicts from the `Not applicable together with` course field.
5. It computes TF-IDF cosine similarity between the completed course and each compared course.
6. It returns a ranked list sorted by similarity.

Similarity is calculated using only:

- `learning_objectives`
- `fields["Academic prerequisites"]`, when available

Example response:

```json
{
  "completed_courses": ["01002"],
  "compared_courses": ["01003", "01004"],
  "ranking": [
    {
      "course_number": "01003",
      "similarity": 0.42,
      "overlap_level": "moderate",
      "recommendation": "Moderate overlap — complementary but some shared content.",
      "similarity_basis": "learning_objectives + Academic prerequisites"
    }
  ]
}
```

### `POST /analyze-rag`

RAG-enhanced overlap analysis.

Under the hood:

1. The API retrieves relevant courses using TF-IDF over learning objectives and academic prerequisites.
2. The LLM extracts completed and compared course codes using only the retrieved catalogue context.
3. Mentioned or extracted course codes are force-included in a second retrieval pass.
4. The deterministic TF-IDF similarity ranking is computed.
5. The LLM receives:
   - retrieved course context
   - completed courses
   - compared courses
   - deterministic similarity scores
6. The LLM generates an explanatory answer and evidence while keeping the deterministic similarity score as the source of truth.

Example response fields:

```json
{
  "answer": "Course 01003 has moderate overlap with 01002...",
  "completed_courses": ["01002"],
  "compared_courses": ["01003"],
  "ranking": [
    {
      "course_number": "01003",
      "similarity": 0.42,
      "overlap_level": "moderate",
      "recommendation": "Moderate overlap — complementary but some shared content.",
      "evidence": [
        "Both courses include related learning objectives..."
      ]
    }
  ],
  "sources": [],
  "rag": {
    "retrieved_course_codes": ["01002", "01003"],
    "retrieval_method": "TF-IDF over learning objectives + Academic prerequisites",
    "similarity_basis": "learning_objectives + Academic prerequisites"
  }
}
```

### `GET /health`

Health check endpoint.

Example response:

```json
{
  "status": "ok",
  "course_count": 1234,
  "chat_model": "Gemma 3 (Chat)",
  "similarity_basis": "learning_objectives + Academic prerequisites"
}
```

---

## 7. How recommendations are decided

The ranking score is a cosine similarity value between `0.0` and `1.0`.

| Similarity | Overlap level | Recommendation |
|---:|---|---|
| `> 0.6` | High | Consider skipping or auditing only. |
| `> 0.3` | Moderate | Complementary, but with some shared content. |
| `<= 0.3` | Low | Safe to take alongside. |

The score is rounded in the API response.

---

## 8. Course data

The backend loads course data from the path configured by `COURSES_PATH`.

Default:

```env
COURSES_PATH=dtu_courses.jsonl
```

Each line should be a JSON object. The important fields are:

```json
{
  "course_code": "01002",
  "title": "Example Course",
  "learning_objectives": [
    "Explain ...",
    "Apply ..."
  ],
  "fields": {
    "Academic prerequisites": "...",
    "Not applicable together with": "01003"
  }
}
```

Course codes are normalized to uppercase strings. The code supports normal DTU five-digit course codes and alphanumeric course codes such as `KU322`.

---

## 9. Run tests

Run the full test suite:

```bash
uv run pytest -v
```

Run only the simple `/analyze` endpoint tests:

```bash
uv run pytest test_analyze_endpoint.py -v
```

Run only the advanced `/analyze-rag` endpoint tests:

```bash
uv run pytest test_rag_endpoint.py -v
```

Run a single test class:

```bash
uv run pytest test_analyze_endpoint.py::TestRagEndpoint -v
```

Run a single test:

```bash
uv run pytest test_rag_endpoint.py::TestAnalyzeRagEndpoint::test_rag_endpoint_basic_query -v
```

The tests check, among other things:

- similarity scores are between `0.0` and `1.0`
- identical courses have high similarity
- invalid courses return an error
- rankings are sorted by descending similarity
- recommendation thresholds behave as expected
- `/analyze-rag` returns RAG-specific fields such as `answer`, `sources`, and `rag`

If tests fail because of environment variables, check that `.env` exists and contains your CampusAI configuration. If tests fail because course data is missing, check that `COURSES_PATH` points to the correct JSONL file.

---

## 10. Example requests with curl

Simple endpoint:

```bash
curl -X POST http://127.0.0.1:8001/analyze \
  -H "Content-Type: application/json" \
  -d '{"question": "I completed 01002. Compare 01003 and 01018."}'
```

Advanced RAG endpoint:

```bash
curl -X POST http://127.0.0.1:8001/analyze-rag \
  -H "Content-Type: application/json" \
  -d '{"question": "I completed 01002. Compare 01003 and 01018."}'
```

Health check:

```bash
curl http://127.0.0.1:8001/health
```

---

## 11. Troubleshooting

### The frontend says it cannot reach the backend

Check that the backend is running on port `8001`.

```bash
curl http://127.0.0.1:8001/health
```

### The API returns empty course lists

Make sure your question contains recognizable course codes, for example:

```text
I completed 01002. Should I take 01003?
```

### A course is reported as not in the database

Check that the course exists in the JSONL file configured by `COURSES_PATH`.

### The LLM extraction fails

Check:

- `CAMPUSAI_API_KEY`
- `CAMPUSAI_API_URL`
- `CAMPUSAI_MODEL`

### Docker starts but the UI does not load

Check logs:

```bash
docker compose logs -f
```

Make sure ports `8000` and `8001` are not already in use.

---

## 12. Project structure

```text
.
├── main.py                    # Backend API
├── ui.py                      # FastAPI frontend
├── dtu_courses.jsonl          # Course catalogue data
├── .env_template              # Environment variable template
├── .env                       # Local secrets; should not be committed
├── Dockerfile                 # Docker image
├── docker-compose.yml         # Docker Compose setup
├── pyproject.toml             # Dependencies and project metadata
├── uv.lock                    # Locked dependency versions
├── test_analyze_endpoint.py   # Tests for /analyze
└── test_rag_endpoint.py       # Tests for /analyze-rag
```