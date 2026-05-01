FROM python:3.13-slim

WORKDIR /
COPY pyproject.toml uv.lock ./

RUN pip install uv && uv sync --frozen

COPY . .

EXPOSE 8000 8001

CMD ["sh", "-c", "uv run uvicorn main:app --host 0.0.0.0 --port 8001 & uv run uvicorn ui:app --host 0.0.0.0 --port 8000"]
