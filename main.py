import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

load_dotenv()

app = FastAPI()

client = openai.OpenAI(
    api_key=os.environ.get("CAMPUSAI_API_KEY"),
    base_url=os.environ.get("CAMPUSAI_API_URL"),
)

CHAT_MODEL = os.environ.get("CAMPUSAI_MODEL")
COURSES_PATH = os.environ.get("COURSES_PATH", "dtu_courses.jsonl")

# Supports normal DTU 5-digit codes, e.g. 02105, and alphanumeric codes, e.g. KU322.
COURSE_CODE_PATTERN = r"\b(?:\d{5}|[A-Za-z]{2}\d{3})\b"

with open(COURSES_PATH, "r", encoding="utf-8") as f:
    courses_data = [json.loads(line) for line in f]

courses_dict: Dict[str, Dict[str, Any]] = {
    str(c.get("course_code")).strip().upper(): c
    for c in courses_data
    if c.get("course_code")
}

def get_learning_objectives_text(course: Dict[str, Any]) -> str:
    """Return only the learning objectives as plain text."""
    learning_objectives = course.get("learning_objectives", []) or []
    if isinstance(learning_objectives, list):
        return "\n".join(str(item) for item in learning_objectives if str(item).strip())
    return str(learning_objectives)

def get_academic_prerequisites_text(course: Dict[str, Any]) -> str:
    """Return Academic prerequisites when present; otherwise an empty string."""
    fields = course.get("fields", {}) or {}
    return str(fields.get("Academic prerequisites", "") or "").strip()

def get_similarity_text(course: Dict[str, Any]) -> str:
    """
    Text used for similarity scoring.

    Similarity is based only on:
    - learning_objectives
    - fields['Academic prerequisites'] when that field exists
    """
    parts = [
        get_learning_objectives_text(course),
        get_academic_prerequisites_text(course),
    ]
    return "\n".join(part for part in parts if part.strip())

# The vectorizer is fitted once on all courses.
course_codes = list(courses_dict.keys())
similarity_documents = [get_similarity_text(courses_dict[code]) for code in course_codes]
course_code_to_index = {code: index for index, code in enumerate(course_codes)}
similarity_vectorizer = TfidfVectorizer(stop_words="english")
similarity_matrix = similarity_vectorizer.fit_transform(similarity_documents)

class Query(BaseModel):
    question: str

# Helper for normalize_course_codes
def normalize_course_code(code: Any) -> str:
    """Normalize user/model course-code output to match database keys."""
    return str(code).strip().upper()

def normalize_course_codes(codes: List[Any]) -> List[str]:
    """Keep only unique course codes while preserving order."""
    normalized: List[str] = []
    for code in codes:
        code = normalize_course_code(code)
        if code and code not in normalized:
            normalized.append(code)
    return normalized

# Used for extracting codes from the 'Not applicable with' field
def extract_course_codes_from_text(text: str) -> List[str]:
    """Find explicit course codes in free text."""
    return normalize_course_codes(re.findall(COURSE_CODE_PATTERN, text or ""))


def get_rag_context_text(course: Dict[str, Any]) -> str:
    """
    Text shown to the LLM for grounded comparison.
    """
    fields = course.get("fields", {}) or {}
    context = {
        "course_code": normalize_course_code(course.get("course_code", "")),
        "title": course.get("title") or course.get("course_title") or course.get("course_name") or "",
        "learning_objectives": course.get("learning_objectives", []) or [],
        "academic_prerequisites": fields.get("Academic prerequisites", "") or "",
        "not_applicable_together_with": fields.get("Not applicable together with", "") or "",
    }
    return json.dumps(context, ensure_ascii=False, indent=2)

# Retrieve exact course records by course code.
# This is used by /analyze-rag after the LLM has extracted course codes.
# The final explanation is grounded in the retrieved course records.
def retrieve_courses_by_codes(course_codes_to_retrieve: List[str]) -> List[Dict[str, Any]]:
    retrieved: List[Dict[str, Any]] = []
    for code in course_codes_to_retrieve:
        if code not in courses_dict:
            continue
        course = courses_dict[code]
        retrieved.append(
            {
                "course_code": code,
                "title": course.get("title") or course.get("course_title") or course.get("course_name") or "",
                "learning_objectives": course.get("learning_objectives", []) or [],
                "academic_prerequisites": get_academic_prerequisites_text(course),
                "fields": course.get("fields", {}) or {},
                "similarity_text": get_similarity_text(course),
                "context": get_rag_context_text(course),
            }
        )
    return retrieved

def format_courses_for_context(courses: List[Dict[str, Any]]) -> str:
    """Compress retrieved courses into context that can be passed to the LLM."""
    blocks = []
    for i, course in enumerate(courses, start=1):
        blocks.append(f"[Source {i}]\n{course['context']}")
    return "\n\n---\n\n".join(blocks)

def ask_llm_for_course_lists(question: str) -> Dict[str, List[str]]:
    """Original extraction approach used by /analyze."""
    sys_prompt = (
        "Extract the completed courses and the list of courses to compare from "
        "the following user query. Output ONLY a JSON object with two keys: "
        "'completed_courses' (list of course codes) and 'compared_courses' "
        "(list of course codes). Course codes can be 5 digits, such as 02105, "
        "or alphanumeric, such as KU322. For example: "
        "{\"completed_courses\": [\"12345\"], \"compared_courses\": [\"KU322\"]}"
    )

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
    except Exception:
        data = {"completed_courses": [], "compared_courses": []}

    return {
        "completed_courses": normalize_course_codes(data.get("completed_courses", []) or []),
        "compared_courses": normalize_course_codes(data.get("compared_courses", []) or []),
    }


def compute_similarity(course1_id: str, course2_id: str) -> float:
    """
    Compute overlap using catalogue-wide TF-IDF vectors.

    The TF-IDF vectorizer is fitted once on all courses at startup, using:
    - learning_objectives
    - fields['Academic prerequisites'] when that field exists
    """
    course1_id = normalize_course_code(course1_id)
    course2_id = normalize_course_code(course2_id)

    if course1_id not in course_code_to_index or course2_id not in course_code_to_index:
        return 0.0

    idx1 = course_code_to_index[course1_id]
    idx2 = course_code_to_index[course2_id]

    return float(
        cosine_similarity(
            similarity_matrix[idx1:idx1 + 1],
            similarity_matrix[idx2:idx2 + 1],
        )[0][0]
    )

def validate_courses(completed_courses: List[str], compared_courses: List[str]) -> Optional[Dict[str, Any]]:
    """Check if the courses are in the database."""
    invalid_courses = []
    for course_code in completed_courses + compared_courses:
        if course_code and course_code not in courses_dict:
            invalid_courses.append(course_code)

    if invalid_courses:
        courses_str = ", ".join(invalid_courses)
        course_word = "Course" if len(invalid_courses) == 1 else "Courses"
        verb = "is" if len(invalid_courses) == 1 else "are"
        return {"error": f"{course_word} {courses_str} {verb} not in the database!"}

    return None


def parse_conflicting_codes(not_applicable: str) -> List[str]:
    """Parse DTU's 'Not applicable together with' field robustly."""
    return extract_course_codes_from_text(not_applicable or "")


def check_not_applicable_conflicts(completed_courses: List[str], compared_courses: List[str]) -> Optional[Dict[str, Any]]:
    """Check completed courses against compared courses."""
    for completed_course in completed_courses:
        fields = courses_dict.get(completed_course, {}).get("fields", {}) or {}
        conflicting_codes = parse_conflicting_codes(fields.get("Not applicable together with", ""))

        for compared_course in compared_courses:
            if compared_course in conflicting_codes:
                return {
                    "error": f"The course {compared_course} is not applicable with course {completed_course} (as mentioned in the course description)."
                }
    # Both directions
    for compared_course in compared_courses:
        fields = courses_dict.get(compared_course, {}).get("fields", {}) or {}
        conflicting_codes = parse_conflicting_codes(fields.get("Not applicable together with", ""))

        for completed_course in completed_courses:
            if completed_course in conflicting_codes:
                return {
                    "error": f"The course {compared_course} is not applicable with course {completed_course} (as mentioned in the course description)."
                }

    return None


def recommendation_from_similarity(similarity: float) -> str:
    """Get recommendation text based on score."""
    rec = "Low overlap — safe to take alongside."
    if similarity > 0.6:
        rec = "High overlap — consider skipping or auditing only."
    elif similarity > 0.3:
        rec = "Moderate overlap — complementary but some shared content."
    return rec


def overlap_level_from_similarity(similarity: float) -> str:
    """Get overlap level based on score."""
    if similarity > 0.6:
        return "high"
    if similarity > 0.3:
        return "moderate"
    return "low"


def build_ranking(completed_courses: List[str], compared_courses: List[str]) -> List[Dict[str, Any]]:
    '''Build a descending sorted ranking of compared courses.'''
    results: List[Dict[str, Any]] = []
    base_course = completed_courses[0] if completed_courses else ""

    for c_code in compared_courses:
        sim = compute_similarity(base_course, c_code)
        results.append(
            {
                "course_number": c_code,
                "similarity": round(float(sim), 2),
                "overlap_level": overlap_level_from_similarity(sim),
                "recommendation": recommendation_from_similarity(sim),
                "similarity_basis": "learning_objectives + Academic prerequisites",
            }
        )

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results


def ask_llm_for_rag_comparison(
    question: str,
    completed_courses: List[str],
    compared_courses: List[str],
    retrieved_courses: List[Dict[str, Any]],
    ranking: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    RAG generation step for the actual comparison.

    The LLM receives retrieved source context and the similarity
    scores. It should explain the comparison based on learning objectives and
    Academic prerequisites, not unrelated fields.
    """
    context = format_courses_for_context(retrieved_courses)
    sys_prompt = """
You are a course advisor using Retrieval-Augmented Generation.
Use ONLY the retrieved course sources and the provided similarity scores.

Comparison rules:
- Compare courses based on learning objectives.
- Also include Academic prerequisites when they exist.
- Do not use unrelated fields to justify similarity.
- Do not invent course details.
- Treat the provided similarity score as the numeric ranking score.
- Return ONLY valid JSON with keys: answer, ranking.
- ranking must be a list. Each item must include: course_number, similarity, overlap_level, recommendation, evidence.
- evidence must be a short list of facts from the retrieved learning objectives and/or Academic prerequisites.
""".strip()

    user_prompt = f"""
User question:
{question}

Completed courses:
{json.dumps(completed_courses, ensure_ascii=False)}

Compared courses:
{json.dumps(compared_courses, ensure_ascii=False)}

Similarity ranking computed from learning objectives + Academic prerequisites:
{json.dumps(ranking, ensure_ascii=False, indent=2)}

Retrieved course sources:
{context}
""".strip()

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
    except Exception as exc:
        return {
            "answer": (
                "RAG comparison generation failed, so this response uses the "
                "similarity ranking computed from learning objectives and Academic prerequisites. "
                f"Error: {type(exc).__name__}"
            ),
            "ranking": ranking,
        }

    # Keep the similarity score as source of truth even if the model changes it.
    ranking_by_code = {item["course_number"]: item for item in ranking}
    model_ranking = parsed.get("ranking", []) if isinstance(parsed.get("ranking", []), list) else []
    final_ranking: List[Dict[str, Any]] = []

    for model_item in model_ranking:
        code = normalize_course_code(model_item.get("course_number", ""))
        if code not in ranking_by_code:
            continue
        base_item = dict(ranking_by_code[code])
        base_item["evidence"] = model_item.get("evidence", []) or []
        # Keep similarity scroe unless the model provides a reply.
        if isinstance(model_item.get("recommendation"), str) and model_item.get("recommendation").strip():
            base_item["rag_recommendation"] = model_item["recommendation"]
        final_ranking.append(base_item)

    final_ranking.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "answer": parsed.get("answer") or "Comparison completed using retrieved course context.",
        "ranking": final_ranking,
    }


def analyze_courses(
    completed_courses: List[str],
    compared_courses: List[str]
) -> Dict[str, Any]:

    validation_error = validate_courses(completed_courses, compared_courses)
    if validation_error:
        return validation_error

    conflict_error = check_not_applicable_conflicts(completed_courses, compared_courses)
    if conflict_error:
        return conflict_error

    response = {
        "completed_courses": completed_courses,
        "compared_courses": compared_courses,
        "ranking": build_ranking(completed_courses, compared_courses),
    }
    
    return response


@app.post("/analyze")
def analyze_endpoint(query: Query):
    extracted = ask_llm_for_course_lists(query.question)
    # In case the LLM did not identify any courses, return error message.
    if not extracted["completed_courses"] and not extracted["compared_courses"]:
        return {
            "error": f"The assistant could not identify the courses."
        }
    return analyze_courses(
        extracted["completed_courses"],
        extracted["compared_courses"],
    )


@app.post("/analyze-rag")
def analyze_rag_endpoint(query: Query):
    """
    RAG version of /analyze.

    1. The LLM extracts completed and compared course codes from the question.
    2. Course codes are normalized and validated against the course database.
    3. The API checks conflicts from the `Not applicable together with` course field.
    4. A TF-IDF similarity ranking is computed from learning objectives
       and academic prerequisites.
    5. The exact extracted course records are retrieved from the course database.
    6. The LLM explains the ranking using only those retrieved records.
    """
    extracted = ask_llm_for_course_lists(query.question)

    completed_courses = extracted["completed_courses"]
    compared_courses = extracted["compared_courses"]

    # If the LLM did not identify any courses, avoid a second LLM call and return
    # the same empty structure expected by the frontend/tests.
    if not completed_courses and not compared_courses:
        return {
            "error": f"The assistant could not identify the courses."
        }

    validation_error = validate_courses(completed_courses, compared_courses)
    if validation_error:
        return validation_error

    conflict_error = check_not_applicable_conflicts(completed_courses, compared_courses)
    if conflict_error:
        return conflict_error

    ranking = build_ranking(completed_courses, compared_courses)

    # This is the retrieval step used for RAG generation: retrieve the exact
    # records that the LLM identified, then ground the explanation in those records.
    retrieved_courses = retrieve_courses_by_codes(completed_courses + compared_courses)

    rag_comparison = ask_llm_for_rag_comparison(
        question=query.question,
        completed_courses=completed_courses,
        compared_courses=compared_courses,
        retrieved_courses=retrieved_courses,
        ranking=ranking,
    )

    return {
        "answer": rag_comparison["answer"],
        "completed_courses": completed_courses,
        "compared_courses": compared_courses,
        "ranking": rag_comparison["ranking"],
        "sources": [
            {
                "source_id": i + 1,
                "course_code": course["course_code"],
                "title": course.get("title", ""),
                "has_academic_prerequisites": bool(course.get("academic_prerequisites")),
            }
            for i, course in enumerate(retrieved_courses)
        ],
        "rag": {
            "retrieved_course_codes": [course["course_code"] for course in retrieved_courses],
            "retrieval_method": "Exact course-code retrieval after LLM extraction",
            "similarity_basis": "learning_objectives + Academic prerequisites",
            "similarity_ranking": ranking,
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "course_count": len(courses_dict),
        "chat_model": CHAT_MODEL,
        "similarity_basis": "learning_objectives + Academic prerequisites",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001)
