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

class Query(BaseModel):
    question: str

## HELPERS
# Normalize user/model course code output to match database keys.
def normalize_course_code(code):
    return str(code).strip().upper()

# Keep only unique course codes and normalize them to match keys.
def normalize_course_codes(codes: List[Any]) -> List[str]:
    normalized: List[str] = []
    for code in codes:
        code = normalize_course_code(code)
        if code and code not in normalized:
            normalized.append(code)
    return normalized

# Find course codes matching the Regex COURSE_CODE_PATTERN.
def extract_course_codes_from_text(text: str) -> List[str]:
    return normalize_course_codes(re.findall(COURSE_CODE_PATTERN, text or ""))

# Return only the learning objectives as plain text.
def get_learning_objectives_text(course: Dict[str, Any]) -> str:
    learning_objectives = course.get("learning_objectives", []) or []
    if isinstance(learning_objectives, list):
        return "\n".join(str(item) for item in learning_objectives if str(item).strip())
    return str(learning_objectives)

# Return academic prerequisites when present; otherwise an empty string.
def get_academic_prerequisites_text(course: Dict[str, Any]) -> str:
    fields = course.get("fields", {}) or {}
    return str(fields.get("Academic prerequisites", "") or "").strip()


# Text used for similarity and RAG retrieval.
# Similarity is based on:
# - learning_objectives
# - fields['Academic prerequisites'] when that field exists
def get_similarity_text(course: Dict[str, Any]) -> str:
    parts = [
        get_learning_objectives_text(course),
        get_academic_prerequisites_text(course),
    ]
    return "\n".join(part for part in parts if part.strip())

# Text shown to the LLM for comparison.
# The comparison is centered on the same fields used for
# similarity: learning objectives and academic prerequisites. A small amount of
# metadata is included so the model can identify the course and mention
# conflicts when relevant.
def get_rag_context_text(course: Dict[str, Any]) -> str:
    fields = course.get("fields", {}) or {}
    context = {
        "course_code": normalize_course_code(course.get("course_code", "")),
        "title": course.get("title") or course.get("course_title") or course.get("course_name") or "",
        "learning_objectives": course.get("learning_objectives", []) or [],
        "academic_prerequisites": fields.get("Academic prerequisites", "") or "",
        "not_applicable_together_with": fields.get("Not applicable together with", "") or "",
    }
    return json.dumps(context, ensure_ascii=False, indent=2)

# -----------------------------------------------------------------------------
# RAG index: use learning objectives + Academic prerequisites for similarity.
# -----------------------------------------------------------------------------
course_codes = list(courses_dict.keys())
similarity_documents = [get_similarity_text(courses_dict[code]) for code in course_codes]

# TfidfVectorizer fails if every document is empty. This fallback should rarely be
# needed, but keeps the API from crashing on incomplete data.
if any(doc.strip() for doc in similarity_documents):
    rag_vectorizer = TfidfVectorizer(stop_words="english")
    rag_matrix = rag_vectorizer.fit_transform(similarity_documents)
else:
    rag_vectorizer = TfidfVectorizer(stop_words=None)
    rag_matrix = rag_vectorizer.fit_transform([code for code in course_codes])


# Text shown to the LLM for comparison.
# The comparison is centered on the same fields used for
# similarity: learning objectives and academic prerequisites. A small amount of
# metadata is included so the model can identify the course and mention
# conflicts when relevant.
def retrieve_relevant_courses(question: str, top_k: int = 12, required_course_codes: Optional[List[str]] = None) -> List[Dict[str, Any]]:

    required = normalize_course_codes(required_course_codes or [])
    exact_codes = [code for code in extract_course_codes_from_text(question) if code in courses_dict]

    query_vector = rag_vectorizer.transform([question])
    scores = cosine_similarity(query_vector, rag_matrix).flatten()
    ranked_indexes = scores.argsort()[::-1]

    selected_codes: List[str] = []
    for code in exact_codes + required:
        if code in courses_dict and code not in selected_codes:
            selected_codes.append(code)

    for idx in ranked_indexes:
        code = course_codes[int(idx)]
        if code not in selected_codes:
            selected_codes.append(code)
        if len(selected_codes) >= top_k:
            break

    retrieved: List[Dict[str, Any]] = []
    for code in selected_codes:
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

# Compress retrieved courses into context that can be passed to the LLM.
def format_courses_for_context(courses: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, course in enumerate(courses, start=1):
        blocks.append(f"[Source {i}]\n{course['context']}")
    return "\n\n---\n\n".join(blocks)

# Simple approach used by /analyze.
def ask_llm_for_course_lists(question: str) -> Dict[str, List[str]]:
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

# RAG extraction: retrieve relevant catalogue entries, then ask the LLM.
def ask_llm_for_course_lists_with_rag(question: str, retrieved_courses: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    context = format_courses_for_context(retrieved_courses)
    sys_prompt = (
        "You extract course codes from a user's question using ONLY the retrieved "
        "course catalogue context. Output ONLY a JSON object with two keys: "
        "'completed_courses' and 'compared_courses'. Both values must be lists "
        "of valid course codes from the context. Course codes can be 5 digits, "
        "such as 02105, or alphanumeric, such as KU322. Do not invent course codes. "
        "If the user clearly says they have already taken, passed, completed, or "
        "studied a course, place it in completed_courses. If the user asks whether "
        "they should take, compare, skip, or evaluate courses, place those in "
        "compared_courses."
    )

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Retrieved course context:\n{context}\n\nUser question:\n{question}"},
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

# Compute overlap using learning objectives and academic prerequisites.
def compute_similarity(course1_id: str, course2_id: str) -> float:
    course1_id = normalize_course_code(course1_id)
    course2_id = normalize_course_code(course2_id)

    if course1_id not in courses_dict or course2_id not in courses_dict:
        return 0.0

    text1 = get_similarity_text(courses_dict[course1_id])
    text2 = get_similarity_text(courses_dict[course2_id])

    if not text1.strip() or not text2.strip():
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])

# Check if provided courses are present in the database
def validate_courses(completed_courses: List[str], compared_courses: List[str]) -> Optional[Dict[str, Any]]:
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

# Parse DTU's 'Not applicable together with' field.
def parse_conflicting_codes(not_applicable: str) -> List[str]:
    return extract_course_codes_from_text(not_applicable or "")

# Check completed courses against compared courses.
def check_not_applicable_conflicts(completed_courses: List[str], compared_courses: List[str]) -> Optional[Dict[str, Any]]:
    for completed_course in completed_courses:
        fields = courses_dict.get(completed_course, {}).get("fields", {}) or {}
        conflicting_codes = parse_conflicting_codes(fields.get("Not applicable together with", ""))

        for compared_course in compared_courses:
            if compared_course in conflicting_codes:
                return {
                    "error": f"The course {compared_course} is not applicable with course {completed_course} (as mentioned in the course description)."
                }

    # Check compared courses against completed courses.
    for compared_course in compared_courses:
        fields = courses_dict.get(compared_course, {}).get("fields", {}) or {}
        conflicting_codes = parse_conflicting_codes(fields.get("Not applicable together with", ""))

        for completed_course in completed_courses:
            if completed_course in conflicting_codes:
                return {
                    "error": f"The course {compared_course} is not applicable with course {completed_course} (as mentioned in the course description)."
                }

    return None

# Retrun recommendation text form the computed similarity.
def recommendation_from_similarity(similarity: float) -> str:
    rec = "Low overlap — safe to take alongside."
    if similarity > 0.6:
        rec = "High overlap — consider skipping or auditing only."
    elif similarity > 0.3:
        rec = "Moderate overlap — complementary but some shared content."
    return rec

# Retrun overlap level form the computed similarity.
def overlap_level_from_similarity(similarity: float) -> str:
    if similarity > 0.6:
        return "high"
    if similarity > 0.3:
        return "moderate"
    return "low"

# Compute ranking of courses to compare
def build_ranking(completed_courses: List[str], compared_courses: List[str]) -> List[Dict[str, Any]]:
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


# RAG generation step for the comparison.
# The LLM receives the source context and the deterministic similarity
# scores. It should explain the comparison based on learning objectives and
# academic prerequisites, not unrelated fields.
def ask_llm_for_rag_comparison(
    question: str,
    completed_courses: List[str],
    compared_courses: List[str],
    retrieved_courses: List[Dict[str, Any]],
    deterministic_ranking: List[Dict[str, Any]],
) -> Dict[str, Any]:
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
{json.dumps(deterministic_ranking, ensure_ascii=False, indent=2)}

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
                "RAG comparison generation failed, so this response uses the deterministic "
                "similarity ranking computed from learning objectives and Academic prerequisites. "
                f"Error: {type(exc).__name__}"
            ),
            "ranking": deterministic_ranking,
        }

    # Keep the deterministic score as source of truth even if the model changes it.
    ranking_by_code = {item["course_number"]: item for item in deterministic_ranking}
    model_ranking = parsed.get("ranking", []) if isinstance(parsed.get("ranking", []), list) else []
    final_ranking: List[Dict[str, Any]] = []

    for model_item in model_ranking:
        code = normalize_course_code(model_item.get("course_number", ""))
        if code not in ranking_by_code:
            continue
        base_item = dict(ranking_by_code[code])
        base_item["evidence"] = model_item.get("evidence", []) or []
        # Keep deterministic recommendation unless the model provides a string and you prefer to expose it.
        if isinstance(model_item.get("recommendation"), str) and model_item.get("recommendation").strip():
            base_item["rag_recommendation"] = model_item["recommendation"]
        final_ranking.append(base_item)

    # Ensure no compared course disappears if the model omits it.
    included = {item["course_number"] for item in final_ranking}
    for item in deterministic_ranking:
        if item["course_number"] not in included:
            final_ranking.append(item)

    final_ranking.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "answer": parsed.get("answer") or "Comparison completed using retrieved course context.",
        "ranking": final_ranking,
    }

# Analyze courses for the simple endpoint
def analyze_courses(
    completed_courses: List[str],
    compared_courses: List[str],
    extra_response: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    completed_courses = normalize_course_codes(completed_courses)
    compared_courses = normalize_course_codes(compared_courses)

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

    if extra_response:
        response.update(extra_response)

    return response

# Simple overlap analysis.
# 1. An LLM extracts two lists from the user question:
#    - `completed_courses`
#    - `compared_courses`
# 2. Course codes are normalized.
# 3. The API validates that all course codes exist in the course database.
# 4. It checks conflicts from the `Not applicable together with` course field.
# 5. It computes TF-IDF cosine similarity between the completed course and each compared course.
# 6. It returns a ranked list sorted by similarity.

# Similarity is calculated using only:

# - `learning_objectives`
# - `fields["Academic prerequisites"]`, when available
@app.post("/analyze")
def analyze_endpoint(query: Query):
    extracted = ask_llm_for_course_lists(query.question)
    return analyze_courses(
        extracted["completed_courses"],
        extracted["compared_courses"],
    )

# RAG version of /analyze.
# 1. Retrieve relevant course records using learning objectives + academic prerequisites.
# 2. Extract the completed and compared course codes with retrieved context.
# 3. Force-include the mentioned courses in a second retrieval pass.
# 4. Compute similarity from learning objectives + academic prerequisites.
# 5. Ask the LLM to explain the comparison using only retrieved RAG context.
@app.post("/analyze-rag")
def analyze_rag_endpoint(query: Query):
    initial_retrieved = retrieve_relevant_courses(query.question, top_k=12)
    extracted = ask_llm_for_course_lists_with_rag(query.question, initial_retrieved)

    completed_courses = normalize_course_codes(extracted["completed_courses"])
    compared_courses = normalize_course_codes(extracted["compared_courses"])

    validation_error = validate_courses(completed_courses, compared_courses)
    if validation_error:
        return validation_error

    conflict_error = check_not_applicable_conflicts(completed_courses, compared_courses)
    if conflict_error:
        return conflict_error

    mentioned_codes = completed_courses + compared_courses
    retrieved_courses = retrieve_relevant_courses(
        query.question,
        top_k=12,
        required_course_codes=mentioned_codes,
    )

    deterministic_ranking = build_ranking(completed_courses, compared_courses)
    rag_comparison = ask_llm_for_rag_comparison(
        question=query.question,
        completed_courses=completed_courses,
        compared_courses=compared_courses,
        retrieved_courses=retrieved_courses,
        deterministic_ranking=deterministic_ranking,
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
            "retrieval_method": "TF-IDF over learning objectives + Academic prerequisites",
            "similarity_basis": "learning_objectives + Academic prerequisites",
            "deterministic_similarity_ranking": deterministic_ranking,
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
    uvicorn.run("main_with_rag:app", host="0.0.0.0", port=8001)
