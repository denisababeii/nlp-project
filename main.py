import json
import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai

load_dotenv()

app = FastAPI()

client = openai.OpenAI(
    api_key=os.environ.get("CAMPUSAI_API_KEY"),
    base_url=os.environ.get("CAMPUSAI_API_URL"),
)

with open("dtu_courses.jsonl", "r") as f:
    courses_data = [json.loads(line) for line in f]

courses_dict = {
    c.get("course_code"): c 
    for c in courses_data
}

class Query(BaseModel):
    question: str

def compute_similarity(course1_id, course2_id):
    if course1_id not in courses_dict or course2_id not in courses_dict:
        return 0.0
    
    text1 = json.dumps(courses_dict[course1_id].get("learning_objectives", [])) + " " + json.dumps(courses_dict[course1_id].get("fields", {}))
    text2 = json.dumps(courses_dict[course2_id].get("learning_objectives", [])) + " " + json.dumps(courses_dict[course2_id].get("fields", {}))
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]

@app.post("/rag")
def rag_endpoint(query: Query):
    sys_prompt = "Extract the completed courses and the list of courses to compare from the following user query. Output ONLY a JSON object with two keys: 'completed_courses' (list of course codes) and 'compared_courses' (list of course codes). For example: {\"completed_courses\": [\"12345\"], \"compared_courses\": [\"67890\"]}"
    
    response = client.chat.completions.create(
        model=os.environ.get("CAMPUSAI_MODEL"),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query.question}
        ],
        response_format={"type": "json_object"}
    )
    
    try:
        data = json.loads(response.choices[0].message.content)
    except Exception:
        data = {"completed_courses": [], "compared_courses": []}
    
    completed_courses = data.get("completed_courses", [])
    compared_courses = data.get("compared_courses", [])

    # Validate that all courses exist in the database
    all_courses = completed_courses + compared_courses
    invalid_courses = []
    for course_code in all_courses:
        if course_code and course_code not in courses_dict:
            invalid_courses.append(course_code)
    
    if invalid_courses:
        courses_str = ", ".join(invalid_courses)
        course_word = "Course" if len(invalid_courses) == 1 else "Courses"
        verb = "is" if len(invalid_courses) == 1 else "are"
        return {"error": f"{course_word} {courses_str} {verb} not in the database!"}
    
    # Check for "Not applicable together with" conflicts (bidirectional)
    # Check if completed courses have conflicts with compared courses
    for completed_course in completed_courses:
        if completed_course in courses_dict:
            fields = courses_dict[completed_course].get("fields", {})
            not_applicable = fields.get("Not applicable together with", "")
            
            if not_applicable:
                # Parse the not_applicable string (split by both . and /)
                conflicting_codes = [code.strip() for code in not_applicable.replace("/", ".").split(".")]
                
                for compared_course in compared_courses:
                    if compared_course in conflicting_codes:
                        return {"error": f"The course {compared_course} is not applicable with course {completed_course} (as mentioned in the course description)."}
    
    # Check if compared courses have conflicts with completed courses
    for compared_course in compared_courses:
        if compared_course in courses_dict:
            fields = courses_dict[compared_course].get("fields", {})
            not_applicable = fields.get("Not applicable together with", "")
            
            if not_applicable:
                # Parse the not_applicable string (split by both . and /)
                conflicting_codes = [code.strip() for code in not_applicable.replace("/", ".").split(".")]
                
                for completed_course in completed_courses:
                    if completed_course in conflicting_codes:
                        return {"error": f"The course {compared_course} is not applicable with course {completed_course} (as mentioned in the course description)."}
    
    results = []
    base_course = completed_courses[0] if completed_courses else ""
    
    for c_code in compared_courses:
        sim = compute_similarity(base_course, c_code)
        
        rec = "Low overlap — safe to take alongside."
        if sim > 0.6: rec = "High overlap — consider skipping or auditing only."
        elif sim > 0.3: rec = "Moderate overlap — complementary but some shared content."
            
        results.append({
            "course_number": c_code,
            "similarity": round(float(sim), 2),
            "recommendation": rec
        })
        
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    return {
        "completed_courses": completed_courses,
        "compared_courses": compared_courses,
        "ranking": results
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001)