My idea for the project came from my own struggle as a DTU Master's student: when creating my study plan, I have a hard time finding out how much overlap there is between various courses that I want to take. This led me to decide to implement a Course Overlap Analyzer.

I would like to build a Retrieval-Augmented Generation (RAG) system over the DTU course data in a dockerized Web service. As input I would provide a course name or number and receive as output a list of courses that have an overlap with the provided one, with each item in the list having an approximate percentage of overlap.

Some ways to extend the initial scope of the project since this could be too small for the 4 week period are:
- Add also an LLM-generated explanation on the comparison between two specific courses
- Add a natural language interface that accepts a plain text question. E.g:

Input:
{
  "question": "I have already taken 02450. Which of these courses would overlap the most: 02456, 42186, 02477?"
}

Output:
{
  "completed_courses": ["02450"],
  "compared_courses": ["02456", "42186", "02477"],
  "ranking": [
    {"course_number": "02456", "similarity": 0.74, "recommendation": "High overlap — consider skipping or auditing only."},
    {"course_number": "42186", "similarity": 0.51, "recommendation": "Moderate overlap — complementary but some shared content."},
    {"course_number": "02477", "similarity": 0.29, "recommendation": "Low overlap — safe to take alongside 02450."}
  ]
}