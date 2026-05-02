import pytest
import json
from fastapi.testclient import TestClient
from main import app, compute_similarity, courses_dict

client = TestClient(app)

# Tests for the analyze endpoint. Run with 'uv run pytest test_analyze_endpoint.py -v'.

class TestSimilarityComputation:
    """Test the compute_similarity function."""
    
    def test_similarity_same_course(self):
        """Test similarity of a course with itself."""
        # Get any course code from the dictionary
        course_code = list(courses_dict.keys())[0]
        sim = compute_similarity(course_code, course_code)
        assert round(sim) == 1, "Same course should have 100% similarity"
    
    def test_similarity_different_courses(self):
        """Test similarity between different courses."""
        course_codes = list(courses_dict.keys())
        if len(course_codes) >= 2:
            sim = compute_similarity(course_codes[0], course_codes[1])
            assert 0.0 <= sim <= 1.0, "Similarity should be between 0 and 1"
    
    def test_similarity_invalid_course(self):
        """Test similarity with invalid course codes."""
        sim = compute_similarity("INVALID1", "INVALID2")
        assert sim == 0.0, "Invalid courses should have 0 similarity"
    
    def test_similarity_one_invalid(self):
        """Test similarity with one valid and one invalid course."""
        course_code = list(courses_dict.keys())[0]
        sim = compute_similarity(course_code, "INVALID")
        assert sim == 0.0, "Invalid course should result in 0 similarity"


class TestEndpoint:
    """Test the /analyze endpoint."""
    
    def test_endpoint_basic_query(self):
        """Test basic endpoint functionality."""
        response = client.post(
            "/analyze",
            json={"question": "I have already taken 01002. Which of these courses would overlap the most: 01003, 01018, 01017?"}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "completed_courses" in data
        assert "compared_courses" in data
        assert "ranking" in data
        
        # Check data types
        assert isinstance(data["completed_courses"], list)
        assert isinstance(data["compared_courses"], list)
        assert isinstance(data["ranking"], list)
    
    def test_endpoint_ranking_structure(self):
        """Test that ranking items have correct structure."""
        response = client.post(
            "/analyze",
            json={"question": "I completed 01002. Compare 01003 and 01018."}
        )
        assert response.status_code == 200
        data = response.json()
        
        for item in data["ranking"]:
            assert "course_number" in item
            assert "similarity" in item
            assert "recommendation" in item
            
            # Check data types
            assert isinstance(item["course_number"], str)
            assert isinstance(item["similarity"], (int, float))
            assert isinstance(item["recommendation"], str)
            assert 0.0 <= item["similarity"] <= 1.0
    
    def test_endpoint_ranking_sorted(self):
        """Test that ranking is sorted by similarity in descending order."""
        response = client.post(
            "/analyze",
            json={"question": "I completed 01002. Compare 01003 and 01018."}
        )
        assert response.status_code == 200
        data = response.json()
        
        if len(data["ranking"]) > 1:
            similarities = [item["similarity"] for item in data["ranking"]]
            assert similarities == sorted(similarities, reverse=True), "Ranking should be sorted by similarity descending"
    
    def test_endpoint_recommendations(self):
        """Test that recommendations are based on similarity thresholds."""
        response = client.post(
            "/analyze",
            json={"question": "I completed 01002. Compare 01003, 01018, and 01017."}
        )
        assert response.status_code == 200
        data = response.json()
        
        for item in data["ranking"]:
            if item["similarity"] > 0.6:
                assert "High overlap" in item["recommendation"]
            elif item["similarity"] > 0.3:
                assert "Moderate overlap" in item["recommendation"]
            else:
                assert "Low overlap" in item["recommendation"]
    
    def test_endpoint_empty_question(self):
        """Test endpoint with empty question."""
        response = client.post(
            "/analyze",
            json={"question": ""}
        )
        assert response.status_code == 200
        data = response.json()
        assert "The assistant could not identify the courses." in data['error']
    
    def test_endpoint_invalid_json(self):
        """Test endpoint with invalid JSON."""
        response = client.post(
            "/analyze",
            json={"invalid_field": "test"}
        )
        assert response.status_code == 422
    
    def test_endpoint_hello_query(self):
        """Test endpoint with 'Hello' query returns empty lists."""
        response = client.post(
            "/analyze",
            json={"question": "Hello"}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check for expected empty response structure
        assert data["error"] == "The assistant could not identify the courses."
    
    def test_endpoint_01002_01003_query(self):
        """Test endpoint with specific course comparison query."""
        response = client.post(
            "/analyze",
            json={"question": "I did 01002, I want to take 01003. Should I?"}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "completed_courses" in data
        assert "compared_courses" in data
        assert "ranking" in data
        
        # Check that we have data
        assert isinstance(data["completed_courses"], list)
        assert isinstance(data["compared_courses"], list)
        assert isinstance(data["ranking"], list)
        
        # Verify ranking has similarity scores and recommendations
        for item in data["ranking"]:
            assert "course_number" in item
            assert "similarity" in item
            assert "recommendation" in item
            assert isinstance(item["similarity"], (int, float))
            assert 0.0 <= item["similarity"] <= 1.0
    
    def test_endpoint_invalid_course(self):
        """Test endpoint with non-existent course code."""
        response = client.post(
            "/analyze",
            json={"question": "I did INVALID123, I want to take 01003. Should I?"}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Should return an error message
        assert "error" in data
        assert "is not in the database!" in data["error"]
        assert "INVALID123" in data["error"]
    
    def test_endpoint_multiple_invalid_courses(self):
        """Test endpoint with multiple non-existent course codes."""
        response = client.post(
            "/analyze",
            json={"question": "I did INVALID1, INVALID2 and INVALID3, I want to take 01003. Should I?"}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Should return an error message listing all invalid courses
        assert "error" in data
        assert "are not in the database!" in data["error"]
        assert "INVALID1" in data["error"]
        assert "INVALID2" in data["error"]
        assert "INVALID3" in data["error"]
    
    def test_endpoint_not_applicable_together(self):
        """Test endpoint detects courses not applicable together."""
        response = client.post(
            "/analyze",
            json={"question": "I did 01003, I want to take 01915. Should I?"}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Should return an error about not being applicable together
        assert "error" in data
        assert "is not applicable with course" in data["error"]
        assert "01003" in data["error"]
        assert "01915" in data["error"]


class TestCoursesData:
    """Test the courses data structure."""
    
    def test_courses_dict_not_empty(self):
        """Test that courses dictionary is loaded."""
        assert len(courses_dict) > 0, "Courses dictionary should not be empty"
    
    def test_course_structure(self):
        """Test that courses have expected structure."""
        for course_code, course_data in list(courses_dict.items())[:5]:
            assert isinstance(course_data, dict), "Course should be a dictionary"
            # Check for common fields
            assert "learning_objectives" in course_data or "fields" in course_data, \
                "Course should have learning_objectives or fields"
    
    def test_course_codes_format(self):
        """Test that course codes are in expected format."""
        for course_code in list(courses_dict.keys())[:10]:
            assert isinstance(course_code, str), "Course code should be a string"
            assert len(course_code) > 0, "Course code should not be empty"


class TestAppHealth:
    """Test general app health and endpoints."""
    
    def test_app_startup(self):
        """Test that app can be instantiated."""
        assert app is not None
    
    def test_root_endpoint_not_found(self):
        """Test that root endpoint returns 404."""
        response = client.get("/")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
