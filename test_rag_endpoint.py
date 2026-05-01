import pytest
from fastapi.testclient import TestClient
from main import app, compute_similarity, courses_dict

client = TestClient(app)
# Tests for the analyze-rag endpoint. Run with 'uv run pytest test_rag_endpoint.py -v'.

class TestAnalyzeRagEndpoint:
    """Test the /analyze-rag endpoint."""
    
    def test_rag_endpoint_basic_query(self):
        """Test basic RAG endpoint functionality."""
        response = client.post(
            "/analyze-rag",
            json={"question": "I have already taken 01002. Which of these courses would overlap the most: 01003, 01018, 01017?"}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "completed_courses" in data
        assert "compared_courses" in data
        assert "ranking" in data
        assert "answer" in data
        assert "sources" in data
        assert "rag" in data
        
        # Check data types
        assert isinstance(data["completed_courses"], list)
        assert isinstance(data["compared_courses"], list)
        assert isinstance(data["ranking"], list)
        assert isinstance(data["sources"], list)
        assert isinstance(data["rag"], dict)
        
        # Make sure "rag" metadata has right keys
        assert "retrieved_course_codes" in data["rag"]
        assert isinstance(data["rag"]["retrieved_course_codes"], list)
        assert "deterministic_similarity_ranking" in data["rag"]
    
    def test_rag_endpoint_ranking_structure(self):
        """Test that ranking items have correct structure and RAG specific fields."""
        response = client.post(
            "/analyze-rag",
            json={"question": "I completed 01002. Compare 01003 and 01018."}
        )
        assert response.status_code == 200
        data = response.json()
        
        for item in data["ranking"]:
            assert "course_number" in item
            assert "similarity" in item
            assert "recommendation" in item
            assert "overlap_level" in item
            
            # Check data types
            assert isinstance(item["course_number"], str)
            assert isinstance(item["similarity"], (int, float))
            assert isinstance(item["recommendation"], str)
            assert 0.0 <= item["similarity"] <= 1.0
            
            # Evidence might be populated if LLM succeeds, or fallback handles it
            if "evidence" in item:
                assert isinstance(item["evidence"], list)

    def test_rag_endpoint_ranking_sorted(self):
        """Test that ranking is sorted by similarity in descending order."""
        response = client.post(
            "/analyze-rag",
            json={"question": "I completed 01002. Compare 01003 and 01018."}
        )
        assert response.status_code == 200
        data = response.json()
        
        if len(data["ranking"]) > 1:
            similarities = [item["similarity"] for item in data["ranking"]]
            assert similarities == sorted(similarities, reverse=True), "Ranking should be sorted by similarity descending"
    
    def test_rag_endpoint_recommendations(self):
        """Test that recommendations are based on similarity thresholds."""
        response = client.post(
            "/analyze-rag",
            json={"question": "I completed 01002. Compare 01003, 01018, and 01017."}
        )
        assert response.status_code == 200
        data = response.json()
        
        for item in data["ranking"]:
            if item["similarity"] > 0.6:
                assert dict(item)["overlap_level"] == "high"
            elif item["similarity"] > 0.3:
                assert dict(item)["overlap_level"] == "moderate"
            else:
                assert dict(item)["overlap_level"] == "low"
    
    def test_rag_endpoint_invalid_json(self):
        """Test RAG endpoint with invalid JSON."""
        response = client.post(
            "/analyze-rag",
            json={"invalid_field": "test"}
        )
        assert response.status_code == 422 
    
    def test_rag_endpoint_hello_query(self):
        """Test RAG endpoint with generic query returns empty lists."""
        response = client.post(
            "/analyze-rag",
            json={"question": "Hello"}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check for expected empty response structure
        assert data["completed_courses"] == []
        assert data["compared_courses"] == []
        assert data["ranking"] == []
    
    def test_rag_endpoint_invalid_course(self):
        """Test RAG endpoint with non-existent course code."""
        response = client.post(
            "/analyze-rag",
            json={"question": "I did INVALID123, I want to take 01003. Should I?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "INVALID123" in data["answer"]
    
    def test_rag_endpoint_multiple_invalid_courses(self):
        """Test RAG endpoint with multiple non-existent course codes."""
        response = client.post(
            "/analyze-rag",
            json={"question": "I did INVALID1, INVALID2 and INVALID3, I want to take 01003. Should I?"}
        )
        assert response.status_code == 200
        data = response.json()
    
        assert "INVALID1" in data["answer"]
        assert "INVALID2" in data["answer"]
        assert "INVALID3" in data["answer"]
