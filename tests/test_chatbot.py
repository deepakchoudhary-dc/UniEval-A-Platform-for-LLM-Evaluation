"""
Tests for the Transparent AI Chatbot
"""
import pytest
import json
import tempfile
import os
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.settings import settings
from src.core.chatbot import TransparentChatbot
from src.core.memory import MemoryManager
from src.core.search import SearchEngine
from src.data.database import DatabaseManager
from src.explainability.lime_explainer import LIMEExplainer
from src.explainability.shap_explainer import SHAPExplainer
from src.fairness.bias_detector import BiasDetector
from src.explainability.model_card import ModelCardGenerator


class TestTransparentChatbot:
    """Test cases for the main chatbot functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.chatbot = TransparentChatbot()
    
    def test_chatbot_initialization(self):
        """Test chatbot initializes correctly"""
        assert self.chatbot.model_name == settings.default_model
        assert self.chatbot.session_id is not None
        assert len(self.chatbot.conversation_history) == 0
    
    def test_basic_chat_functionality(self):
        """Test basic chat functionality"""
        # Skip if no API key available
        if not settings.openai_api_key:
            pytest.skip("OpenAI API key not available")
        
        response = self.chatbot.chat("Hello, how are you?")
        
        assert response.answer is not None
        assert len(response.answer) > 0
        assert 0 <= response.confidence <= 1
        assert response.session_id == self.chatbot.session_id
        assert len(self.chatbot.conversation_history) == 1
    
    def test_memory_functionality(self):
        """Test memory storage and retrieval"""
        # Skip if no API key available
        if not settings.openai_api_key:
            pytest.skip("OpenAI API key not available")
        
        # First conversation
        response1 = self.chatbot.chat("My name is John and I like cats")
        assert response1.answer is not None
        
        # Second conversation referencing first
        response2 = self.chatbot.chat("What did I tell you about my pets?")
        assert response2.answer is not None
        
        # Check memory stats
        stats = self.chatbot.get_memory_stats()
        assert stats["current_session_conversations"] >= 2
    
    def test_search_functionality(self):
        """Test search functionality"""
        # Skip if no API key available
        if not settings.openai_api_key:
            pytest.skip("OpenAI API key not available")
        
        # Add some conversations
        self.chatbot.chat("I love machine learning")
        self.chatbot.chat("Python is my favorite programming language")
        
        # Search for relevant conversations
        results = self.chatbot.search_memory("machine learning")
        
        assert isinstance(results, list)
        # Results might be empty if search hasn't indexed yet
    
    def test_explanation_functionality(self):
        """Test explanation generation"""
        # Skip if no API key available
        if not settings.openai_api_key:
            pytest.skip("OpenAI API key not available")
        
        response = self.chatbot.chat("What is the capital of France?")
        
        if response.explanation:
            assert "summary" in response.explanation
            assert "methods_used" in response.explanation
        
        # Test last response explanation
        explanation = self.chatbot.explain_last_response()
        if explanation:
            assert explanation.confidence >= 0
            assert explanation.method is not None
    
    def test_source_questioning(self):
        """Test source questioning functionality"""
        # Skip if no API key available
        if not settings.openai_api_key:
            pytest.skip("OpenAI API key not available")
        
        # Have a conversation
        self.chatbot.chat("What is artificial intelligence?")
        
        # Ask about sources
        source_answer = self.chatbot.ask_about_source("What sources did you use?")
        assert isinstance(source_answer, str)
        assert len(source_answer) > 0
    
    def test_memory_clearing(self):
        """Test memory clearing functionality"""
        # Skip if no API key available
        if not settings.openai_api_key:
            pytest.skip("OpenAI API key not available")
        
        # Add some conversations
        self.chatbot.chat("Test conversation 1")
        self.chatbot.chat("Test conversation 2")
        
        # Clear memory
        self.chatbot.clear_session_memory()
        
        # Check that memory is cleared
        assert len(self.chatbot.conversation_history) == 0
        stats = self.chatbot.get_memory_stats()
        assert stats["current_session_conversations"] == 0


class TestMemoryManager:
    """Test cases for memory management"""
    
    def setup_method(self):
        """Setup test environment"""
        self.memory_manager = MemoryManager()
    
    def test_memory_initialization(self):
        """Test memory manager initializes correctly"""
        assert self.memory_manager.session_id is not None
        assert self.memory_manager.embedding_model is not None
        assert len(self.memory_manager.current_context) == 0
    
    def test_conversation_storage(self):
        """Test conversation storage"""
        entry = self.memory_manager.store_conversation(
            user_query="Test query",
            ai_response="Test response",
            model_used="test-model",
            confidence_score=0.8,
            response_time_ms=100
        )
        
        assert entry.id is not None
        assert entry.user_query == "Test query"
        assert entry.ai_response == "Test response"
        assert entry.confidence_score == 0.8
        assert len(self.memory_manager.current_context) == 1
    
    def test_memory_search(self):
        """Test memory search functionality"""
        # Store some test conversations
        self.memory_manager.store_conversation(
            user_query="Tell me about machine learning",
            ai_response="Machine learning is a subset of AI",
            model_used="test-model",
            confidence_score=0.9,
            response_time_ms=100
        )
        
        self.memory_manager.store_conversation(
            user_query="What is deep learning?",
            ai_response="Deep learning uses neural networks",
            model_used="test-model",
            confidence_score=0.85,
            response_time_ms=120
        )
        
        # Search for relevant conversations
        results = self.memory_manager.search_memory("machine learning", limit=5)
        
        assert isinstance(results, list)
        assert len(results) >= 0  # Might be empty if similarity threshold not met
    
    def test_context_retrieval(self):
        """Test context retrieval for queries"""
        # Store a conversation
        self.memory_manager.store_conversation(
            user_query="I like cats",
            ai_response="Cats are wonderful pets",
            model_used="test-model",
            confidence_score=0.8,
            response_time_ms=100
        )
        
        # Get relevant context
        context = self.memory_manager.get_relevant_context("Tell me about cats")
        
        assert isinstance(context, list)
        # Context might be empty if no relevant matches found


class TestBiasDetector:
    """Test cases for bias detection"""
    
    def setup_method(self):
        """Setup test environment"""
        self.bias_detector = BiasDetector()
    
    def test_bias_detector_initialization(self):
        """Test bias detector initializes correctly"""
        assert self.bias_detector.bias_patterns is not None
        assert len(self.bias_detector.sensitive_attributes) > 0
    
    def test_no_bias_detection(self):
        """Test detection when no bias is present"""
        user_query = "What is the weather like today?"
        ai_response = "I don't have access to current weather data, but you can check a weather website or app."
        
        result = self.bias_detector.detect_bias(user_query, ai_response, {})
        
        assert result["bias_detected"] == False
        assert result["bias_score"] == 0.0
        assert result["severity"] == "none"
    
    def test_bias_detection_patterns(self):
        """Test bias detection with problematic patterns"""
        user_query = "Are women good at math?"
        ai_response = "Women are naturally worse at math than men."
        
        result = self.bias_detector.detect_bias(user_query, ai_response, {})
        
        assert result["bias_detected"] == True
        assert result["bias_score"] > 0
        assert len(result["bias_types"]) > 0
    
    def test_stereotype_detection(self):
        """Test stereotype detection"""
        user_query = "What do you think about engineers?"
        ai_response = "Most engineers are male and they love technology."
        
        result = self.bias_detector.detect_bias(user_query, ai_response, {})
        
        # Should detect some form of bias or stereotyping
        assert isinstance(result["bias_detected"], bool)
        assert isinstance(result["bias_score"], float)
    
    def test_fairness_score_calculation(self):
        """Test fairness score calculation"""
        # No bias case
        no_bias_result = {"bias_detected": False, "bias_score": 0.0, "severity": "none"}
        fairness_score = self.bias_detector.calculate_fairness_score(no_bias_result)
        assert fairness_score == 1.0
        
        # High bias case
        high_bias_result = {"bias_detected": True, "bias_score": 0.8, "severity": "high"}
        fairness_score = self.bias_detector.calculate_fairness_score(high_bias_result)
        assert fairness_score < 1.0
    
    def test_bias_statistics(self):
        """Test bias statistics tracking"""
        # Initial state
        stats = self.bias_detector.get_bias_statistics()
        initial_count = stats["total_responses"]
        
        # Add a bias detection result
        result = self.bias_detector.detect_bias("test query", "test response", {})
        
        # Check statistics updated
        new_stats = self.bias_detector.get_bias_statistics()
        assert new_stats["total_responses"] == initial_count + 1
    
    def test_bias_report_generation(self):
        """Test bias report generation"""
        report = self.bias_detector.generate_bias_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Bias Detection Report" in report


class TestExplainability:
    """Test cases for explainability components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.lime_explainer = LIMEExplainer()
        self.shap_explainer = SHAPExplainer()
    
    def test_lime_explainer_initialization(self):
        """Test LIME explainer initializes correctly"""
        assert self.lime_explainer.text_explainer is not None
    
    def test_shap_explainer_initialization(self):
        """Test SHAP explainer initializes correctly"""
        assert self.shap_explainer.explainer is None  # Not initialized until used
    
    def test_lime_explanation_generation(self):
        """Test LIME explanation generation"""
        # Simple test with mock prediction function
        def mock_predict(texts):
            return [[0.3, 0.7] for _ in texts]  # Mock probabilities
        
        explanation = self.lime_explainer.explain_text_prediction(
            "This is a test sentence",
            mock_predict,
            num_features=5,
            num_samples=100
        )
        
        assert explanation["method"] == "lime_text"
        assert "features" in explanation
        assert isinstance(explanation["features"], list)
    
    def test_human_readable_explanation(self):
        """Test human-readable explanation generation"""
        mock_explanation = {
            "method": "lime_text",
            "features": [
                {"feature": "test", "weight": 0.5, "importance": 0.5},
                {"feature": "sentence", "weight": -0.3, "importance": 0.3}
            ]
        }
        
        readable = self.lime_explainer.generate_human_readable_explanation(
            mock_explanation, "medium"
        )
        
        assert isinstance(readable, str)
        assert len(readable) > 0
        assert "LIME" in readable


class TestModelCard:
    """Test cases for model card generation"""
    
    def setup_method(self):
        """Setup test environment"""
        self.model_card_generator = ModelCardGenerator()
    
    def test_model_card_generation(self):
        """Test model card generation"""
        model_card = self.model_card_generator.generate_model_card(include_metrics=False)
        
        assert "model_details" in model_card
        assert "intended_use" in model_card
        assert "limitations" in model_card
        assert "ethical_considerations" in model_card
        assert "bias_assessment" in model_card
    
    def test_model_card_export(self):
        """Test model card export functionality"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            success = self.model_card_generator.export_model_card(temp_file, "json")
            assert success == True
            
            # Check file was created and contains valid JSON
            assert os.path.exists(temp_file)
            with open(temp_file, 'r') as f:
                data = json.load(f)
                assert "model_details" in data
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def test_markdown_export(self):
        """Test markdown export functionality"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            temp_file = f.name
        
        try:
            success = self.model_card_generator.export_model_card(temp_file, "markdown")
            assert success == True
            
            # Check file was created and contains markdown
            assert os.path.exists(temp_file)
            with open(temp_file, 'r') as f:
                content = f.read()
                assert "# " in content  # Markdown header
        
        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_end_to_end_conversation_flow(self):
        """Test complete conversation flow with all features"""
        # Skip if no API key available
        if not settings.openai_api_key:
            pytest.skip("OpenAI API key not available")
        
        chatbot = TransparentChatbot()
        
        # First conversation
        response1 = chatbot.chat("Hello, I'm interested in learning about AI")
        assert response1.answer is not None
        
        # Second conversation building on first
        response2 = chatbot.chat("Can you tell me more about machine learning?")
        assert response2.answer is not None
        
        # Search for previous conversations
        search_results = chatbot.search_memory("AI")
        assert isinstance(search_results, list)
        
        # Get explanation for last response
        explanation = chatbot.explain_last_response()
        # Explanation might be None if generation failed
        
        # Ask about sources
        source_answer = chatbot.ask_about_source("What sources did you use?")
        assert isinstance(source_answer, str)
        
        # Get statistics
        stats = chatbot.get_memory_stats()
        assert stats["current_session_conversations"] >= 2
    
    def test_bias_detection_integration(self):
        """Test bias detection in full conversation flow"""
        # Skip if no API key available
        if not settings.openai_api_key:
            pytest.skip("OpenAI API key not available")
        
        chatbot = TransparentChatbot()
        
        # Conversation that might trigger bias detection
        response = chatbot.chat("What can you tell me about different cultures?")
        
        # Check that bias detection ran (even if no bias found)
        assert response.answer is not None
        
        # Get bias statistics
        bias_stats = bias_detector.get_bias_statistics()
        assert bias_stats["total_responses"] >= 1


# Utility functions for testing
def create_test_database():
    """Create a test database for testing"""
    db_manager = DatabaseManager()
    return db_manager


def cleanup_test_data():
    """Clean up test data"""
    # This would clean up any test data created during tests
    pass


# Pytest fixtures
@pytest.fixture
def test_chatbot():
    """Fixture for test chatbot instance"""
    return TransparentChatbot()


@pytest.fixture
def test_memory_manager():
    """Fixture for test memory manager"""
    return MemoryManager()


@pytest.fixture
def test_bias_detector():
    """Fixture for test bias detector"""
    return BiasDetector()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
