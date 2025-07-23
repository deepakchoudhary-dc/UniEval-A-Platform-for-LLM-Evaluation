"""
Database models and operations for the AI Chatbot
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel
import json

from config.settings import settings

Base = declarative_base()


class ConversationEntry(Base):
    """Database model for conversation entries"""
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True)
    user_query = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Metadata
    model_used = Column(String(100))
    confidence_score = Column(Float)
    response_time_ms = Column(Integer)
    
    # Explainability data
    explanation_data = Column(JSON)  # LIME/SHAP results
    data_sources = Column(JSON)  # Referenced memory entries
    decision_factors = Column(JSON)  # Key factors in decision making
    
    # Search and embedding
    embedding_vector = Column(Text)  # Serialized embedding for similarity search
    keywords = Column(Text)  # Extracted keywords for search
    topic_category = Column(String(100))
    
    # Audit trail
    audit_log = Column(JSON)
    bias_check_results = Column(JSON)
    fairness_score = Column(Float)


class UserSession(Base):
    """Database model for user sessions"""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
    user_preferences = Column(JSON)
    memory_summary = Column(Text)


class ModelCard(Base):
    """Database model for storing model cards"""
    __tablename__ = "model_cards"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), unique=True)
    version = Column(String(50))
    description = Column(Text)
    capabilities = Column(JSON)
    limitations = Column(JSON)
    training_data = Column(JSON)
    bias_assessment = Column(JSON)
    performance_metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Pydantic models for API
class ConversationResponse(BaseModel):
    """Response model for conversation"""
    answer: str
    explanation: Optional[Dict[str, Any]] = None
    confidence: float
    sources: List[str] = []
    timestamp: datetime
    session_id: str
    bias_check_results: Optional[Dict[str, Any]] = None  # Add bias detection results


class MemorySearchResult(BaseModel):
    """Result model for memory search"""
    id: int
    snippet: str
    timestamp: datetime
    similarity_score: float
    context: str
    session_id: str


class ExplanationResult(BaseModel):
    """Explanation result model"""
    method: str  # "lime" or "shap"
    confidence: float
    key_factors: List[Dict[str, Any]]
    data_sources: List[str]
    decision_reasoning: str


class DatabaseManager:
    """Database operations manager"""
    
    def __init__(self):
        self.engine = create_engine(settings.database_url)
        Base.metadata.create_all(bind=self.engine)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.db_session = SessionLocal()
    
    def save_conversation(
        self,
        session_id: str,
        user_query: str,
        ai_response: str,
        model_used: str,
        confidence_score: float,
        response_time_ms: int,
        explanation_data: Optional[Dict] = None,
        data_sources: Optional[List[str]] = None,
        decision_factors: Optional[Dict] = None,
        embedding_vector: Optional[str] = None,
        keywords: Optional[str] = None,
        topic_category: Optional[str] = None,
        audit_log: Optional[Dict] = None,
        bias_check_results: Optional[Dict] = None,
        fairness_score: Optional[float] = None
    ) -> ConversationEntry:
        """Save a conversation entry to the database"""
        
        entry = ConversationEntry(
            session_id=session_id,
            user_query=user_query,
            ai_response=ai_response,
            model_used=model_used,
            confidence_score=confidence_score,
            response_time_ms=response_time_ms,
            explanation_data=explanation_data,
            data_sources=data_sources,
            decision_factors=decision_factors,
            embedding_vector=embedding_vector,
            keywords=keywords,
            topic_category=topic_category,
            audit_log=audit_log,
            bias_check_results=bias_check_results,
            fairness_score=fairness_score
        )
        
        self.db_session.add(entry)
        self.db_session.commit()
        self.db_session.refresh(entry)
        
        # Clean up old entries if limit exceeded
        self._cleanup_old_entries()
        
        return entry
    
    def search_conversations(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = None
    ) -> List[ConversationEntry]:
        """Search conversations by query text"""
        
        if limit is None:
            limit = settings.max_search_results
        
        db_query = self.db_session.query(ConversationEntry)
        
        if session_id:
            db_query = db_query.filter(ConversationEntry.session_id == session_id)
        
        # Simple text search (can be enhanced with vector similarity)
        db_query = db_query.filter(
            ConversationEntry.user_query.contains(query) |
            ConversationEntry.ai_response.contains(query) |
            ConversationEntry.keywords.contains(query)
        )
        
        return db_query.order_by(ConversationEntry.timestamp.desc()).limit(limit).all()
    
    def get_conversation_by_id(self, conversation_id: int) -> Optional[ConversationEntry]:
        """Get a specific conversation by ID"""
        return self.db_session.query(ConversationEntry).filter(
            ConversationEntry.id == conversation_id
        ).first()
    
    def get_session_history(self, session_id: str, limit: int = 50) -> List[ConversationEntry]:
        """Get conversation history for a session"""
        return self.db_session.query(ConversationEntry).filter(
            ConversationEntry.session_id == session_id
        ).order_by(ConversationEntry.timestamp.desc()).limit(limit).all()
    
    def create_or_update_session(self, session_id: str, preferences: Optional[Dict] = None) -> UserSession:
        """Create or update a user session"""
        session = self.db_session.query(UserSession).filter(
            UserSession.session_id == session_id
        ).first()
        
        if session:
            session.last_active = datetime.utcnow()
            if preferences:
                session.user_preferences = preferences
        else:
            session = UserSession(
                session_id=session_id,
                user_preferences=preferences or {}
            )
            self.db_session.add(session)
        
        self.db_session.commit()
        return session
    
    def save_model_card(
        self,
        model_name: str,
        version: str,
        description: str,
        capabilities: Dict,
        limitations: Dict,
        training_data: Dict,
        bias_assessment: Dict,
        performance_metrics: Dict
    ) -> ModelCard:
        """Save or update a model card"""
        
        card = self.db_session.query(ModelCard).filter(
            ModelCard.model_name == model_name
        ).first()
        
        if card:
            card.version = version
            card.description = description
            card.capabilities = capabilities
            card.limitations = limitations
            card.training_data = training_data
            card.bias_assessment = bias_assessment
            card.performance_metrics = performance_metrics
            card.updated_at = datetime.utcnow()
        else:
            card = ModelCard(
                model_name=model_name,
                version=version,
                description=description,
                capabilities=capabilities,
                limitations=limitations,
                training_data=training_data,
                bias_assessment=bias_assessment,
                performance_metrics=performance_metrics
            )
            self.db_session.add(card)
        
        self.db_session.commit()
        return card
    
    def get_model_card(self, model_name: str) -> Optional[ModelCard]:
        """Get model card by name"""
        return self.db_session.query(ModelCard).filter(
            ModelCard.model_name == model_name
        ).first()
    
    def _cleanup_old_entries(self):
        """Remove old conversation entries if limit exceeded"""
        total_entries = self.db_session.query(ConversationEntry).count()
        
        if total_entries > settings.max_memory_entries:
            # Delete oldest entries beyond the limit
            entries_to_delete = total_entries - settings.max_memory_entries
            old_entries = self.db_session.query(ConversationEntry).order_by(
                ConversationEntry.timestamp.asc()
            ).limit(entries_to_delete).all()
            
            for entry in old_entries:
                self.db_session.delete(entry)
            
            self.db_session.commit()
        
        # Also delete entries older than retention period
        cutoff_date = datetime.utcnow() - timedelta(days=settings.memory_retention_days)
        old_entries = self.db_session.query(ConversationEntry).filter(
            ConversationEntry.timestamp < cutoff_date
        ).all()
        
        for entry in old_entries:
            self.db_session.delete(entry)
        
        self.db_session.commit()
    
    def close(self):
        """Close database connection"""
        self.db_session.close()


# Global database manager instance
db_manager = DatabaseManager()
