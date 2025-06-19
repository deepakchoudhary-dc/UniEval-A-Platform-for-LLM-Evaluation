"""
Memory management system for the AI Chatbot
"""
import json
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMADB = True
except ImportError:
    chromadb = None
    ChromaSettings = None
    HAS_CHROMADB = False
    
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.data import db_manager, ConversationEntry, MemorySearchResult
from config.settings import settings


class MemoryManager:
    """Manages chatbot memory with semantic search capabilities"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.session_id = str(uuid.uuid4())
        
        # Initialize ChromaDB for vector storage (if available)
        if HAS_CHROMADB:
            try:
                self.chroma_client = chromadb.Client(ChromaSettings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=settings.search_index_path
                ))
                
                try:
                    self.collection = self.chroma_client.get_collection("conversations")
                except:
                    self.collection = self.chroma_client.create_collection("conversations")
                    
                self.has_vector_db = True
            except Exception as e:
                print(f"Warning: ChromaDB initialization failed: {e}")
                self.chroma_client = None
                self.collection = None
                self.has_vector_db = False
        else:
            print("Warning: ChromaDB not available. Using fallback memory storage.")
            self.chroma_client = None
            self.collection = None
            self.has_vector_db = False
        
        # Initialize NLTK components
        self._init_nltk()
          # Memory context for current session
        self.current_context = []
        self.context_limit = 10  # Keep last 10 exchanges in active context
    
    def _init_nltk(self):
        """Initialize NLTK components"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def store_conversation(
        self,
        user_query: str,
        ai_response: str,
        model_used: str,
        confidence_score: float,
        response_time_ms: int,
        session_id: Optional[str] = None,
        explanation_data: Optional[Dict] = None,
        data_sources: Optional[List[str]] = None,
        decision_factors: Optional[Dict] = None,
        bias_check_results: Optional[Dict] = None,
        fairness_score: Optional[float] = None
    ) -> ConversationEntry:
        """Store a conversation in memory with metadata"""
        
        # Generate embeddings for semantic search
        combined_text = f"{user_query} {ai_response}"
        embedding = self.embedding_model.encode(combined_text)
        embedding_vector = json.dumps(embedding.tolist())
        
        # Extract keywords
        keywords = self._extract_keywords(combined_text)
        
        # Determine topic category
        topic_category = self._categorize_topic(user_query)
        
        # Create audit log
        audit_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_used": model_used,
            "embedding_model": settings.embedding_model,
            "processing_steps": [
                "query_received",
                "context_retrieved",
                "response_generated",
                "explanation_computed",
                "bias_checked",
                "memory_stored"
            ]
        }
          # Save to database
        entry = db_manager.save_conversation(
            session_id=session_id or self.session_id,
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
          # Store in vector database for similarity search (if available)
        if self.has_vector_db:
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[combined_text],                metadatas=[{
                    "id": entry.id,
                    "session_id": session_id or self.session_id,
                    "timestamp": entry.timestamp.isoformat(),
                    "user_query": user_query,
                    "ai_response": ai_response,
                    "topic_category": topic_category,
                    "keywords": keywords
                }],
                ids=[str(entry.id)]
            )
          # Update current session context (only for current session)
        if (session_id or self.session_id) == self.session_id:
            self.current_context.append({
                "user_query": user_query,
                "ai_response": ai_response,
                "timestamp": entry.timestamp,
                "id": entry.id
            })
        
        # Keep context within limit
        if len(self.current_context) > self.context_limit:
            self.current_context.pop(0)
        
        return entry
    
    def search_memory(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = None,
        similarity_threshold: float = 0.7
    ) -> List[MemorySearchResult]:
        """Search memory using semantic similarity and text matching"""
        
        if limit is None:
            limit = settings.max_search_results
        
        results = []
        
        # Semantic search using embeddings (if vector DB is available)
        if self.has_vector_db:
            query_embedding = self.embedding_model.encode(query)
            
            # Search in ChromaDB
            chroma_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit * 2  # Get more results to filter by session if needed
            )
            
            # Process ChromaDB results
            for i, (doc, metadata, distance) in enumerate(zip(
                chroma_results['documents'][0],
                chroma_results['metadatas'][0],
                chroma_results['distances'][0]
            )):
                similarity_score = 1 - distance  # Convert distance to similarity
                
                if similarity_score >= similarity_threshold:
                    # Filter by session if specified
                    if session_id and metadata.get('session_id') != session_id:
                        continue
                      # Create result with context
                    context = self._get_conversation_context(metadata['id'])
                    
                    result = MemorySearchResult(
                        id=metadata['id'],
                        snippet=doc[:200] + "..." if len(doc) > 200 else doc,
                        timestamp=datetime.fromisoformat(metadata['timestamp']),
                        similarity_score=similarity_score,
                        context=context,
                        session_id=metadata['session_id']
                    )
                    results.append(result)
        else:
            # Fallback to text-based search when vector DB is not available
            print("Using fallback text-based search (ChromaDB not available)")
        
        # Also do text-based search in database
        db_results = db_manager.search_conversations(query, session_id, limit)
        
        # Combine and deduplicate results
        seen_ids = {r.id for r in results}
        for db_result in db_results:
            if db_result.id not in seen_ids:
                result = MemorySearchResult(
                    id=db_result.id,
                    snippet=f"{db_result.user_query[:100]}... -> {db_result.ai_response[:100]}...",
                    timestamp=db_result.timestamp,
                    similarity_score=0.5,  # Lower score for text-only matches
                    context=self._get_conversation_context(db_result.id),
                    session_id=db_result.session_id
                )
                results.append(result)
        
        # Sort by similarity score and timestamp
        results.sort(key=lambda x: (x.similarity_score, x.timestamp), reverse=True)
        
        return results[:limit]
    
    def get_relevant_context(
        self,
        current_query: str,
        max_context_items: int = 5
    ) -> List[Dict[str, Any]]:
        """Get relevant context from memory for current query"""
        
        # First, get the most recent conversations from current session
        recent_context = self.current_context[-max_context_items:]
        
        # Then, search for semantically similar conversations
        similar_memories = self.search_memory(
            current_query,
            limit=max_context_items,
            similarity_threshold=0.6
        )
        
        # Combine and deduplicate
        context_items = []
        seen_ids = set()
        
        # Add recent context first
        for item in recent_context:
            if item['id'] not in seen_ids:
                context_items.append({
                    "source": "recent_conversation",
                    "user_query": item['user_query'],
                    "ai_response": item['ai_response'],
                    "timestamp": item['timestamp'],
                    "relevance_score": 1.0,
                    "id": item['id']
                })
                seen_ids.add(item['id'])
        
        # Add similar memories
        for memory in similar_memories:
            if memory.id not in seen_ids and len(context_items) < max_context_items:
                context_items.append({
                    "source": "similar_memory",
                    "snippet": memory.snippet,
                    "timestamp": memory.timestamp,
                    "relevance_score": memory.similarity_score,
                    "context": memory.context,
                    "id": memory.id
                })
                seen_ids.add(memory.id)
        
        return context_items[:max_context_items]
    
    def get_conversation_source(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed source information for a specific conversation"""
        
        entry = db_manager.get_conversation_by_id(conversation_id)
        if not entry:
            return None
        
        return {
            "id": entry.id,
            "session_id": entry.session_id,
            "timestamp": entry.timestamp,
            "user_query": entry.user_query,
            "ai_response": entry.ai_response,
            "model_used": entry.model_used,
            "confidence_score": entry.confidence_score,
            "data_sources": entry.data_sources,
            "decision_factors": entry.decision_factors,
            "explanation_data": entry.explanation_data,
            "audit_log": entry.audit_log,
            "topic_category": entry.topic_category,
            "keywords": entry.keywords
        }
    
    def clear_session_memory(self, session_id: Optional[str] = None):
        """Clear memory for a specific session or current session"""
        
        target_session = session_id or self.session_id
        
        # Clear from database
        db_entries = db_manager.get_session_history(target_session, limit=1000)
        for entry in db_entries:
            db_manager.db_session.delete(entry)
        db_manager.db_session.commit()
        
        # Clear from vector database
        # Note: ChromaDB doesn't support filtering by metadata for deletion in all versions
        # This is a limitation that might need manual handling
        
        if target_session == self.session_id:
            self.current_context.clear()
    
    def _extract_keywords(self, text: str, max_keywords: int = 10) -> str:
        """Extract keywords from text for search indexing"""
        
        # Tokenize and clean
        tokens = word_tokenize(text.lower())
        
        # Remove stopwords and non-alphabetic tokens
        keywords = [
            self.lemmatizer.lemmatize(token)
            for token in tokens
            if token.isalpha() and token not in self.stop_words and len(token) > 2
        ]
        
        # Get unique keywords
        unique_keywords = list(set(keywords))
        
        # Return top keywords (could implement TF-IDF here for better ranking)
        return " ".join(unique_keywords[:max_keywords])
    
    def _categorize_topic(self, query: str) -> str:
        """Categorize the topic of a query (simple rule-based approach)"""
        
        query_lower = query.lower()
        
        # Define topic categories and keywords
        categories = {
            "weather": ["weather", "temperature", "rain", "sunny", "cloudy", "forecast"],
            "technology": ["computer", "software", "programming", "code", "tech", "ai", "machine learning"],
            "health": ["health", "medical", "doctor", "medicine", "symptom", "treatment"],
            "travel": ["travel", "trip", "vacation", "flight", "hotel", "destination"],
            "food": ["food", "recipe", "cooking", "restaurant", "meal", "eat"],
            "entertainment": ["movie", "music", "book", "game", "entertainment", "fun"],
            "education": ["learn", "study", "school", "education", "teach", "knowledge"],
            "finance": ["money", "investment", "bank", "finance", "cost", "price"],
            "personal": ["personal", "advice", "help", "problem", "question"]
        }
        
        # Score each category
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                category_scores[category] = score
        
        # Return the highest scoring category or "general"
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return "general"
    
    def _get_conversation_context(self, conversation_id: int) -> str:
        """Get context around a specific conversation"""
        
        entry = db_manager.get_conversation_by_id(conversation_id)
        if not entry:
            return ""
        
        context_parts = [
            f"User asked: {entry.user_query}",
            f"AI responded: {entry.ai_response[:200]}...",
            f"Confidence: {entry.confidence_score:.2f}",
            f"Topic: {entry.topic_category}",
            f"Time: {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        ]
        
        return " | ".join(context_parts)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system"""
        
        total_conversations = db_manager.db_session.query(ConversationEntry).count()
        session_conversations = len(self.current_context)
          # Get topic distribution
        from sqlalchemy import func
        topic_query = db_manager.db_session.query(
            ConversationEntry.topic_category,
            func.count(ConversationEntry.id).label('count')
        ).group_by(ConversationEntry.topic_category).all()
        
        topic_distribution = {topic: count for topic, count in topic_query}
        
        return {
            "total_conversations": total_conversations,
            "current_session_conversations": session_conversations,
            "active_sessions": 1,  # Simplified for single session
            "topic_distribution": topic_distribution,
            "memory_limit": settings.max_memory_entries,
            "retention_days": settings.memory_retention_days
        }


# Global memory manager instance
memory_manager = MemoryManager()
