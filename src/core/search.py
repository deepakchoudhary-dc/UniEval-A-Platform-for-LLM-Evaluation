"""
Search functionality for the AI Chatbot
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import whoosh
from whoosh.index import create_in, open_dir, exists_in
from whoosh.fields import Schema, TEXT, ID, DATETIME, NUMERIC
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh.query import Query
from whoosh.searching import Results
import spacy
from spacy.matcher import Matcher

from src.data.database import db_manager, ConversationEntry
from src.core.memory import memory_manager
from config.settings import settings


class SearchEngine:
    """Advanced search engine for chatbot memory and conversations"""
    
    def __init__(self):
        self.index_dir = settings.search_index_path
        self.index = None
        self.nlp = None
        self.matcher = None
        
        # Initialize search index
        self._init_search_index()
        
        # Initialize NLP components
        self._init_nlp()
    
    def _init_search_index(self):
        """Initialize Whoosh search index"""
        
        # Define schema for search index
        schema = Schema(
            id=ID(stored=True, unique=True),
            session_id=ID(stored=True),
            user_query=TEXT(stored=True, phrase=True),
            ai_response=TEXT(stored=True, phrase=True),
            combined_text=TEXT(stored=True, phrase=True),
            keywords=TEXT(stored=True),
            topic_category=TEXT(stored=True),
            timestamp=DATETIME(stored=True),
            confidence_score=NUMERIC(stored=True),
            model_used=TEXT(stored=True)
        )
        
        # Create or open index
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        
        if exists_in(self.index_dir):
            self.index = open_dir(self.index_dir)
        else:
            self.index = create_in(self.index_dir, schema)
            # Index existing conversations
            self._index_existing_conversations()
    
    def _init_nlp(self):
        """Initialize spaCy NLP components"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            return
        
        # Initialize matcher for specific patterns
        self.matcher = Matcher(self.nlp.vocab)
        
        # Add patterns for common query types
        self._add_search_patterns()
    
    def _add_search_patterns(self):
        """Add patterns for recognizing search intent"""
        if not self.matcher:
            return
        
        # Pattern for "find conversations about X"
        pattern1 = [{"LOWER": {"IN": ["find", "search", "look"]}},
                   {"LOWER": {"IN": ["for", "about"]}},
                   {"IS_ALPHA": True, "OP": "+"}]
        
        # Pattern for "what did we discuss about X"
        pattern2 = [{"LOWER": "what"},
                   {"LOWER": "did"},
                   {"LOWER": {"IN": ["we", "i"]}},
                   {"LOWER": {"IN": ["discuss", "talk", "say"]}},
                   {"LOWER": "about"},
                   {"IS_ALPHA": True, "OP": "+"}]
        
        # Pattern for "show me conversations from yesterday/last week"
        pattern3 = [{"LOWER": {"IN": ["show", "find"]}},
                   {"LOWER": "me"},
                   {"LOWER": "conversations"},
                   {"LOWER": "from"},
                   {"IS_ALPHA": True, "OP": "+"}]
        
        self.matcher.add("SEARCH_INTENT", [pattern1, pattern2, pattern3])
    
    def _index_existing_conversations(self):
        """Index all existing conversations in the database"""
        
        conversations = db_manager.db_session.query(ConversationEntry).all()
        
        writer = self.index.writer()
        
        for conv in conversations:
            combined_text = f"{conv.user_query} {conv.ai_response}"
            
            writer.add_document(
                id=str(conv.id),
                session_id=conv.session_id,
                user_query=conv.user_query,
                ai_response=conv.ai_response,
                combined_text=combined_text,
                keywords=conv.keywords or "",
                topic_category=conv.topic_category or "",
                timestamp=conv.timestamp,
                confidence_score=conv.confidence_score or 0.0,
                model_used=conv.model_used or ""
            )
        
        writer.commit()
    
    def index_conversation(self, conversation: ConversationEntry):
        """Index a single conversation"""
        
        combined_text = f"{conversation.user_query} {conversation.ai_response}"
        
        writer = self.index.writer()
        writer.add_document(
            id=str(conversation.id),
            session_id=conversation.session_id,
            user_query=conversation.user_query,
            ai_response=conversation.ai_response,
            combined_text=combined_text,
            keywords=conversation.keywords or "",
            topic_category=conversation.topic_category or "",
            timestamp=conversation.timestamp,
            confidence_score=conversation.confidence_score or 0.0,
            model_used=conversation.model_used or ""
        )
        writer.commit()
    
    def search(
        self,
        query: str,
        session_id: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        topic_filter: Optional[str] = None,
        limit: int = 10,
        search_type: str = "comprehensive"
    ) -> List[Dict[str, Any]]:
        """
        Comprehensive search across conversations
        
        Args:
            query: Search query
            session_id: Filter by specific session
            date_range: Tuple of (start_date, end_date)
            topic_filter: Filter by topic category
            limit: Maximum results to return
            search_type: Type of search ("text", "semantic", "comprehensive")
        """
        
        results = []
        
        if search_type in ["text", "comprehensive"]:
            text_results = self._text_search(query, session_id, date_range, topic_filter, limit)
            results.extend(text_results)
        
        if search_type in ["semantic", "comprehensive"]:
            semantic_results = self._semantic_search(query, session_id, limit)
            results.extend(semantic_results)
        
        # Deduplicate and merge results
        results = self._merge_and_deduplicate_results(results)
        
        # Sort by relevance score
        results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return results[:limit]
    
    def _text_search(
        self,
        query: str,
        session_id: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        topic_filter: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform text-based search using Whoosh"""
        
        # Parse search query
        parsed_query = self._parse_search_query(query)
        
        # Create query parser
        parser = MultifieldParser(
            ["user_query", "ai_response", "combined_text", "keywords"],
            self.index.schema
        )
        
        whoosh_query = parser.parse(parsed_query)
        
        # Add filters
        if session_id:
            from whoosh.query import Term, And
            session_filter = Term("session_id", session_id)
            whoosh_query = And([whoosh_query, session_filter])
        
        if date_range:
            from whoosh.query import DateRange
            date_filter = DateRange("timestamp", date_range[0], date_range[1])
            whoosh_query = And([whoosh_query, date_filter])
        
        if topic_filter:
            from whoosh.query import Term, And
            topic_filter_query = Term("topic_category", topic_filter)
            whoosh_query = And([whoosh_query, topic_filter_query])
        
        # Execute search
        with self.index.searcher() as searcher:
            results = searcher.search(whoosh_query, limit=limit)
            
            search_results = []
            for result in results:
                search_results.append({
                    "id": int(result["id"]),
                    "session_id": result["session_id"],
                    "user_query": result["user_query"],
                    "ai_response": result["ai_response"],
                    "timestamp": result["timestamp"],
                    "topic_category": result["topic_category"],
                    "relevance_score": result.score,
                    "search_type": "text",
                    "matched_fields": list(result.fields()),
                    "highlight": self._get_highlights(result, query)
                })
            
            return search_results
    
    def _semantic_search(
        self,
        query: str,
        session_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Perform semantic search using memory manager"""
        
        semantic_results = memory_manager.search_memory(
            query, session_id, limit, similarity_threshold=0.5
        )
        
        search_results = []
        for result in semantic_results:
            # Get full conversation details
            conversation = memory_manager.get_conversation_source(result.id)
            if conversation:
                search_results.append({
                    "id": result.id,
                    "session_id": result.session_id,
                    "user_query": conversation["user_query"],
                    "ai_response": conversation["ai_response"],
                    "timestamp": result.timestamp,
                    "topic_category": conversation.get("topic_category", ""),
                    "relevance_score": result.similarity_score,
                    "search_type": "semantic",
                    "context": result.context,
                    "snippet": result.snippet
                })
        
        return search_results
    
    def _parse_search_query(self, query: str) -> str:
        """Parse and enhance search query"""
        
        if not self.nlp:
            return query
        
        doc = self.nlp(query)
        
        # Extract key entities and terms
        enhanced_terms = []
        
        # Add original query
        enhanced_terms.append(query)
        
        # Add lemmatized terms
        lemmatized = " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
        if lemmatized:
            enhanced_terms.append(lemmatized)
        
        # Add named entities
        entities = [ent.text for ent in doc.ents]
        enhanced_terms.extend(entities)
        
        # Combine with OR operator
        return " OR ".join([f'"{term}"' for term in enhanced_terms if term.strip()])
    
    def _get_highlights(self, result, query: str) -> Dict[str, str]:
        """Get highlighted snippets from search results"""
        
        highlights = {}
        
        # Simple highlighting (can be enhanced with proper highlighting)
        query_words = query.lower().split()
        
        for field in ["user_query", "ai_response"]:
            text = result.get(field, "")
            if text:
                # Simple highlight implementation
                highlighted = text
                for word in query_words:
                    if word in text.lower():
                        highlighted = highlighted.replace(
                            word, f"<mark>{word}</mark>",
                            1  # Replace only first occurrence
                        )
                highlights[field] = highlighted
        
        return highlights
    
    def _merge_and_deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge and deduplicate search results from different methods"""
        
        # Group by conversation ID
        grouped = {}
        for result in results:
            conv_id = result["id"]
            if conv_id not in grouped:
                grouped[conv_id] = result
            else:
                # Merge results for same conversation
                existing = grouped[conv_id]
                
                # Take the higher relevance score
                if result["relevance_score"] > existing["relevance_score"]:
                    existing["relevance_score"] = result["relevance_score"]
                
                # Merge search types
                existing_types = existing.get("search_type", "").split(",")
                new_type = result.get("search_type", "")
                if new_type not in existing_types:
                    existing["search_type"] = ",".join(existing_types + [new_type])
                
                # Add additional context if available
                if "context" in result and "context" not in existing:
                    existing["context"] = result["context"]
                
                if "highlight" in result and "highlight" not in existing:
                    existing["highlight"] = result["highlight"]
        
        return list(grouped.values())
    
    def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on partial query"""
        
        suggestions = []
        
        # Get suggestions from indexed keywords
        with self.index.searcher() as searcher:
            reader = searcher.reader()
            
            # Get terms from keywords field that start with partial query
            keywords_field = reader.field_terms("keywords")
            for term in keywords_field:
                if term.startswith(partial_query.lower()):
                    suggestions.append(term)
                    if len(suggestions) >= limit:
                        break
        
        # Get suggestions from topic categories
        with self.index.searcher() as searcher:
            reader = searcher.reader()
            topics_field = reader.field_terms("topic_category")
            for topic in topics_field:
                if topic.startswith(partial_query.lower()) and topic not in suggestions:
                    suggestions.append(topic)
                    if len(suggestions) >= limit:
                        break
        
        return suggestions[:limit]
    
    def get_trending_topics(self, days: int = 7, limit: int = 10) -> List[Dict[str, Any]]:
        """Get trending topics from recent conversations"""
        
        # Get conversations from the last N days
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        recent_conversations = db_manager.db_session.query(ConversationEntry).filter(
            ConversationEntry.timestamp >= cutoff_date
        ).all()
        
        # Count topic occurrences
        topic_counts = {}
        for conv in recent_conversations:
            topic = conv.topic_category or "general"
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        # Sort by count
        trending = [
            {"topic": topic, "count": count, "percentage": count / len(recent_conversations) * 100}
            for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return trending[:limit]
    
    def advanced_search(
        self,
        filters: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Advanced search with complex filters
        
        Filters can include:
        - text: Text query
        - session_id: Specific session
        - date_from, date_to: Date range
        - topic: Topic category
        - confidence_min, confidence_max: Confidence score range
        - model: Specific model used
        """
        
        # Build Whoosh query
        query_parts = []
        
        if "text" in filters and filters["text"]:
            parser = MultifieldParser(
                ["user_query", "ai_response", "combined_text", "keywords"],
                self.index.schema
            )
            text_query = parser.parse(self._parse_search_query(filters["text"]))
            query_parts.append(text_query)
        
        # Add other filters
        from whoosh.query import Term, NumericRange, DateRange, And
        
        if "session_id" in filters:
            query_parts.append(Term("session_id", filters["session_id"]))
        
        if "topic" in filters:
            query_parts.append(Term("topic_category", filters["topic"]))
        
        if "model" in filters:
            query_parts.append(Term("model_used", filters["model"]))
        
        if "date_from" in filters or "date_to" in filters:
            date_from = filters.get("date_from", datetime.min)
            date_to = filters.get("date_to", datetime.max)
            query_parts.append(DateRange("timestamp", date_from, date_to))
        
        if "confidence_min" in filters or "confidence_max" in filters:
            conf_min = filters.get("confidence_min", 0.0)
            conf_max = filters.get("confidence_max", 1.0)
            query_parts.append(NumericRange("confidence_score", conf_min, conf_max))
        
        # Combine all query parts
        if query_parts:
            final_query = And(query_parts) if len(query_parts) > 1 else query_parts[0]
        else:
            # If no filters, return recent conversations
            final_query = NumericRange("confidence_score", 0.0, 1.0)
        
        # Execute search
        with self.index.searcher() as searcher:
            results = searcher.search(final_query, limit=limit, sortedby="timestamp", reverse=True)
            
            search_results = []
            for result in results:
                search_results.append({
                    "id": int(result["id"]),
                    "session_id": result["session_id"],
                    "user_query": result["user_query"],
                    "ai_response": result["ai_response"],
                    "timestamp": result["timestamp"],
                    "topic_category": result["topic_category"],
                    "confidence_score": result["confidence_score"],
                    "model_used": result["model_used"],
                    "relevance_score": result.score,
                    "search_type": "advanced"
                })
            
            return search_results


# Global search engine instance
search_engine = SearchEngine()
