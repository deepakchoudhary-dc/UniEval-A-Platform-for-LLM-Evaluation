"""
FastAPI routes for the Transparent AI Chatbot
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Union, List, Dict, Any, Optional
from datetime import datetime
import json
import re
import logging
import time
import uuid

from config.settings import settings
from src.core.chatbot import TransparentChatbot
from src.explainability.model_card import model_card_generator
from src.fairness.bias_detector import bias_detector
from src.utils.logger import get_logger, audit_log

# Setup logger
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Transparent AI Chatbot API",
    description="Enterprise AI Chatbot with Memory, Search, and Explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Global chatbot instance management
chatbot_instances = {}
MAX_INSTANCES = 100  # Prevent memory leaks


# Enhanced Request/Response models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    session_id: Optional[str] = Field(None, description="Session identifier")
    enable_memory_search: bool = Field(True, description="Enable memory search")
    enable_explanation: bool = Field(True, description="Enable AI explainability")
    enable_bias_check: bool = Field(True, description="Enable bias detection")
    simple_response: bool = Field(False, description="Return simplified response")
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        # Remove potential injection patterns
        if re.search(r'<script|javascript:|data:|vbscript:', v, re.IGNORECASE):
            raise ValueError('Invalid message content')
        return v.strip()


class ChatResponse(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    session_id: str
    timestamp: datetime
    sources: Optional[List[str]] = None
    explanation: Optional[Dict[str, Any]] = None
    bias_check: Optional[Dict[str, Any]] = None
    response_time_ms: Optional[int] = None


class SimpleChatResponse(BaseModel):
    answer: str
    confidence: float
    session_id: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    timestamp: datetime
    request_id: Optional[str] = None


# Middleware for request logging and timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    # Log request
    logger.info(f"Request {request_id}: {request.method} {request.url}")
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    try:
        response = await call_next(request)
        process_time = (time.time() - start_time) * 1000
        
        # Log response
        logger.info(f"Request {request_id} completed in {process_time:.2f}ms - Status: {response.status_code}")
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        logger.error(f"Request {request_id} failed in {process_time:.2f}ms: {str(e)}")
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(e),
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id
            }
        )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.warning(f"HTTP {exc.status_code} for request {request_id}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    request_id = getattr(request.state, 'request_id', 'unknown')
    logger.warning(f"Validation error for request {request_id}: {str(exc)}")
    
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_id
        }
    )


def get_or_create_chatbot(session_id: str) -> TransparentChatbot:
    """Get or create chatbot instance with memory management"""
    if session_id not in chatbot_instances:
        # Clean up old instances if we have too many
        if len(chatbot_instances) >= MAX_INSTANCES:
            # Remove oldest instances (simplified cleanup)
            oldest_sessions = list(chatbot_instances.keys())[:10]
            for old_session in oldest_sessions:
                del chatbot_instances[old_session]
            logger.info(f"Cleaned up {len(oldest_sessions)} old chatbot instances")
        
        chatbot_instances[session_id] = TransparentChatbot()
        chatbot_instances[session_id].set_session_id(session_id)
        logger.info(f"Created new chatbot instance for session: {session_id}")
    
    return chatbot_instances[session_id]
    timestamp: datetime
    sources: List[str] = []
    reasoning: Optional[str] = None  # Simplified reasoning summary


class CleanChatResponse(BaseModel):
    answer: str  # Clean, simplified answer without formatting
    confidence: float
    session_id: str
    timestamp: datetime
    sources: List[str] = []
    reasoning: Optional[str] = None
    bias_detected: bool = False
    bias_summary: Optional[str] = None


class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    search_type: str = "comprehensive"
    limit: int = 10


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_results: int
    search_query: str


class ExplanationResponse(BaseModel):
    explanation: Dict[str, Any]
    session_id: str
    available: bool


class MemoryStatsResponse(BaseModel):
    stats: Dict[str, Any]
    session_id: str


class BiasReportResponse(BaseModel):
    report: str
    statistics: Dict[str, Any]


class ModelCardResponse(BaseModel):
    model_card: Dict[str, Any]
    generated_at: datetime


# Helper functions
def get_chatbot(session_id: str) -> TransparentChatbot:
    """Get or create chatbot instance for session"""
    if session_id not in chatbot_instances:
        chatbot_instances[session_id] = TransparentChatbot()
        chatbot_instances[session_id].set_session_id(session_id)
    return chatbot_instances[session_id]


def clean_answer_text(text: str) -> str:
    """Clean answer text by formatting it properly while preserving mathematical content"""
    if not text:
        return ""
    
    # Convert LaTeX math expressions to readable format (PRESERVE CONTENT)
    text = re.sub(r'\\\[(.*?)\\\]', r'\n\n\1\n\n', text, flags=re.DOTALL)  # Block math - preserve content
    text = re.sub(r'\\\((.*?)\\\)', r'\1', text, flags=re.DOTALL)  # Inline math - preserve content
    text = re.sub(r'\$\$(.*?)\$\$', r'\n\n\1\n\n', text, flags=re.DOTALL)  # Double dollar math - preserve content
    text = re.sub(r'\$([^$\n]*?)\$', r'\1', text)  # Single dollar math - preserve content
    
    # Clean LaTeX commands but preserve mathematical content
    text = re.sub(r'\\tau', 'τ', text)           # Replace tau symbol
    text = re.sub(r'\\omega', 'ω', text)         # Replace omega symbol
    text = re.sub(r'\\alpha', 'α', text)         # Replace alpha symbol
    text = re.sub(r'\\times', '×', text)         # Replace times symbol
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', text)  # Convert fractions
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)  # Remove LaTeX commands but keep content
    text = re.sub(r'\\[a-zA-Z]+', '', text)      # Remove remaining LaTeX commands
    text = re.sub(r'\{([^}]*)\}', r'\1', text)   # Remove braces but keep content
      # Clean markdown formatting but preserve important formatting
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)  # Bold italic to plain
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)      # Bold to plain (keep content)
    text = re.sub(r'\*(.*?)\*', r'\1', text)          # Italic to plain (keep content)
    text = re.sub(r'`(.*?)`', r'\1', text)            # Inline code to plain (keep content)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Remove code blocks
    text = re.sub(r'#{1,6}\s*(.+)', r'\1', text)      # Headers - keep content only
    
    # Remove HTML-like tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove URLs and links
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Markdown links
    
    # Convert lists to clean readable format
    text = re.sub(r'^\s*[-*+]\s+(.+)', r'• \1', text, flags=re.MULTILINE)  # Convert bullet points
    text = re.sub(r'^\s*(\d+)\.\s+(.+)', r'\1. \2', text, flags=re.MULTILINE)  # Fix numbered lists
    
    # Clean up special formatting characters
    text = re.sub(r'[\\]{2,}', '', text)              # Multiple backslashes
    text = re.sub(r'[_]{2,}', '', text)               # Multiple underscores
    text = re.sub(r'[|]{2,}', '', text)               # Multiple pipes
    text = re.sub(r'[~]{2,}', '', text)               # Tildes (strikethrough)
    
    # Remove section breaks and formatting artifacts
    text = re.sub(r'###\s*(\d+)\.\s*\*\*([^*]+)\*\*', r'\1. \2', text)  # Fix numbered sections
    text = re.sub(r'###\s*(.+)', r'\1', text)         # Fix section headers
    text = re.sub(r'---+', '', text)                  # Horizontal rules
    
    # Clean up whitespace and newlines
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)    # Multiple newlines to double
    text = re.sub(r'[ \t]+', ' ', text)               # Multiple spaces to single
    text = re.sub(r'\n[ \t]+', '\n', text)            # Remove leading whitespace on lines
    text = re.sub(r'[ \t]+\n', '\n', text)            # Remove trailing whitespace on lines
    text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Remove leading spaces from lines
    
    # Remove escape characters and clean quotes
    text = text.replace('\\n', '\n').replace('\\t', ' ').replace('\\"', '"').replace("\\'", "'")
    text = text.replace('\\\\', '\\')
    
    # Clean up punctuation and spacing
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)      # Remove space before punctuation
    text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)     # Ensure single space after punctuation
    text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)  # Proper sentence spacing
    
    # Remove any remaining formatting artifacts
    text = re.sub(r'^\s*\|\s*', '', text, flags=re.MULTILINE)  # Table pipes
    text = re.sub(r'\s*\|\s*$', '', text, flags=re.MULTILINE)  # Table pipes at end
    text = re.sub(r'\s*\|\s*', ' | ', text)                    # Clean remaining pipes
    
    # Final cleanup - remove extra whitespace
    text = re.sub(r' +', ' ', text)                   # Multiple spaces to single
    text = re.sub(r'\n{3,}', '\n\n', text)            # Limit consecutive newlines
    
    return text.strip()


# API Routes
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Transparent AI Chatbot API",
        "version": "1.0.0",
        "description": "AI Chatbot with Memory, Search, and Explainability",
        "features": [
            "Conversational AI with memory",
            "Semantic search across conversations",
            "LIME/SHAP explanations",
            "Bias detection and fairness assessment",
            "Transparent decision-making"
        ],        "endpoints": {
            "chat": "/chat - Detailed responses with full transparency data",
            "chat/simple": "/chat/simple - Simplified responses with basic explanations",
            "chat/clean": "/chat/clean - Clean, presentable responses without formatting",
            "search": "/search - Search conversation memory",
            "explain": "/explain/{session_id} - Get detailed explanations",
            "stats": "/stats/{session_id} - Memory and performance statistics",
            "bias_report": "/bias-report - Bias detection report",
            "model_card": "/model-card - AI model documentation"
        }
    }


@app.post("/chat", response_model=Union[ChatResponse, SimpleChatResponse])
async def chat(request: ChatRequest, http_request: Request, background_tasks: BackgroundTasks):
    """Enhanced chat endpoint with robust error handling and performance tracking"""
    
    start_time = time.time()
    request_id = getattr(http_request.state, 'request_id', 'unknown')
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{int(datetime.now().timestamp())}"
        
        # Audit log
        audit_log("chat_request", session_id, {
            "message_length": len(request.message),
            "request_id": request_id,
            "enable_explanation": request.enable_explanation,
            "enable_bias_check": request.enable_bias_check
        })
        
        # Get chatbot instance
        chatbot = get_or_create_chatbot(session_id)
        
        # Process chat request with timeout handling
        try:
            response = chatbot.chat(
                user_query=request.message,
                enable_memory_search=request.enable_memory_search,
                enable_explanation=request.enable_explanation,
                enable_bias_check=request.enable_bias_check
            )
        except Exception as chat_error:
            logger.error(f"Chat processing error for request {request_id}: {str(chat_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Chat processing failed: {str(chat_error)}"
            )
        
        # Calculate response time
        response_time_ms = int((time.time() - start_time) * 1000)
        
        # Log successful response
        logger.info(f"Chat completed for session {session_id} in {response_time_ms}ms")
        
        # Return appropriate response format
        if request.simple_response:
            return SimpleChatResponse(
                answer=response.answer,
                confidence=response.confidence,
                session_id=session_id
            )
        else:
            return ChatResponse(
                answer=response.answer,
                confidence=response.confidence,
                session_id=session_id,
                timestamp=response.timestamp,
                sources=response.sources,
                explanation=response.explanation,
                bias_check=getattr(response, 'bias_check_results', None),
                response_time_ms=response_time_ms
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint for request {request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing your request"
        )


@app.post("/chat/simple", response_model=SimpleChatResponse)
async def chat_simple(request: ChatRequest, background_tasks: BackgroundTasks):
    """Simple chat endpoint with clean, user-friendly responses"""
    
    try:
        # Force simple response
        request.simple_response = True
        
        # Use the main chat logic
        response = await chat(request, background_tasks)
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/clean", response_model=CleanChatResponse)
async def chat_clean(request: ChatRequest, background_tasks: BackgroundTasks):
    """Clean chat endpoint with simplified, presentable responses"""
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{datetime.now().timestamp()}"
        
        # Get chatbot instance
        chatbot = get_chatbot(session_id)
        
        # Process chat request
        response = chatbot.chat(
            user_query=request.message,
            enable_memory_search=request.enable_memory_search,
            enable_explanation=request.enable_explanation,
            enable_bias_check=request.enable_bias_check
        )
        
        # Clean the answer text
        clean_answer = clean_answer_text(response.answer)
        
        # Extract simplified reasoning
        reasoning = None
        if response.explanation and response.explanation.get('summary', {}).get('reasoning'):
            raw_reasoning = response.explanation['summary']['reasoning']
            reasoning = raw_reasoning[:150] + "..." if len(raw_reasoning) > 150 else raw_reasoning
        
        # Extract bias information
        bias_detected = False
        bias_summary = None
        if hasattr(response, 'bias_check_results') and response.bias_check_results:
            bias_detected = response.bias_check_results.get('bias_detected', False)
            if bias_detected:
                bias_types = response.bias_check_results.get('bias_types', [])
                severity = response.bias_check_results.get('severity', 'unknown')
                bias_summary = f"Detected {', '.join(bias_types)} bias with {severity} severity"
        
        return CleanChatResponse(
            answer=clean_answer,
            confidence=response.confidence,
            session_id=session_id,
            timestamp=response.timestamp,
            sources=response.sources[:3],  # Limit to top 3 sources
            reasoning=reasoning,
            bias_detected=bias_detected,
            bias_summary=bias_summary
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search conversation memory"""
    
    try:
        # Get chatbot instance
        session_id = request.session_id or "default"
        chatbot = get_chatbot(session_id)
        
        # Perform search
        results = chatbot.search_memory(
            query=request.query,
            search_type=request.search_type,
            limit=request.limit
        )
        
        # Format results for API response
        formatted_results = []
        for result in results:
            formatted_result = {
                "id": result.get("id"),
                "user_query": result.get("user_query", ""),
                "ai_response": result.get("ai_response", ""),
                "timestamp": result.get("timestamp"),
                "relevance_score": result.get("relevance_score", 0.0),
                "search_type": result.get("search_type", "unknown"),
                "topic_category": result.get("topic_category", ""),
                "session_id": result.get("session_id", "")
            }
            formatted_results.append(formatted_result)
        
        return SearchResponse(
            results=formatted_results,
            total_results=len(results),
            search_query=request.query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/{session_id}", response_model=ExplanationResponse)
async def explain(session_id: str):
    """Get explanation for last response in session"""
    
    try:
        chatbot = get_chatbot(session_id)
        explanation = chatbot.explain_last_response()
        
        if explanation:
            explanation_data = {
                "method": explanation.method,
                "confidence": explanation.confidence,
                "key_factors": explanation.key_factors,
                "data_sources": explanation.data_sources,
                "decision_reasoning": explanation.decision_reasoning
            }
            
            return ExplanationResponse(
                explanation=explanation_data,
                session_id=session_id,
                available=True
            )
        else:
            return ExplanationResponse(
                explanation={},
                session_id=session_id,
                available=False
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/{session_id}", response_model=MemoryStatsResponse)
async def get_stats(session_id: str):
    """Get memory and performance statistics"""
    
    try:
        chatbot = get_chatbot(session_id)
        stats = chatbot.get_memory_stats()
        
        return MemoryStatsResponse(
            stats=stats,
            session_id=session_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/bias-report", response_model=BiasReportResponse)
async def get_bias_report():
    """Get bias detection report"""
    
    try:
        report = bias_detector.generate_bias_report()
        statistics = bias_detector.get_bias_statistics()
        
        return BiasReportResponse(
            report=report,
            statistics=statistics
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-card", response_model=ModelCardResponse)
async def get_model_card(include_metrics: bool = True):
    """Get model card information"""
    
    try:
        model_card = model_card_generator.generate_model_card(include_metrics=include_metrics)
        
        return ModelCardResponse(
            model_card=model_card,
            generated_at=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/source-question/{session_id}")
async def ask_source_question(session_id: str, question: str):
    """Ask questions about data sources and reasoning"""
    
    try:
        chatbot = get_chatbot(session_id)
        answer = chatbot.ask_about_source(question)
        
        return {
            "question": question,
            "answer": answer,
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear memory for a session"""
    
    try:
        if session_id in chatbot_instances:
            chatbot_instances[session_id].clear_session_memory()
            return {"message": f"Memory cleared for session {session_id}"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def list_sessions():
    """List active sessions"""
    
    try:
        sessions = []
        for session_id, chatbot in chatbot_instances.items():
            stats = chatbot.get_memory_stats()
            sessions.append({
                "session_id": session_id,
                "conversation_count": stats.get("current_session_conversations", 0),
                "active": True
            })
        
        return {
            "sessions": sessions,
            "total_active_sessions": len(sessions)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "active_sessions": len(chatbot_instances),
            "settings_validated": True
        }
        
        # Check database connection
        try:
            from src.data.database import db_manager
            # Simple query to test database
            test_count = db_manager.db_session.execute("SELECT COUNT(*) FROM conversations").scalar()
            health_status["database"] = "connected"
            health_status["total_conversations"] = test_count
        except Exception as e:
            health_status["database"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check search engine
        try:
            from src.core.search import search_engine
            health_status["search_engine"] = "available"
        except Exception as e:
            health_status["search_engine"] = f"error: {str(e)}"
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Evaluation endpoints
@app.get("/evaluation/summary/{session_id}")
async def get_evaluation_summary(session_id: str, limit: int = 10):
    """
    Get evaluation summary for a session.
    
    Args:
        session_id: Session identifier
        limit: Number of recent evaluations to include
        
    Returns:
        Evaluation summary with scores and trends
    """
    try:
        chatbot = get_chatbot(session_id)
        summary = chatbot.get_evaluation_summary(limit=limit)
        
        audit_log(
            action="get_evaluation_summary",
            session_id=session_id,
            details={"limit": limit}
        )
        
        return {
            "status": "success",
            "data": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting evaluation summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get evaluation summary: {str(e)}"
        )


@app.post("/evaluation/conversation/{session_id}")
async def evaluate_conversation(session_id: str):
    """
    Evaluate the quality of the current conversation.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Conversation-level evaluation results
    """
    try:
        chatbot = get_chatbot(session_id)
        evaluation_result = await chatbot.evaluate_conversation_quality()
        
        audit_log(
            action="evaluate_conversation",
            session_id=session_id,
            details={"evaluation_score": evaluation_result.get("overall_conversation_score", 0)}
        )
        
        return {
            "status": "success",
            "data": evaluation_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error evaluating conversation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate conversation: {str(e)}"
        )


@app.get("/evaluation/database-summary")
async def get_database_evaluation_summary(limit: int = 100):
    """
    Get evaluation summary from database across all sessions.
    
    Args:
        limit: Number of recent evaluations to analyze
        
    Returns:
        Comprehensive evaluation statistics
    """
    try:
        from src.data.database import db_manager
        
        summary = db_manager.get_evaluation_summary(limit=limit)
        
        audit_log(
            action="get_database_evaluation_summary",
            details={"limit": limit}
        )
        
        return {
            "status": "success",
            "data": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting database evaluation summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get database evaluation summary: {str(e)}"
        )


class EvaluationRequest(BaseModel):
    input_text: str = Field(..., min_length=1, max_length=5000)
    output_text: str = Field(..., min_length=1, max_length=10000)
    context: Optional[str] = None
    expected_output: Optional[str] = None
    criteria: Optional[List[str]] = None


@app.post("/evaluation/single")
async def evaluate_single_response(request: EvaluationRequest):
    """
    Evaluate a single response using Opik.
    
    Args:
        request: Evaluation request with input/output texts
        
    Returns:
        Evaluation results and recommendations
    """
    try:
        from src.evaluation.opik_evaluator import OpikEvaluator, evaluate_chatbot_response
        
        evaluator = OpikEvaluator(project_name="api-evaluation")
        
        if not evaluator.is_available():
            return {
                "status": "warning",
                "message": "Opik evaluation not available, using fallback",
                "data": evaluator._fallback_evaluation(request.input_text, request.output_text),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        evaluation_result = await evaluate_chatbot_response(
            evaluator=evaluator,
            user_input=request.input_text,
            bot_response=request.output_text,
            context=request.context
        )
        
        audit_log(
            action="single_response_evaluation",
            details={
                "evaluation_score": evaluation_result.get("overall_score", 0),
                "criteria": request.criteria or []
            }
        )
        
        return {
            "status": "success",
            "data": evaluation_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in single response evaluation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to evaluate response: {str(e)}"
        )


# WebSocket support for real-time chat (optional)
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    
    await websocket.accept()
    chatbot = get_chatbot(session_id)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            user_message = message_data.get("message", "")
            
            if user_message:
                # Process with chatbot
                response = chatbot.chat(user_message)
                
                # Send response
                response_data = {
                    "type": "chat_response",
                    "answer": response.answer,
                    "confidence": response.confidence,
                    "sources": response.sources,
                    "timestamp": response.timestamp.isoformat()
                }
                
                await websocket.send_text(json.dumps(response_data))
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
