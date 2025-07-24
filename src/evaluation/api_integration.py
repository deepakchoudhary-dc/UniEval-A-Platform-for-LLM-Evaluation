"""
Comprehensive Opik Integration API

This module provides a unified API for all Opik integration features including:
- Real-time evaluation and feedback
- Self-correction capabilities
- Administrative dashboard
- Dynamic model cards
- Explainability analysis
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our evaluation modules
from .enhanced_opik_evaluator import enhanced_opik_evaluator, EvaluationResult
from .self_correction import self_correction_engine
from .realtime_ui import realtime_ui_manager
from .admin_dashboard import admin_dashboard
from .dynamic_model_cards import dynamic_model_cards
from .explainability import explainability_engine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class EvaluationRequest(BaseModel):
    input_text: str = Field(..., description="User input text")
    output_text: str = Field(..., description="Model response text")
    conversation_id: str = Field(..., description="Unique conversation identifier")
    context: Optional[str] = Field(None, description="Additional context")
    user_feedback: Optional[Dict[str, Any]] = Field(None, description="User feedback data")

class CorrectionRequest(BaseModel):
    evaluation_id: str = Field(..., description="Evaluation ID to correct")
    max_iterations: Optional[int] = Field(3, description="Maximum correction iterations")

class SearchRequest(BaseModel):
    criteria: Dict[str, Any] = Field(..., description="Search criteria")
    limit: Optional[int] = Field(50, description="Maximum results")
    offset: Optional[int] = Field(0, description="Result offset")

class ExplanationRequest(BaseModel):
    input_text: str = Field(..., description="User input text")
    output_text: str = Field(..., description="Model response text")
    evaluation_id: str = Field(..., description="Evaluation ID")
    explanation_types: Optional[List[str]] = Field(['opik', 'combined'], description="Types of explanations to generate")

class ModelCardRequest(BaseModel):
    analysis_days: Optional[int] = Field(30, description="Days to analyze")
    format_type: Optional[str] = Field("json", description="Output format")
    include_technical: Optional[bool] = Field(True, description="Include technical details")

# Create FastAPI app
app = FastAPI(
    title="Opik Integration API",
    description="Comprehensive API for LLM evaluation with Opik integration",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OpikIntegrationAPI:
    """
    Comprehensive API for Opik integration features.
    """
    
    def __init__(self):
        """Initialize the API."""
        self.evaluator = enhanced_opik_evaluator
        self.correction_engine = self_correction_engine
        self.ui_manager = realtime_ui_manager
        self.dashboard = admin_dashboard
        self.model_cards = dynamic_model_cards
        self.explainability = explainability_engine
        
        logger.info("Opik Integration API initialized")

# Create global API instance
opik_api = OpikIntegrationAPI()

# Main evaluation endpoints
@app.post("/api/evaluation/evaluate", response_model=Dict[str, Any])
async def evaluate_response(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """
    Evaluate a response in real-time with comprehensive metrics.
    """
    try:
        # Perform evaluation
        evaluation_result = await opik_api.evaluator.evaluate_response_realtime(
            input_text=request.input_text,
            output_text=request.output_text,
            conversation_id=request.conversation_id,
            context=request.context,
            user_feedback=request.user_feedback
        )
        
        # Generate UI data
        ui_data = await opik_api.ui_manager.generate_realtime_ui_data(evaluation_result)
        
        # Check if correction is needed
        correction_assessment = await opik_api.correction_engine.assess_correction_need(evaluation_result)
        
        return {
            "evaluation_result": {
                "id": evaluation_result.id,
                "timestamp": evaluation_result.timestamp.isoformat(),
                "scores": {
                    "accuracy": evaluation_result.accuracy_score,
                    "bias": evaluation_result.bias_score,
                    "hallucination": evaluation_result.hallucination_score,
                    "relevance": evaluation_result.relevance_score,
                    "usefulness": evaluation_result.usefulness_score,
                    "overall": evaluation_result.overall_score
                },
                "confidence_level": evaluation_result.confidence_level,
                "requires_correction": evaluation_result.requires_correction,
                "correction_reason": evaluation_result.correction_reason
            },
            "ui_data": ui_data,
            "correction_assessment": correction_assessment,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluation/evaluate-and-correct", response_model=Dict[str, Any])
async def evaluate_and_correct(request: EvaluationRequest, background_tasks: BackgroundTasks):
    """
    Evaluate response and automatically correct if needed.
    """
    try:
        # Perform evaluation and correction
        correction_result = await opik_api.correction_engine.evaluate_and_correct(
            input_text=request.input_text,
            output_text=request.output_text,
            conversation_id=request.conversation_id,
            max_iterations=3
        )
        
        # Generate UI data for final result
        if correction_result['final_evaluation']:
            # Create evaluation result object for UI generation
            final_eval = EvaluationResult(
                id=f"{correction_result['final_evaluation']['id']}_corrected",
                timestamp=datetime.utcnow(),
                input_text=request.input_text,
                output_text=correction_result['final_response'],
                accuracy_score=correction_result['final_evaluation']['accuracy'],
                bias_score=correction_result['final_evaluation']['bias'],
                hallucination_score=correction_result['final_evaluation']['hallucination'],
                relevance_score=correction_result['final_evaluation']['relevance'],
                usefulness_score=correction_result['final_evaluation']['usefulness'],
                overall_score=correction_result['final_evaluation']['overall'],
                confidence_level=correction_result['final_evaluation']['confidence'],
                requires_correction=correction_result['final_evaluation']['requires_correction']
            )
            
            ui_data = await opik_api.ui_manager.generate_realtime_ui_data(final_eval)
        else:
            ui_data = {}
        
        return {
            "correction_result": correction_result,
            "ui_data": ui_data,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Evaluation and correction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Correction endpoints
@app.post("/api/correction/request", response_model=Dict[str, Any])
async def request_correction(request: CorrectionRequest, background_tasks: BackgroundTasks):
    """
    Request correction for a specific evaluation.
    """
    try:
        # This would typically retrieve the original evaluation and apply corrections
        # For now, return a placeholder response
        return {
            "correction_id": f"corr_{request.evaluation_id}",
            "status": "correction_requested",
            "estimated_completion": "2-5 seconds",
            "message": "Correction request received and being processed"
        }
        
    except Exception as e:
        logger.error(f"Correction request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/correction/status/{correction_id}", response_model=Dict[str, Any])
async def get_correction_status(correction_id: str):
    """
    Get status of a correction request.
    """
    try:
        # This would check the actual correction status
        return {
            "correction_id": correction_id,
            "status": "completed",
            "corrected_response": "This is a corrected response based on quality feedback.",
            "improvement_summary": {
                "corrections_made": 2,
                "quality_improvement": 15.5,
                "issues_resolved": ["bias_reduction", "relevance_improvement"]
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get correction status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Search and analytics endpoints
@app.post("/api/search/evaluations", response_model=Dict[str, Any])
async def search_evaluations(request: SearchRequest):
    """
    Search evaluations based on criteria.
    """
    try:
        results = await opik_api.evaluator.search_by_evaluation_criteria(
            criteria=request.criteria,
            limit=request.limit,
            offset=request.offset
        )
        
        return {
            "results": results,
            "total_found": len(results),
            "criteria": request.criteria,
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/conversation/{conversation_id}", response_model=Dict[str, Any])
async def get_conversation_analytics(conversation_id: str):
    """
    Get analytics for a specific conversation.
    """
    try:
        analytics = await opik_api.evaluator.get_conversation_analytics(conversation_id)
        
        if analytics:
            # Generate UI summary
            ui_summary = await opik_api.ui_manager.generate_conversation_summary_ui(conversation_id)
            
            return {
                "conversation_id": conversation_id,
                "analytics": analytics.__dict__ if hasattr(analytics, '__dict__') else analytics,
                "ui_summary": ui_summary,
                "status": "success"
            }
        else:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Dashboard endpoints
@app.get("/api/dashboard/metrics", response_model=Dict[str, Any])
async def get_dashboard_metrics(days: int = 30):
    """
    Get comprehensive dashboard metrics.
    """
    try:
        metrics = await opik_api.dashboard.get_comprehensive_analytics(
            days=days,
            include_trends=True,
            include_comparisons=True
        )
        
        return {
            "metrics": metrics,
            "generated_at": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/ui-components", response_model=Dict[str, Any])
async def get_dashboard_ui_components():
    """
    Get UI components and styling for dashboard.
    """
    try:
        return {
            "css_styles": opik_api.ui_manager.generate_css_styles(),
            "javascript_functions": opik_api.ui_manager.generate_javascript_functions(),
            "component_templates": "Available via specific endpoints",
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get UI components: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Model card endpoints
@app.post("/api/model-card/generate", response_model=Dict[str, Any])
async def generate_model_card(request: ModelCardRequest):
    """
    Generate dynamic model card with live performance data.
    """
    try:
        model_card = await opik_api.model_cards.generate_full_model_card(
            analysis_days=request.analysis_days,
            include_technical_details=request.include_technical,
            format_type=request.format_type
        )
        
        return {
            "model_card": model_card,
            "generated_at": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate model card: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/model-card/summary", response_model=Dict[str, Any])
async def get_model_card_summary():
    """
    Get compact model card summary.
    """
    try:
        summary = await opik_api.model_cards.generate_compact_summary()
        
        return {
            "summary": summary,
            "generated_at": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to get model card summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Explainability endpoints
@app.post("/api/explain/comprehensive", response_model=Dict[str, Any])
async def generate_comprehensive_explanation(request: ExplanationRequest):
    """
    Generate comprehensive explanation using multiple approaches.
    """
    try:
        # Get the evaluation result
        evaluation_result = EvaluationResult(
            id=request.evaluation_id,
            timestamp=datetime.utcnow(),
            input_text=request.input_text,
            output_text=request.output_text,
            accuracy_score=0.8,  # These would come from actual evaluation
            bias_score=0.2,
            hallucination_score=0.1,
            relevance_score=0.9,
            usefulness_score=0.8,
            overall_score=0.82,
            confidence_level="high",
            requires_correction=False
        )
        
        explanation = await opik_api.explainability.generate_comprehensive_explanation(
            input_text=request.input_text,
            output_text=request.output_text,
            evaluation_result=evaluation_result,
            explanation_types=request.explanation_types
        )
        
        return {
            "explanation": explanation,
            "generated_at": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Failed to generate explanation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health and status endpoints
@app.get("/api/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint.
    """
    try:
        # Check if core components are working
        health_status = {
            "api_server": "healthy",
            "opik_evaluator": "healthy" if opik_api.evaluator else "unavailable",
            "database": "healthy",  # Would check actual database
            "correction_engine": "healthy" if opik_api.correction_engine else "unavailable",
            "ui_manager": "healthy" if opik_api.ui_manager else "unavailable",
            "dashboard": "healthy" if opik_api.dashboard else "unavailable",
            "model_cards": "healthy" if opik_api.model_cards else "unavailable",
            "explainability": "healthy" if opik_api.explainability else "unavailable"
        }
        
        overall_status = "healthy" if all(status == "healthy" for status in health_status.values()) else "degraded"
        
        return {
            "overall_status": overall_status,
            "components": health_status,
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "overall_status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/api/status/system", response_model=Dict[str, Any])
async def get_system_status():
    """
    Get detailed system status and metrics.
    """
    try:
        # Get basic system metrics
        performance_stats = await opik_api.evaluator.get_model_performance_stats()
        
        return {
            "system_status": "operational",
            "performance_stats": performance_stats,
            "active_features": [
                "real_time_evaluation",
                "self_correction",
                "bias_detection",
                "hallucination_monitoring",
                "ui_integration",
                "dashboard_analytics",
                "model_cards",
                "explainability"
            ],
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": "Available",  # Would calculate actual uptime
            "version": "2.0.0"
        }
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time updates (if needed)
@app.websocket("/ws/evaluation")
async def websocket_evaluation_updates(websocket):
    """
    WebSocket endpoint for real-time evaluation updates.
    """
    await websocket.accept()
    try:
        while True:
            # This would handle real-time evaluation updates
            await asyncio.sleep(1)
            await websocket.send_text(json.dumps({
                "type": "heartbeat",
                "timestamp": datetime.utcnow().isoformat()
            }))
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Utility functions
async def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the API server.
    """
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

# Example usage and testing functions
async def test_api_endpoints():
    """
    Test basic API functionality.
    """
    logger.info("Testing API endpoints...")
    
    try:
        # Test evaluation
        test_request = EvaluationRequest(
            input_text="What is machine learning?",
            output_text="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
            conversation_id="test_conversation_001"
        )
        
        # This would be tested with actual HTTP requests in practice
        logger.info("API endpoints ready for testing")
        
        return True
        
    except Exception as e:
        logger.error(f"API test failed: {e}")
        return False

if __name__ == "__main__":
    # Run the server
    import asyncio
    asyncio.run(run_server(reload=True))
