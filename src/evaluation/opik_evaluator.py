"""
Opik LLM Evaluation Module

This module provides comprehensive LLM evaluation using Opik framework.
Features include response quality evaluation, hallucination detection,
bias assessment, and performance monitoring.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json

# Set up basic logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import opik
    from opik import Opik
    from opik.evaluation import evaluate
    from opik.evaluation.metrics import (
        AnswerRelevance,
        Hallucination, 
        Moderation, 
        Sentiment,
        Usefulness,
        GEval
    )
    import opik.evaluation.metrics as metrics
    OPIK_AVAILABLE = True
    logger.info("Opik successfully imported")
except ImportError as e:
    OPIK_AVAILABLE = False
    logger.warning(f"Opik not available: {e}")
except Exception as e:
    OPIK_AVAILABLE = False
    logger.warning(f"Opik import error: {e}")

# Try to import the custom logger, fallback to basic logger
try:
    from ..utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    pass  # Keep the basic logger


class OpikEvaluator:
    """
    Comprehensive LLM evaluation using Opik framework.
    
    Provides evaluation for:
    - Response quality and relevance
    - Hallucination detection
    - Content moderation
    - Bias assessment
    - Performance metrics
    """
    
    def __init__(self, project_name: str = "transparent-ai-chatbot"):
        """
        Initialize Opik evaluator.
        
        Args:
            project_name: Name of the Opik project for tracking evaluations
        """
        self.project_name = project_name
        self.client = None
        self.evaluation_results = []
        
        if OPIK_AVAILABLE:
            try:
                self.client = Opik(project_name=project_name)
                logger.info(f"Opik evaluator initialized for project: {project_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Opik client: {e}")
                self.client = None
        else:
            logger.warning("Opik not available. Evaluation features will be limited.")
    
    def is_available(self) -> bool:
        """Check if Opik evaluation is available."""
        return OPIK_AVAILABLE and self.client is not None
    
    async def evaluate_response(
        self,
        input_text: str,
        output_text: str,
        context: Optional[str] = None,
        expected_output: Optional[str] = None,
        evaluation_criteria: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation of LLM response.
        
        Args:
            input_text: User input/query
            output_text: LLM generated response
            context: Additional context used for generation
            expected_output: Expected/reference output for comparison
            evaluation_criteria: Specific criteria to evaluate
            
        Returns:
            Dict containing evaluation results and metrics
        """
        if not self.is_available():
            return self._fallback_evaluation(input_text, output_text)
        
        try:
            evaluation_result = {
                "timestamp": datetime.now().isoformat(),
                "input": input_text,
                "output": output_text,
                "context": context,
                "metrics": {},
                "overall_score": 0.0,
                "recommendations": []
            }
            
            # Define evaluation metrics
            evaluation_metrics = []
            
            # Answer relevance evaluation
            if "relevance" in (evaluation_criteria or ["relevance"]):
                relevance_metric = AnswerRelevance()
                evaluation_metrics.append(relevance_metric)
            
            # Hallucination detection
            if "hallucination" in (evaluation_criteria or ["hallucination"]):
                hallucination_metric = Hallucination()
                evaluation_metrics.append(hallucination_metric)
            
            # Content moderation
            if "moderation" in (evaluation_criteria or ["moderation"]):
                moderation_metric = Moderation()
                evaluation_metrics.append(moderation_metric)
            
            # Faithfulness to context (replaced with sentiment analysis)
            if context and "sentiment" in (evaluation_criteria or ["sentiment"]):
                sentiment_metric = Sentiment()
                evaluation_metrics.append(sentiment_metric)
            
            # Accuracy if expected output provided (replaced with usefulness)
            if expected_output and "usefulness" in (evaluation_criteria or ["usefulness"]):
                usefulness_metric = Usefulness()
                evaluation_metrics.append(usefulness_metric)
            
            # Prepare evaluation data
            evaluation_data = [{
                "input": input_text,
                "output": output_text,
                "expected_output": expected_output or "",
                "context": context or ""
            }]
            
            # Run evaluation
            if evaluation_metrics:
                evaluation_results = evaluate(
                    dataset=evaluation_data,
                    metrics=evaluation_metrics,
                    task="chat_evaluation"
                )
                
                # Process results
                for metric_name, metric_result in evaluation_results.items():
                    if hasattr(metric_result, 'score'):
                        evaluation_result["metrics"][metric_name] = {
                            "score": metric_result.score,
                            "details": getattr(metric_result, 'details', {})
                        }
            
            # Calculate overall score
            scores = [m.get("score", 0) for m in evaluation_result["metrics"].values()]
            evaluation_result["overall_score"] = sum(scores) / len(scores) if scores else 0.0
            
            # Generate recommendations
            evaluation_result["recommendations"] = self._generate_recommendations(
                evaluation_result["metrics"]
            )
            
            # Store evaluation result
            self.evaluation_results.append(evaluation_result)
            
            # Log to Opik
            await self._log_to_opik(evaluation_result)
            
            logger.info(f"Response evaluation completed. Overall score: {evaluation_result['overall_score']:.3f}")
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error during Opik evaluation: {e}")
            return self._fallback_evaluation(input_text, output_text)
    
    def evaluate_response_sync(
        self,
        input_text: str,
        output_text: str,
        context: Optional[str] = None,
        expected_output: Optional[str] = None,
        evaluation_criteria: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous wrapper for evaluate_response.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.evaluate_response(
                input_text=input_text,
                output_text=output_text,
                context=context,
                expected_output=expected_output,
                evaluation_criteria=evaluation_criteria,
                metadata=metadata
            )
        )
    
    def _fallback_evaluation(self, input_text: str, output_text: str) -> Dict[str, Any]:
        """
        Fallback evaluation when Opik is not available.
        
        Provides basic heuristic-based evaluation.
        """
        logger.info("Using fallback evaluation (Opik not available)")
        
        # Basic heuristic evaluation
        response_length = len(output_text)
        input_length = len(input_text)
        
        # Simple relevance check (keyword overlap)
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        relevance_score = len(input_words.intersection(output_words)) / max(len(input_words), 1)
        
        # Basic quality indicators
        quality_indicators = {
            "has_content": response_length > 10,
            "appropriate_length": 50 <= response_length <= 2000,
            "proper_formatting": "." in output_text or "!" in output_text or "?" in output_text,
            "keyword_relevance": relevance_score > 0.1
        }
        
        quality_score = sum(quality_indicators.values()) / len(quality_indicators)
        
        evaluation_result = {
            "timestamp": datetime.now().isoformat(),
            "input": input_text,
            "output": output_text,
            "metrics": {
                "fallback_quality": {
                    "score": quality_score,
                    "details": quality_indicators
                },
                "fallback_relevance": {
                    "score": relevance_score,
                    "details": {"keyword_overlap_ratio": relevance_score}
                }
            },
            "metric_scores": {
                "quality_score": quality_score,
                "relevance_score": relevance_score,
                "safety_score": 1.0 - 0.0,  # Fallback assumes safe
                "usefulness_score": quality_score
            },
            "overall_score": (quality_score + relevance_score) / 2,
            "response_time_ms": 1.0,  # Fallback processing time
            "evaluation_method": "fallback_heuristic",
            "warnings": ["Using fallback evaluation - Opik not available"],
            "recommendations": [
                "Install Opik for comprehensive evaluation",
                "Consider response length and relevance",
                "Ensure proper formatting and structure"
            ],
            "evaluation_type": "fallback"
        }
        
        return evaluation_result
    
    async def _log_to_opik(self, evaluation_result: Dict[str, Any]) -> None:
        """Log evaluation results to Opik platform."""
        try:
            if self.client:
                # Create trace for evaluation
                trace = self.client.trace(
                    name="response_evaluation",
                    input={"query": evaluation_result["input"]},
                    output={"response": evaluation_result["output"]},
                    metadata={
                        "evaluation_timestamp": evaluation_result["timestamp"],
                        "overall_score": evaluation_result["overall_score"],
                        "metrics": evaluation_result["metrics"]
                    }
                )
                logger.debug("Evaluation logged to Opik successfully")
        except Exception as e:
            logger.warning(f"Failed to log evaluation to Opik: {e}")
    
    def _generate_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate actionable recommendations based on evaluation metrics.
        
        Args:
            metrics: Dictionary of evaluation metrics and scores
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        for metric_name, metric_data in metrics.items():
            score = metric_data.get("score", 0)
            
            if "relevance" in metric_name.lower() and score < 0.7:
                recommendations.append(
                    "Improve answer relevance by ensuring response directly addresses the query"
                )
            
            if "hallucination" in metric_name.lower() and score < 0.8:
                recommendations.append(
                    "Review response for potential hallucinations or unsupported claims"
                )
            
            if "moderation" in metric_name.lower() and score < 0.9:
                recommendations.append(
                    "Content may require moderation review for appropriateness"
                )
            
            if "faithfulness" in metric_name.lower() and score < 0.8:
                recommendations.append(
                    "Ensure response stays faithful to provided context and sources"
                )
            
            if "accuracy" in metric_name.lower() and score < 0.8:
                recommendations.append(
                    "Verify factual accuracy and alignment with expected output"
                )
        
        if not recommendations:
            recommendations.append("Response meets evaluation criteria well")
        
        return recommendations
    
    async def evaluate_conversation_quality(
        self,
        conversation_history: List[Dict[str, str]],
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate overall conversation quality.
        
        Args:
            conversation_history: List of conversation turns
            criteria: Specific evaluation criteria
            
        Returns:
            Conversation-level evaluation results
        """
        if not conversation_history:
            return {"error": "No conversation history provided"}
        
        try:
            conversation_metrics = {
                "turn_count": len(conversation_history),
                "avg_response_length": 0,
                "coherence_score": 0.0,
                "engagement_score": 0.0,
                "quality_trend": "stable"
            }
            
            response_lengths = []
            turn_evaluations = []
            
            # Evaluate each turn
            for i, turn in enumerate(conversation_history):
                user_input = turn.get("user", "")
                assistant_response = turn.get("assistant", "")
                
                if user_input and assistant_response:
                    turn_eval = await self.evaluate_response(
                        user_input, 
                        assistant_response,
                        evaluation_criteria=criteria
                    )
                    turn_evaluations.append(turn_eval["overall_score"])
                    response_lengths.append(len(assistant_response))
            
            # Calculate conversation metrics
            if turn_evaluations:
                conversation_metrics["avg_response_length"] = sum(response_lengths) / len(response_lengths)
                conversation_metrics["coherence_score"] = sum(turn_evaluations) / len(turn_evaluations)
                
                # Simple engagement metric based on response consistency
                if len(turn_evaluations) > 1:
                    score_variance = sum((x - conversation_metrics["coherence_score"]) ** 2 
                                       for x in turn_evaluations) / len(turn_evaluations)
                    conversation_metrics["engagement_score"] = max(0, 1.0 - score_variance)
                
                # Quality trend analysis
                if len(turn_evaluations) >= 3:
                    recent_avg = sum(turn_evaluations[-3:]) / 3
                    early_avg = sum(turn_evaluations[:3]) / 3
                    
                    if recent_avg > early_avg + 0.1:
                        conversation_metrics["quality_trend"] = "improving"
                    elif recent_avg < early_avg - 0.1:
                        conversation_metrics["quality_trend"] = "declining"
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "conversation_metrics": conversation_metrics,
                "turn_evaluations": turn_evaluations,
                "overall_conversation_score": conversation_metrics["coherence_score"],
                "recommendations": self._generate_conversation_recommendations(conversation_metrics)
            }
            
            logger.info(f"Conversation evaluation completed. Score: {result['overall_conversation_score']:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating conversation quality: {e}")
            return {"error": str(e)}
    
    def _generate_conversation_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations for conversation improvement."""
        recommendations = []
        
        if metrics["coherence_score"] < 0.7:
            recommendations.append("Focus on maintaining coherence throughout the conversation")
        
        if metrics["engagement_score"] < 0.6:
            recommendations.append("Work on maintaining consistent response quality")
        
        if metrics["quality_trend"] == "declining":
            recommendations.append("Address declining response quality in longer conversations")
        
        if metrics["avg_response_length"] < 50:
            recommendations.append("Consider providing more detailed responses")
        elif metrics["avg_response_length"] > 500:
            recommendations.append("Consider more concise responses for better engagement")
        
        if not recommendations:
            recommendations.append("Conversation quality is well-maintained")
        
        return recommendations
    
    def get_evaluation_summary(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get summary of recent evaluations.
        
        Args:
            limit: Number of recent evaluations to include
            
        Returns:
            Summary statistics and trends
        """
        recent_evaluations = self.evaluation_results[-limit:] if self.evaluation_results else []
        
        if not recent_evaluations:
            return {"message": "No evaluations available"}
        
        scores = [eval_result["overall_score"] for eval_result in recent_evaluations]
        
        summary = {
            "total_evaluations": len(recent_evaluations),
            "average_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
            "recent_evaluations": recent_evaluations[-5:],  # Last 5 evaluations
            "trends": {
                "improving": sum(1 for s in scores[-5:] if s > 0.8) > 3,
                "stable": abs(scores[-1] - scores[0]) < 0.1 if len(scores) > 1 else True
            }
        }
        
        return summary
    
    async def batch_evaluate(
        self,
        evaluation_batch: List[Dict[str, str]],
        criteria: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Evaluate multiple responses in batch.
        
        Args:
            evaluation_batch: List of evaluation items with input/output pairs
            criteria: Evaluation criteria to apply
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for item in evaluation_batch:
            input_text = item.get("input", "")
            output_text = item.get("output", "")
            context = item.get("context")
            expected_output = item.get("expected_output")
            
            if input_text and output_text:
                result = await self.evaluate_response(
                    input_text=input_text,
                    output_text=output_text,
                    context=context,
                    expected_output=expected_output,
                    evaluation_criteria=criteria
                )
                results.append(result)
        
        logger.info(f"Batch evaluation completed for {len(results)} items")
        return results


# Utility functions for integration
async def evaluate_chatbot_response(
    evaluator: OpikEvaluator,
    user_input: str,
    bot_response: str,
    context: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate a single chatbot response.
    
    Args:
        evaluator: OpikEvaluator instance
        user_input: User's input/query
        bot_response: Chatbot's response
        context: Additional context used
        
    Returns:
        Evaluation results
    """
    return await evaluator.evaluate_response(
        input_text=user_input,
        output_text=bot_response,
        context=context,
        evaluation_criteria=["relevance", "hallucination", "moderation", "faithfulness"]
    )


def create_evaluation_report(evaluation_results: List[Dict[str, Any]]) -> str:
    """
    Create a formatted evaluation report.
    
    Args:
        evaluation_results: List of evaluation results
        
    Returns:
        Formatted report string
    """
    if not evaluation_results:
        return "No evaluation results available."
    
    total_evaluations = len(evaluation_results)
    avg_score = sum(result["overall_score"] for result in evaluation_results) / total_evaluations
    
    report = f"""
LLM Evaluation Report
====================

Total Evaluations: {total_evaluations}
Average Score: {avg_score:.3f}

Recent Evaluations:
"""
    
    for i, result in enumerate(evaluation_results[-5:], 1):
        timestamp = result.get("timestamp", "Unknown")
        score = result.get("overall_score", 0)
        report += f"\n{i}. [{timestamp[:19]}] Score: {score:.3f}"
        
        if result.get("recommendations"):
            report += f"\n   Recommendations: {'; '.join(result['recommendations'][:2])}"
    
    return report
