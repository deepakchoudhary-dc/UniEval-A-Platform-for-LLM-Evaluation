"""
Evaluation Module

This module provides comprehensive LLM evaluation capabilities using Opik and other frameworks.
"""

from .opik_evaluator import (
    OpikEvaluator,
    evaluate_chatbot_response,
    create_evaluation_report,
    OPIK_AVAILABLE
)

__all__ = [
    "OpikEvaluator",
    "evaluate_chatbot_response", 
    "create_evaluation_report",
    "OPIK_AVAILABLE"
]
