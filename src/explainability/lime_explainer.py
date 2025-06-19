"""
LIME (Local Interpretable Model-agnostic Explanations) integration
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from lime.lime_text import LimeTextExplainer
from lime.lime_tabular import LimeTabularExplainer
import json

from config.settings import settings


class LIMEExplainer:
    """LIME-based explainability for chatbot responses"""
    
    def __init__(self):
        self.text_explainer = LimeTextExplainer(
            class_names=['negative', 'positive'],
            feature_selection='auto',
            split_expression=r'\W+',
            bow=True
        )
        
        self.tabular_explainer = None
        self.feature_names = []
    
    def explain_text_prediction(
        self,
        text: str,
        predict_fn: Callable,
        num_features: int = 10,
        num_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Explain a text-based prediction using LIME
        
        Args:
            text: Input text to explain
            predict_fn: Function that takes text and returns prediction probabilities
            num_features: Number of features to include in explanation
            num_samples: Number of samples to generate for explanation
        """
        
        try:
            # Generate explanation
            explanation = self.text_explainer.explain_instance(
                text,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples
            )
            
            # Extract explanation data
            explanation_data = {
                "method": "lime_text",
                "text": text,
                "features": [],
                "prediction_score": None,
                "local_prediction": None,
                "intercept": explanation.intercept[1] if hasattr(explanation, 'intercept') else 0.0
            }
            
            # Get feature explanations
            for feature, weight in explanation.as_list():
                explanation_data["features"].append({
                    "feature": feature,
                    "weight": weight,
                    "importance": abs(weight)
                })
            
            # Sort by importance
            explanation_data["features"].sort(key=lambda x: x["importance"], reverse=True)
            
            # Get prediction probability if available
            try:
                local_pred = explanation.local_pred[1] if hasattr(explanation, 'local_pred') else None
                explanation_data["local_prediction"] = local_pred
            except:
                pass
            
            return explanation_data
            
        except Exception as e:
            return {
                "method": "lime_text",
                "error": str(e),
                "text": text,
                "features": []
            }
    
    def explain_response_generation(
        self,
        user_query: str,
        context_features: Dict[str, Any],
        response_generator: Callable,
        num_features: int = 10
    ) -> Dict[str, Any]:
        """
        Explain why a specific response was generated
        
        Args:
            user_query: The user's input query
            context_features: Features extracted from context (memory, etc.)
            response_generator: Function that generates response given inputs
            num_features: Number of features to explain
        """
        
        # Create a wrapper function for LIME
        def predict_response_quality(texts):
            """Predict response quality/relevance for LIME"""
            predictions = []
            
            for text in texts:
                try:
                    # Generate response
                    response_data = response_generator(text, context_features)
                    
                    # Extract quality metrics (confidence, relevance, etc.)
                    confidence = response_data.get('confidence', 0.5)
                    
                    # Return probability distribution [low_quality, high_quality]
                    predictions.append([1 - confidence, confidence])
                    
                except:
                    # Default to neutral prediction if error
                    predictions.append([0.5, 0.5])
            
            return np.array(predictions)
        
        # Generate explanation
        explanation_data = self.explain_text_prediction(
            user_query,
            predict_response_quality,
            num_features=num_features
        )
        
        # Add context feature analysis
        explanation_data["context_analysis"] = self._analyze_context_features(context_features)
        
        return explanation_data
    
    def explain_memory_retrieval(
        self,
        query: str,
        retrieved_memories: List[Dict[str, Any]],
        memory_scorer: Callable
    ) -> Dict[str, Any]:
        """
        Explain why specific memories were retrieved for a query
        
        Args:
            query: The search query
            retrieved_memories: List of retrieved memory items
            memory_scorer: Function that scores memory relevance
        """
        
        # Create wrapper for memory scoring
        def predict_memory_relevance(texts):
            predictions = []
            for text in texts:
                try:
                    relevance_scores = []
                    for memory in retrieved_memories:
                        score = memory_scorer(text, memory)
                        relevance_scores.append(score)
                    
                    avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.5
                    predictions.append([1 - avg_relevance, avg_relevance])
                    
                except:
                    predictions.append([0.5, 0.5])
            
            return np.array(predictions)
        
        explanation_data = self.explain_text_prediction(
            query,
            predict_memory_relevance,
            num_features=8
        )
        
        # Add memory-specific analysis
        explanation_data["memory_analysis"] = {
            "retrieved_count": len(retrieved_memories),
            "top_memories": [
                {
                    "id": mem.get("id"),
                    "snippet": mem.get("snippet", "")[:100],
                    "relevance_score": mem.get("similarity_score", 0.0),
                    "timestamp": mem.get("timestamp", "").isoformat() if hasattr(mem.get("timestamp", ""), 'isoformat') else str(mem.get("timestamp", ""))
                }
                for mem in retrieved_memories[:3]
            ]
        }
        
        return explanation_data
    
    def _analyze_context_features(self, context_features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the importance of context features"""
        
        analysis = {
            "feature_count": len(context_features),
            "important_features": [],
            "feature_types": {}
        }
        
        # Categorize features by type
        for key, value in context_features.items():
            feature_type = type(value).__name__
            if feature_type not in analysis["feature_types"]:
                analysis["feature_types"][feature_type] = 0
            analysis["feature_types"][feature_type] += 1
            
            # Identify important features (heuristic-based)
            importance_score = self._calculate_feature_importance(key, value)
            
            if importance_score > 0.3:  # Threshold for importance
                analysis["important_features"].append({
                    "name": key,
                    "type": feature_type,
                    "importance": importance_score,
                    "value_summary": self._summarize_feature_value(value)
                })
        
        # Sort by importance
        analysis["important_features"].sort(key=lambda x: x["importance"], reverse=True)
        
        return analysis
    
    def _calculate_feature_importance(self, key: str, value: Any) -> float:
        """Calculate importance score for a context feature"""
        
        importance = 0.0
        
        # Key-based importance
        important_keys = [
            "user_query", "recent_context", "similar_memories",
            "confidence", "model_used", "timestamp"
        ]
        
        for important_key in important_keys:
            if important_key.lower() in key.lower():
                importance += 0.3
                break
        
        # Value-based importance
        if isinstance(value, (list, dict)):
            # Collections with more items are more important
            length = len(value)
            importance += min(length / 10.0, 0.5)
        elif isinstance(value, str):
            # Longer strings are more important
            length = len(value)
            importance += min(length / 200.0, 0.3)
        elif isinstance(value, (int, float)):
            # Higher numeric values might be more important
            if value > 0.7:  # For confidence scores, etc.
                importance += 0.2
        
        return min(importance, 1.0)
    
    def _summarize_feature_value(self, value: Any) -> str:
        """Create a summary of a feature value"""
        
        if isinstance(value, dict):
            return f"Dict with {len(value)} keys: {list(value.keys())[:3]}"
        elif isinstance(value, list):
            return f"List with {len(value)} items"
        elif isinstance(value, str):
            return value[:50] + "..." if len(value) > 50 else value
        else:
            return str(value)
    
    def generate_human_readable_explanation(
        self,
        explanation_data: Dict[str, Any],
        detail_level: str = "medium"
    ) -> str:
        """
        Generate human-readable explanation from LIME results
        
        Args:
            explanation_data: LIME explanation data
            detail_level: "low", "medium", or "high"
        """
        
        if "error" in explanation_data:
            return f"Could not generate explanation: {explanation_data['error']}"
        
        explanation_parts = []
        
        # Basic explanation
        method = explanation_data.get("method", "unknown")
        explanation_parts.append(f"This explanation was generated using {method.upper()} analysis.")
        
        # Feature analysis
        features = explanation_data.get("features", [])
        if features:
            if detail_level == "low":
                top_feature = features[0]
                explanation_parts.append(
                    f"The most important factor was '{top_feature['feature']}' "
                    f"with an impact score of {top_feature['weight']:.3f}."
                )
            
            elif detail_level == "medium":
                top_features = features[:3]
                feature_descriptions = []
                for feat in top_features:
                    impact = "positive" if feat['weight'] > 0 else "negative"
                    feature_descriptions.append(
                        f"'{feat['feature']}' ({impact} impact: {abs(feat['weight']):.3f})"
                    )
                
                explanation_parts.append(
                    f"The top factors influencing this response were: {', '.join(feature_descriptions)}."
                )
            
            elif detail_level == "high":
                explanation_parts.append("Detailed factor analysis:")
                for i, feat in enumerate(features[:5], 1):
                    impact = "positive" if feat['weight'] > 0 else "negative"
                    explanation_parts.append(
                        f"{i}. '{feat['feature']}': {impact} impact of {abs(feat['weight']):.3f}"
                    )
        
        # Context analysis
        context_analysis = explanation_data.get("context_analysis", {})
        if context_analysis and detail_level in ["medium", "high"]:
            important_features = context_analysis.get("important_features", [])
            if important_features:
                explanation_parts.append(
                    f"\nKey context factors: {', '.join([f['name'] for f in important_features[:3]])}."
                )
        
        # Memory analysis
        memory_analysis = explanation_data.get("memory_analysis", {})
        if memory_analysis and detail_level in ["medium", "high"]:
            retrieved_count = memory_analysis.get("retrieved_count", 0)
            if retrieved_count > 0:
                explanation_parts.append(
                    f"\nThis response used information from {retrieved_count} previous conversation(s)."
                )
        
        # Confidence information
        local_pred = explanation_data.get("local_prediction")
        if local_pred is not None and detail_level == "high":
            explanation_parts.append(f"\nModel confidence for this explanation: {local_pred:.3f}")
        
        return "\n".join(explanation_parts)
    
    def create_explanation_summary(
        self,
        explanation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a concise summary of the explanation"""
        
        summary = {
            "method": explanation_data.get("method", "unknown"),
            "top_factors": [],
            "confidence": explanation_data.get("local_prediction", 0.0),
            "feature_count": len(explanation_data.get("features", [])),
            "has_context": "context_analysis" in explanation_data,
            "has_memory": "memory_analysis" in explanation_data
        }
        
        # Extract top factors
        features = explanation_data.get("features", [])
        for feat in features[:3]:
            summary["top_factors"].append({
                "name": feat["feature"],
                "impact": "positive" if feat["weight"] > 0 else "negative",
                "strength": abs(feat["weight"])
            })
        
        return summary


# Global LIME explainer instance
lime_explainer = LIMEExplainer()
