"""
SHAP (SHapley Additive exPlanations) integration
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable, Union
import shap
import json
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

from config.settings import settings


class SHAPExplainer:
    """SHAP-based explainability for chatbot responses"""
    
    def __init__(self):
        self.explainer = None
        self.model = None
        self.tokenizer = None
        self.text_pipeline = None
        
        # Initialize text classification pipeline for explanations
        self._init_text_pipeline()
    
    def _init_text_pipeline(self):
        """Initialize text classification pipeline"""
        try:
            # Use a lightweight model for text classification
            self.text_pipeline = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
        except Exception as e:
            print(f"Warning: Could not initialize text pipeline: {e}")
            self.text_pipeline = None
    
    def explain_text_classification(
        self,
        text: str,
        classifier_fn: Callable = None,
        max_evals: int = 500
    ) -> Dict[str, Any]:
        """
        Explain text classification using SHAP
        
        Args:
            text: Input text to explain
            classifier_fn: Custom classifier function, uses default if None
            max_evals: Maximum number of evaluations for SHAP
        """
        
        if classifier_fn is None and self.text_pipeline is None:
            return {
                "method": "shap_text",
                "error": "No text pipeline available",
                "text": text,
                "explanations": []
            }
        
        try:
            # Use provided classifier or default pipeline
            predict_fn = classifier_fn or self._default_text_classifier
            
            # Create SHAP explainer
            explainer = shap.Explainer(predict_fn, masker=shap.maskers.Text())
            
            # Generate SHAP values
            shap_values = explainer([text], max_evals=max_evals)
            
            # Extract explanation data
            explanation_data = {
                "method": "shap_text",
                "text": text,
                "explanations": [],
                "base_value": float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else 0.0,
                "prediction": float(shap_values.values[0].sum() + (shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0))
            }
            
            # Process SHAP values for each token
            if hasattr(shap_values, 'data') and hasattr(shap_values, 'values'):
                tokens = shap_values.data[0]
                values = shap_values.values[0]
                
                for token, value in zip(tokens, values):
                    if token.strip():  # Skip empty tokens
                        explanation_data["explanations"].append({
                            "token": token,
                            "shap_value": float(value),
                            "contribution": "positive" if value > 0 else "negative",
                            "magnitude": abs(float(value))
                        })
                
                # Sort by magnitude
                explanation_data["explanations"].sort(
                    key=lambda x: x["magnitude"], reverse=True
                )
            
            return explanation_data
            
        except Exception as e:
            return {
                "method": "shap_text",
                "error": str(e),
                "text": text,
                "explanations": []
            }
    
    def _default_text_classifier(self, texts: List[str]) -> np.ndarray:
        """Default text classifier using the pipeline"""
        if not self.text_pipeline:
            # Return dummy predictions if no pipeline
            return np.array([[0.5, 0.5] for _ in texts])
        
        predictions = []
        for text in texts:
            try:
                result = self.text_pipeline(text)
                # Convert to probability array [negative, positive]
                if len(result) == 2:
                    neg_score = result[0]['score'] if result[0]['label'] == 'NEGATIVE' else result[1]['score']
                    pos_score = result[1]['score'] if result[1]['label'] == 'POSITIVE' else result[0]['score']
                    predictions.append([neg_score, pos_score])
                else:
                    predictions.append([0.5, 0.5])
            except:
                predictions.append([0.5, 0.5])
        
        return np.array(predictions)
    
    def explain_response_factors(
        self,
        user_query: str,
        context_data: Dict[str, Any],
        response_generator: Callable,
        feature_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Explain factors contributing to response generation
        
        Args:
            user_query: User's input query
            context_data: Context information (memory, preferences, etc.)
            response_generator: Function that generates responses
            feature_names: Names of features to analyze
        """
        
        try:
            # Convert context data to numerical features
            features, feature_names_final = self._extract_numerical_features(
                user_query, context_data, feature_names
            )
            
            if len(features) == 0:
                return {
                    "method": "shap_tabular",
                    "error": "No numerical features extracted",
                    "explanations": []
                }
            
            # Create wrapper function for response quality
            def predict_response_quality(feature_matrix):
                predictions = []
                for feature_row in feature_matrix:
                    try:
                        # Reconstruct context from features
                        reconstructed_context = self._reconstruct_context_from_features(
                            feature_row, feature_names_final, context_data
                        )
                        
                        # Generate response
                        response_data = response_generator(user_query, reconstructed_context)
                        
                        # Extract quality metrics
                        confidence = response_data.get('confidence', 0.5)
                        predictions.append(confidence)
                        
                    except:
                        predictions.append(0.5)
                
                return np.array(predictions)
            
            # Create SHAP explainer for tabular data
            explainer = shap.Explainer(
                predict_response_quality,
                np.array([features])  # Background data
            )
            
            # Generate SHAP values
            shap_values = explainer(np.array([features]))
            
            # Extract explanation data
            explanation_data = {
                "method": "shap_tabular",
                "user_query": user_query,
                "explanations": [],
                "base_value": float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else 0.0,
                "prediction": float(shap_values.values[0].sum() + (shap_values.base_values[0] if hasattr(shap_values, 'base_values') else 0))
            }
            
            # Process SHAP values for each feature
            values = shap_values.values[0] if hasattr(shap_values, 'values') else []
            
            for i, (feature_name, shap_value) in enumerate(zip(feature_names_final, values)):
                explanation_data["explanations"].append({
                    "feature": feature_name,
                    "shap_value": float(shap_value),
                    "contribution": "positive" if shap_value > 0 else "negative",
                    "magnitude": abs(float(shap_value)),
                    "feature_value": features[i]
                })
            
            # Sort by magnitude
            explanation_data["explanations"].sort(
                key=lambda x: x["magnitude"], reverse=True
            )
            
            return explanation_data
            
        except Exception as e:
            return {
                "method": "shap_tabular",
                "error": str(e),
                "explanations": []
            }
    
    def _extract_numerical_features(
        self,
        user_query: str,
        context_data: Dict[str, Any],
        feature_names: List[str] = None
    ) -> tuple:
        """Extract numerical features from context data"""
        
        features = []
        names = []
        
        # Query-based features
        query_length = len(user_query.split())
        query_char_length = len(user_query)
        features.extend([query_length, query_char_length])
        names.extend(["query_word_count", "query_char_count"])
        
        # Context-based features
        for key, value in context_data.items():
            if isinstance(value, (int, float)):
                features.append(float(value))
                names.append(f"context_{key}")
            elif isinstance(value, bool):
                features.append(1.0 if value else 0.0)
                names.append(f"context_{key}")
            elif isinstance(value, str):
                # String length as feature
                features.append(float(len(value)))
                names.append(f"context_{key}_length")
            elif isinstance(value, list):
                # List length as feature
                features.append(float(len(value)))
                names.append(f"context_{key}_count")
            elif isinstance(value, dict):
                # Dict size as feature
                features.append(float(len(value)))
                names.append(f"context_{key}_size")
        
        # Memory-related features
        if "recent_context" in context_data:
            recent_context = context_data["recent_context"]
            if isinstance(recent_context, list):
                features.append(float(len(recent_context)))
                names.append("recent_context_count")
        
        if "similar_memories" in context_data:
            similar_memories = context_data["similar_memories"]
            if isinstance(similar_memories, list):
                features.append(float(len(similar_memories)))
                names.append("similar_memories_count")
                
                # Average similarity score
                if similar_memories:
                    avg_similarity = np.mean([
                        m.get('similarity_score', 0.0) 
                        for m in similar_memories 
                        if isinstance(m, dict)
                    ])
                    features.append(float(avg_similarity))
                    names.append("avg_similarity_score")
        
        # Time-based features (if timestamp available)
        if "timestamp" in context_data:
            try:
                from datetime import datetime
                if isinstance(context_data["timestamp"], datetime):
                    hour = context_data["timestamp"].hour
                    day_of_week = context_data["timestamp"].weekday()
                    features.extend([float(hour), float(day_of_week)])
                    names.extend(["hour_of_day", "day_of_week"])
            except:
                pass
        
        return features, names
    
    def _reconstruct_context_from_features(
        self,
        feature_values: List[float],
        feature_names: List[str],
        original_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reconstruct context dictionary from feature values"""
        
        # Start with original context
        reconstructed = original_context.copy()
        
        # Update with modified feature values
        for feature_name, feature_value in zip(feature_names, feature_values):
            if feature_name.startswith("context_"):
                original_key = feature_name.replace("context_", "").replace("_length", "").replace("_count", "").replace("_size", "")
                
                # Only update if the original key exists and the value type matches
                if original_key in original_context:
                    original_value = original_context[original_key]
                    
                    if isinstance(original_value, (int, float)):
                        reconstructed[original_key] = feature_value
                    elif isinstance(original_value, bool):
                        reconstructed[original_key] = feature_value > 0.5
        
        return reconstructed
    
    def explain_memory_selection(
        self,
        query: str,
        candidate_memories: List[Dict[str, Any]],
        selection_function: Callable
    ) -> Dict[str, Any]:
        """
        Explain why certain memories were selected over others
        
        Args:
            query: Search query
            candidate_memories: List of candidate memories
            selection_function: Function that selects/scores memories
        """
        
        if not candidate_memories:
            return {
                "method": "shap_memory",
                "error": "No candidate memories provided",
                "explanations": []
            }
        
        try:
            # Extract features from memories
            memory_features = []
            feature_names = []
            
            for i, memory in enumerate(candidate_memories):
                features = self._extract_memory_features(memory, query)
                if i == 0:
                    # Initialize feature names from first memory
                    feature_names = list(features.keys())
                memory_features.append(list(features.values()))
            
            if not memory_features:
                return {
                    "method": "shap_memory",
                    "error": "No features extracted from memories",
                    "explanations": []
                }
            
            # Create prediction function
            def predict_memory_selection(feature_matrix):
                predictions = []
                for feature_row in feature_matrix:
                    # Reconstruct memory from features
                    reconstructed_memory = self._reconstruct_memory_from_features(
                        feature_row, feature_names, candidate_memories[0]
                    )
                    
                    try:
                        score = selection_function(query, reconstructed_memory)
                        predictions.append(score)
                    except:
                        predictions.append(0.5)
                
                return np.array(predictions)
            
            # Generate explanations for each memory
            explanations = []
            
            for i, memory_feature_row in enumerate(memory_features):
                try:
                    # Create explainer for this memory
                    explainer = shap.Explainer(
                        predict_memory_selection,
                        np.array([memory_feature_row])
                    )
                    
                    shap_values = explainer(np.array([memory_feature_row]))
                    
                    memory_explanation = {
                        "memory_id": memory.get("id", i),
                        "memory_snippet": memory.get("snippet", "")[:100],
                        "feature_explanations": []
                    }
                    
                    values = shap_values.values[0] if hasattr(shap_values, 'values') else []
                    
                    for j, (feature_name, shap_value) in enumerate(zip(feature_names, values)):
                        memory_explanation["feature_explanations"].append({
                            "feature": feature_name,
                            "shap_value": float(shap_value),
                            "contribution": "positive" if shap_value > 0 else "negative",
                            "magnitude": abs(float(shap_value))
                        })
                    
                    explanations.append(memory_explanation)
                    
                except Exception as e:
                    explanations.append({
                        "memory_id": memory.get("id", i),
                        "error": str(e)
                    })
            
            return {
                "method": "shap_memory",
                "query": query,
                "memory_count": len(candidate_memories),
                "explanations": explanations
            }
            
        except Exception as e:
            return {
                "method": "shap_memory",
                "error": str(e),
                "explanations": []
            }
    
    def _extract_memory_features(self, memory: Dict[str, Any], query: str) -> Dict[str, float]:
        """Extract numerical features from a memory item"""
        
        features = {}
        
        # Basic memory features
        features["similarity_score"] = float(memory.get("similarity_score", 0.0))
        features["confidence"] = float(memory.get("confidence", 0.0))
        
        # Text-based features
        snippet = memory.get("snippet", "")
        features["snippet_length"] = float(len(snippet))
        features["snippet_word_count"] = float(len(snippet.split()))
        
        # Query overlap features
        query_words = set(query.lower().split())
        snippet_words = set(snippet.lower().split())
        overlap = len(query_words.intersection(snippet_words))
        features["query_overlap"] = float(overlap)
        features["query_overlap_ratio"] = float(overlap / max(len(query_words), 1))
        
        # Time-based features
        timestamp = memory.get("timestamp")
        if timestamp:
            try:
                from datetime import datetime
                if isinstance(timestamp, datetime):
                    now = datetime.utcnow()
                    time_diff = (now - timestamp).total_seconds()
                    features["age_hours"] = float(time_diff / 3600)
                    features["is_recent"] = float(time_diff < 86400)  # Less than 24 hours
            except:
                features["age_hours"] = 0.0
                features["is_recent"] = 0.0
        
        # Session-based features
        session_id = memory.get("session_id", "")
        features["same_session"] = float(bool(session_id))
        
        # Topic-based features
        topic = memory.get("topic_category", "")
        features["has_topic"] = float(bool(topic))
        
        return features
    
    def _reconstruct_memory_from_features(
        self,
        feature_values: List[float],
        feature_names: List[str],
        original_memory: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Reconstruct memory from feature values"""
        
        reconstructed = original_memory.copy()
        
        for feature_name, feature_value in zip(feature_names, feature_values):
            if feature_name == "similarity_score":
                reconstructed["similarity_score"] = feature_value
            elif feature_name == "confidence":
                reconstructed["confidence"] = feature_value
        
        return reconstructed
    
    def generate_explanation_summary(
        self,
        explanation_data: Dict[str, Any],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Generate a summary of SHAP explanations"""
        
        method = explanation_data.get("method", "unknown")
        
        summary = {
            "method": method,
            "has_error": "error" in explanation_data,
            "top_factors": [],
            "positive_factors": [],
            "negative_factors": []
        }
        
        if "error" in explanation_data:
            summary["error"] = explanation_data["error"]
            return summary
        
        explanations = explanation_data.get("explanations", [])
        
        if method == "shap_text":
            # Text explanations
            for exp in explanations[:top_k]:
                factor = {
                    "token": exp["token"],
                    "impact": exp["contribution"],
                    "strength": exp["magnitude"]
                }
                summary["top_factors"].append(factor)
                
                if exp["contribution"] == "positive":
                    summary["positive_factors"].append(factor)
                else:
                    summary["negative_factors"].append(factor)
        
        elif method == "shap_tabular":
            # Tabular explanations
            for exp in explanations[:top_k]:
                factor = {
                    "feature": exp["feature"],
                    "impact": exp["contribution"],
                    "strength": exp["magnitude"],
                    "value": exp["feature_value"]
                }
                summary["top_factors"].append(factor)
                
                if exp["contribution"] == "positive":
                    summary["positive_factors"].append(factor)
                else:
                    summary["negative_factors"].append(factor)
        
        elif method == "shap_memory":
            # Memory explanations
            memory_summaries = []
            for memory_exp in explanations:
                if "error" not in memory_exp:
                    memory_summary = {
                        "memory_id": memory_exp["memory_id"],
                        "top_features": memory_exp["feature_explanations"][:3]
                    }
                    memory_summaries.append(memory_summary)
            
            summary["memory_explanations"] = memory_summaries
        
        return summary


# Global SHAP explainer instance
shap_explainer = SHAPExplainer()
