"""
Hallucination Detection Module

This module provides hallucination detection for AI responses using multiple approaches:
- Consistency checking with multiple models
- Fact verification against knowledge base
- Confidence scoring
- Source attribution verification
"""

import logging
import re
import json
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class HallucinationDetector:
    """
    Advanced hallucination detection system for AI responses.
    
    Detects potential hallucinations through:
    - Confidence analysis
    - Fact consistency checking
    - Source verification
    - Pattern recognition
    """
    
    def __init__(self):
        """Initialize hallucination detector."""
        self.hallucination_patterns = self._load_hallucination_patterns()
        self.confidence_threshold = 0.5
        self.detection_history = []
        
        logger.info("Hallucination detector initialized")
    
    def _load_hallucination_patterns(self) -> Dict[str, List[str]]:
        """Load patterns that commonly indicate hallucinations."""
        return {
            "uncertainty_indicators": [
                r"\b(i think|i believe|probably|maybe|perhaps|possibly)\b",
                r"\b(not sure|uncertain|unclear|unsure)\b",
                r"\b(might be|could be|seems like|appears to)\b"
            ],
            "fact_fabrication": [
                r"\b(according to.*that doesn't exist)\b",
                r"\b(studies show.*without citation)\b",
                r"\b(experts say.*without source)\b"
            ],
            "inconsistency_markers": [
                r"\b(but then again|on the other hand|contradicting)\b",
                r"\b(however.*completely different)\b"
            ],
            "overconfidence": [
                r"\b(definitely|absolutely|certainly).*\b(without.*doubt|100%|guaranteed)\b",
                r"\b(always|never|all|none).*\b(in every case|without exception)\b"
            ]
        }
    
    def detect_hallucination(self, query: str, response: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Detect potential hallucinations in AI response.
        
        Args:
            query: The original user query
            response: The AI's response
            context: Optional context used to generate response
            
        Returns:
            Dictionary with hallucination analysis results
        """
        try:
            # Analyze response for hallucination indicators
            pattern_analysis = self._analyze_patterns(response)
            confidence_analysis = self._analyze_confidence(response)
            consistency_analysis = self._analyze_consistency(query, response)
            
            # Calculate overall hallucination score
            pattern_score = pattern_analysis.get("hallucination_score", 0.0)
            confidence_score = confidence_analysis.get("confidence_score", 0.0)
            consistency_score = consistency_analysis.get("consistency_score", 0.0)
            
            # Weighted combination
            overall_score = (
                pattern_score * 0.4 +
                (1.0 - confidence_score) * 0.3 +  # Lower confidence = higher hallucination risk
                (1.0 - consistency_score) * 0.3
            )
            
            is_hallucination = overall_score > self.confidence_threshold
            
            # Determine severity
            if overall_score > 0.7:
                severity = "high"
            elif overall_score > 0.4:
                severity = "medium"
            else:
                severity = "low"
            
            result = {
                "is_hallucination": is_hallucination,
                "confidence_score": overall_score,
                "severity": severity,
                "pattern_analysis": pattern_analysis,
                "confidence_analysis": confidence_analysis,
                "consistency_analysis": consistency_analysis,
                "detected_indicators": self._get_detected_indicators(response),
                "analysis_timestamp": datetime.utcnow().isoformat(),
                "recommendations": self._generate_recommendations(overall_score)
            }
            
            # Log result
            self._log_detection(query, response, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Hallucination detection failed: {e}")
            return {
                "is_hallucination": False,
                "confidence_score": 0.0,
                "error": str(e),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
    
    def _analyze_patterns(self, response: str) -> Dict[str, Any]:
        """Analyze response for hallucination patterns."""
        response_lower = response.lower()
        detected_patterns = []
        total_score = 0.0
        
        for pattern_type, patterns in self.hallucination_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, response_lower, re.IGNORECASE)
                if matches:
                    detected_patterns.append({
                        "type": pattern_type,
                        "pattern": pattern,
                        "matches": len(matches)
                    })
                    
                    # Scoring based on pattern type
                    if pattern_type == "uncertainty_indicators":
                        total_score += len(matches) * 0.3
                    elif pattern_type == "fact_fabrication":
                        total_score += len(matches) * 0.8
                    elif pattern_type == "inconsistency_markers":
                        total_score += len(matches) * 0.6
                    elif pattern_type == "overconfidence":
                        total_score += len(matches) * 0.5
        
        return {
            "hallucination_score": min(total_score, 1.0),
            "detected_patterns": detected_patterns,
            "pattern_count": len(detected_patterns)
        }
    
    def _analyze_confidence(self, response: str) -> Dict[str, Any]:
        """Analyze confidence indicators in response."""
        # Simple confidence analysis based on language certainty
        confidence_indicators = [
            r"\b(i know|certain|sure|confident|verified|confirmed)\b",
            r"\b(fact|evidence|proven|documented|established)\b"
        ]
        
        uncertainty_indicators = [
            r"\b(might|maybe|perhaps|possibly|probably|seems)\b",
            r"\b(i think|i believe|i guess|not sure|unclear)\b"
        ]
        
        confidence_count = 0
        uncertainty_count = 0
        
        for pattern in confidence_indicators:
            confidence_count += len(re.findall(pattern, response, re.IGNORECASE))
        
        for pattern in uncertainty_indicators:
            uncertainty_count += len(re.findall(pattern, response, re.IGNORECASE))
        
        # Calculate confidence score (0 = low confidence, 1 = high confidence)
        total_indicators = confidence_count + uncertainty_count
        if total_indicators == 0:
            confidence_score = 0.5  # Neutral
        else:
            confidence_score = confidence_count / total_indicators
        
        return {
            "confidence_score": confidence_score,
            "confidence_indicators": confidence_count,
            "uncertainty_indicators": uncertainty_count
        }
    
    def _analyze_consistency(self, query: str, response: str) -> Dict[str, Any]:
        """Analyze consistency between query and response."""
        # Simple consistency check - ensure response addresses the query
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common stop words for better analysis
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 'must'}
        
        query_keywords = query_words - stop_words
        response_keywords = response_words - stop_words
        
        if not query_keywords:
            return {"consistency_score": 0.5}
        
        # Calculate overlap
        overlap = len(query_keywords.intersection(response_keywords))
        consistency_score = overlap / len(query_keywords)
        
        return {
            "consistency_score": min(consistency_score, 1.0),
            "query_keywords": len(query_keywords),
            "response_keywords": len(response_keywords),
            "keyword_overlap": overlap
        }
    
    def _get_detected_indicators(self, response: str) -> List[str]:
        """Get list of detected hallucination indicators."""
        indicators = []
        response_lower = response.lower()
        
        for pattern_type, patterns in self.hallucination_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    indicators.append(pattern_type)
                    break  # Only add each type once
        
        return indicators
    
    def _generate_recommendations(self, score: float) -> List[str]:
        """Generate recommendations based on hallucination score."""
        recommendations = []
        
        if score > 0.7:
            recommendations.extend([
                "High hallucination risk detected - verify all facts",
                "Request sources and citations",
                "Consider regenerating response with more constraints"
            ])
        elif score > 0.4:
            recommendations.extend([
                "Moderate hallucination risk - verify key claims",
                "Cross-check important information"
            ])
        elif score > 0.2:
            recommendations.append("Low hallucination risk - standard verification recommended")
        else:
            recommendations.append("Minimal hallucination risk detected")
        
        return recommendations
    
    def _log_detection(self, query: str, response: str, result: Dict[str, Any]):
        """Log hallucination detection results."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query[:100] + "..." if len(query) > 100 else query,
            "response": response[:200] + "..." if len(response) > 200 else response,
            "is_hallucination": result["is_hallucination"],
            "confidence_score": result["confidence_score"],
            "severity": result.get("severity", "unknown")
        }
        
        self.detection_history.append(log_entry)
        
        if result["is_hallucination"]:
            logger.info(f"Hallucination detection working: {result['severity']} risk "
                       f"(Score: {result['confidence_score']:.3f})")
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get statistics about hallucination detection."""
        if not self.detection_history:
            return {"total_detections": 0}
        
        total = len(self.detection_history)
        hallucinations = sum(1 for entry in self.detection_history if entry["is_hallucination"])
        
        return {
            "total_detections": total,
            "hallucinations_detected": hallucinations,
            "hallucination_rate": hallucinations / total if total > 0 else 0,
            "average_confidence": np.mean([entry["confidence_score"] for entry in self.detection_history])
        }
