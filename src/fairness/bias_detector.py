"""
Bias Detection and Fairness Assessment for AI Chatbot
"""
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# Try to import fairness libraries (may not be available in all environments)
try:
    from aif360.datasets import StandardDataset
    from aif360.metrics import BinaryLabelDatasetMetric
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    FAIRNESS_LIBS_AVAILABLE = True
except ImportError:
    FAIRNESS_LIBS_AVAILABLE = False
    print("Warning: Advanced fairness libraries not available. Using basic bias detection.")

from config.settings import settings


class BiasDetector:
    """Detect and assess bias in AI chatbot responses"""
    
    def __init__(self):
        # Bias detection patterns and keywords
        self.bias_patterns = self._load_bias_patterns()
        self.sensitive_attributes = [
            "gender", "race", "ethnicity", "religion", "age", "nationality",
            "sexual_orientation", "disability", "socioeconomic_status"
        ]
        # Historical data for fairness assessment
        self.response_history = []
        self.bias_incidents = []
    
    def _load_bias_patterns(self) -> Dict[str, List[str]]:
        """Load bias detection patterns and keywords"""
        
        return {
            "gender_bias": [
                r"\b(he|she) is (better|worse) at\b",
                r"\b(men|women) (can't|cannot|shouldn't)\b",
                r"\b(boys|girls) are (naturally|typically)\b",
                r"\bmale (nurse|teacher|secretary)\b",
                r"\bfemale (engineer|doctor|ceo)\b",
                r"\bnursing is a (woman|women)'?s job\b",
                r"\b(woman|women)'?s job\b",
                r"\b(man|men)'?s job\b",
                r"\bwomen are (emotional|weak|bad at)\b",
                r"\bmen are (strong|logical|good at)\b",
                r"\bgirls can't (handle|understand|do)\b",
                r"\bboys don't (cry|cook|care)\b",
                r"\b(stay|belong) in the kitchen\b",
                r"\bnot suitable for (women|men)\b",
                r"\b(man|men)'?s job\b",
                r"\b(unusual|weird|strange) because.*is a (woman|man)'?s\b",
                r"\b(strong|weak) (men|women) (always|never|typically)\b",
                r"\b(men|women) (always|never|typically|usually) (are|do|have)\b"
            ],
            
            "racial_bias": [
                r"\b(black|white|asian|hispanic) people (are|tend to)\b",
                r"\btypical (african|asian|hispanic)\b",
                r"\b(race|ethnicity) determines\b",
                r"\bcultural (superiority|inferiority)\b"
            ],
              
            "age_bias": [
                r"\b(young|old|older|younger) people (can't|cannot|are)\b",
                r"\b(young|old|older|younger) workers (can't|cannot|are)\b",
                r"\btoo (young|old|older|younger) (to|for)\b",
                r"\b(millennials|boomers) (always|never)\b",
                r"\bage (limits|prevents|determines)\b",
                r"\b(older|younger) workers can't\b",
                r"\b(old|young) people (always|never|can't)\b",
                r"\bcan't learn.*as well as (young|old|older|younger)\b"
            ],
            
            "religious_bias": [
                r"\b(christian|muslim|jewish|hindu|buddhist) people (are|believe)\b",
                r"\breligion (makes|causes|prevents)\b",
                r"\b(believers|non-believers) (can't|cannot)\b"
            ],
            
            "socioeconomic_bias": [
                r"\b(poor|rich) people (are|tend to|always)\b",
                r"\beconomic (status|class) (determines|dictates)\b",
                r"\b(wealth|poverty) (makes|causes) people\b"
            ],
            
            "stereotyping": [
                r"\ball (women|men|people) (are|like|hate)\b",
                r"\bmost (girls|boys|teenagers|adults)\b",
                r"\btypical (person|individual) from\b",
                r"\bgenerally (speaking|true) that\b",
                r"\b(always|never|typically) (bully|dominate|control)\b",
                r"\b(strong|powerful|weak) (people|men|women) (always|never|typically)\b",
                r"\bwhy (do|are) (all|most) (men|women|people)\b"
            ],
            
            "discriminatory_language": [
                r"\b(inferior|superior) (race|gender|group)\b",
                r"\bnot (qualified|capable) because of\b",
                r"\b(naturally|biologically) (better|worse)\b",
                r"\bgenetic (advantage|disadvantage)\b"
            ]
        }
    
    def detect_bias(
        self,
        user_query: str,
        ai_response: str,
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive bias detection in AI response
        
        Args:
            user_query: User's input query
            ai_response: AI's response
            context_data: Context used for response generation
        
        Returns:
            Dictionary containing bias analysis results        """
        
        bias_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": user_query,
            "response": ai_response,
            "bias_detected": False,
            "bias_types": [],
            "bias_score": 0.0,
            "detailed_analysis": {},
            "recommendations": [],
            "severity": "none"
        }
        
        # Pattern-based bias detection on USER QUERY (most important)
        query_pattern_results = self._detect_pattern_bias(user_query)
        bias_results["detailed_analysis"]["query_pattern_detection"] = query_pattern_results
        
        # Pattern-based bias detection on AI RESPONSE
        response_pattern_results = self._detect_pattern_bias(ai_response)
        bias_results["detailed_analysis"]["response_pattern_detection"] = response_pattern_results
          # Language analysis on both query and response
        query_language_results = self._analyze_language_bias(user_query)
        response_language_results = self._analyze_language_bias(ai_response)
        bias_results["detailed_analysis"]["query_language_analysis"] = query_language_results
        bias_results["detailed_analysis"]["response_language_analysis"] = response_language_results
        
        # Context bias analysis
        context_results = self._analyze_context_bias(user_query, ai_response, context_data)
        bias_results["detailed_analysis"]["context_analysis"] = context_results
        
        # Stereotype detection on both query and response
        query_stereotype_results = self._detect_stereotypes(user_query)
        response_stereotype_results = self._detect_stereotypes(ai_response)
        bias_results["detailed_analysis"]["query_stereotype_detection"] = query_stereotype_results
        bias_results["detailed_analysis"]["response_stereotype_detection"] = response_stereotype_results
        
        # Aggregate results from all detection methods
        all_detections = [
            query_pattern_results, response_pattern_results,
            query_language_results, response_language_results,
            context_results,
            query_stereotype_results, response_stereotype_results
        ]
        
        for detection in all_detections:
            if detection.get("bias_detected", False):
                bias_results["bias_detected"] = True
                bias_results["bias_types"].extend(detection.get("bias_types", []))
                bias_results["bias_score"] = max(bias_results["bias_score"], detection.get("bias_score", 0.0))
        
        # Remove duplicate bias types
        bias_results["bias_types"] = list(set(bias_results["bias_types"]))
        
        # Determine severity
        bias_results["severity"] = self._determine_severity(bias_results["bias_score"])
        
        # Generate recommendations
        bias_results["recommendations"] = self._generate_recommendations(bias_results)
        
        # Store for historical analysis        self._store_bias_result(bias_results)
        
        return bias_results
    
    def _detect_pattern_bias(self, text: str) -> Dict[str, Any]:
        """Detect bias using predefined patterns"""
        
        results = {
            "bias_detected": False,
            "bias_types": [],
            "bias_score": 0.0,
            "matches": []
        }
        
        text_lower = text.lower()
        print(f"🔧 DEBUG: Checking text for bias patterns: '{text_lower[:100]}...'")
        
        # Track total potential bias indicators
        total_bias_indicators = 0
        
        for bias_type, patterns in self.bias_patterns.items():
            pattern_matches = []
            for pattern in patterns:
                try:
                    matches = re.findall(pattern, text_lower, re.IGNORECASE)
                    if matches:
                        total_bias_indicators += len(matches)
                        pattern_matches.extend(matches)
                        print(f"🚨 BIAS DETECTED: {bias_type} - Pattern: {pattern} - Matches: {matches}")
                        
                        results["bias_detected"] = True
                        if bias_type not in results["bias_types"]:
                            results["bias_types"].append(bias_type)
                        
                        results["matches"].append({
                            "bias_type": bias_type,
                            "pattern": pattern,
                            "matches": matches,
                            "match_count": len(matches)
                        })
                        
                except re.error as e:
                    print(f"⚠️ Regex error in pattern {pattern}: {e}")
                    continue
            
            # Calculate bias score based on pattern matches
            if pattern_matches:
                # Assign different severity scores to different bias types
                severity_weights = {
                    "gender_bias": 0.8,
                    "racial_bias": 0.9,
                    "age_bias": 0.6,
                    "religious_bias": 0.7,
                    "socioeconomic_bias": 0.6,
                    "stereotyping": 0.7,
                    "discriminatory_language": 0.9,
                    "implicit_bias": 0.8,
                    "profession_bias": 0.6
                }
                weight = severity_weights.get(bias_type, 0.5)
                pattern_score = len(pattern_matches) * weight
                results["bias_score"] = max(results["bias_score"], pattern_score)
        
        # Enhanced keyword-based detection for subtle bias
        bias_keywords = [
            "typical", "natural", "obviously", "of course", "everyone knows",
            "naturally", "biologically", "genetically", "inherently",
            "stereotypical", "traditional", "conventional", "usual"
        ]
        
        sensitive_topics = [
            "women", "men", "girls", "boys", "female", "male",
            "black", "white", "asian", "hispanic", "african", "european",
            "young", "old", "elderly", "teenager", "millennial", "boomer",
            "muslim", "christian", "jewish", "hindu", "buddhist",
            "poor", "rich", "wealthy", "homeless", "upper class", "lower class"
        ]
        
        # Check for combinations of bias keywords with sensitive topics
        for keyword in bias_keywords:
            for topic in sensitive_topics:
                pattern = rf"\b{keyword}.*{topic}\b|\b{topic}.*{keyword}\b"
                if re.search(pattern, text_lower, re.IGNORECASE):
                    total_bias_indicators += 1
                    results["bias_detected"] = True
                    if "contextual_bias" not in results["bias_types"]:
                        results["bias_types"].append("contextual_bias")
                    
                    results["matches"].append({
                        "bias_type": "contextual_bias",
                        "pattern": f"keyword+topic: {keyword} + {topic}",
                        "matches": [f"{keyword}...{topic}"],
                        "match_count": 1
                    })
                    
                    print(f"🚨 CONTEXTUAL BIAS DETECTED: {keyword} + {topic}")
        
        # Adjust bias score based on total indicators
        if total_bias_indicators > 0:
            results["bias_score"] = min(1.0, results["bias_score"] + (total_bias_indicators * 0.1))
        
        print(f"📊 Pattern detection complete. Bias detected: {results['bias_detected']}, Score: {results['bias_score']}")
        return results
        
        results["bias_score"] = min(results["bias_score"], 1.0)
        return results
    
    def _analyze_language_bias(self, text: str) -> Dict[str, Any]:
        """Analyze language for subtle bias indicators"""
        
        results = {
            "bias_detected": False,
            "bias_types": [],
            "bias_score": 0.0,
            "indicators": []
        }
        
        # Check for absolute statements
        absolute_patterns = [
            r"\ball (women|men|people|individuals)\b",
            r"\bevery (woman|man|person)\b",
            r"\balways (true|false) that\b",
            r"\bnever (true|false) that\b"
        ]
        
        for pattern in absolute_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                results["bias_detected"] = True
                results["bias_types"].append("overgeneralization")
                results["bias_score"] += 0.15
                results["indicators"].append(f"Absolute statement detected: {pattern}")
        
        # Check for loaded language
        loaded_words = [
            "naturally", "obviously", "clearly", "inherently",
            "biologically", "genetically", "traditionally"
        ]
        
        found_loaded = [word for word in loaded_words if word in text.lower()]
        if found_loaded:
            results["bias_detected"] = True
            results["bias_types"].append("loaded_language")
            results["bias_score"] += len(found_loaded) * 0.1
            results["indicators"].append(f"Loaded language: {', '.join(found_loaded)}")
        
        # Check for exclusionary language
        exclusionary_patterns = [
            r"\b(only|just) (men|women|people like)\b",
            r"\bnot (suitable|appropriate) for\b",
            r"\b(can't|cannot) (because|due to) (their|his|her)\b"
        ]
        
        for pattern in exclusionary_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                results["bias_detected"] = True
                results["bias_types"].append("exclusionary_language")
                results["bias_score"] += 0.2
                results["indicators"].append(f"Exclusionary language detected")
        
        results["bias_score"] = min(results["bias_score"], 1.0)
        return results
    
    def _analyze_context_bias(
        self,
        user_query: str,
        ai_response: str,
        context_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze bias in context usage and response generation"""
        
        results = {
            "bias_detected": False,
            "bias_types": [],
            "bias_score": 0.0,
            "issues": []
        }
        
        # Check if response reinforces query bias
        if self._contains_bias_indicators(user_query):
            if self._response_reinforces_bias(user_query, ai_response):
                results["bias_detected"] = True
                results["bias_types"].append("bias_reinforcement")
                results["bias_score"] += 0.3
                results["issues"].append("Response reinforces biased assumptions in query")
        
        # Check memory context for bias patterns
        similar_memories = context_data.get("similar_memories", [])
        biased_memories = 0
        
        for memory in similar_memories:
            memory_text = memory.get("snippet", "")
            if self._contains_bias_indicators(memory_text):
                biased_memories += 1
        
        if biased_memories > 0 and len(similar_memories) > 0:
            bias_ratio = biased_memories / len(similar_memories)
            if bias_ratio > 0.3:  # More than 30% of memories show bias
                results["bias_detected"] = True
                results["bias_types"].append("context_bias")
                results["bias_score"] += bias_ratio * 0.2
                results["issues"].append(f"Context memories show bias ({bias_ratio:.1%})")
        
        return results
    
    def _detect_stereotypes(self, text: str) -> Dict[str, Any]:
        """Detect stereotypical assumptions in text"""
        
        results = {
            "bias_detected": False,
            "bias_types": [],
            "bias_score": 0.0,
            "stereotypes": []
        }
        
        # Common stereotypes (simplified detection)
        stereotype_patterns = {
            "gender_roles": [
                r"\b(women|girls) (love|like|prefer) (shopping|cooking|cleaning)\b",
                r"\b(men|boys) (love|like|prefer) (sports|cars|technology)\b",
                r"\b(mothers|fathers) (should|must|need to)\b"
            ],
            "profession_stereotypes": [
                r"\b(male|female) (nurse|teacher|engineer|doctor)\b",
                r"\b(typical|usual) (programmer|secretary|manager)\b"
            ],
            "cultural_stereotypes": [
                r"\b(asians|africans|europeans) (are good at|excel in)\b",
                r"\b(cultural|ethnic) (tendency|trait|characteristic)\b"
            ]
        }
        
        for stereotype_type, patterns in stereotype_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    results["bias_detected"] = True
                    results["bias_types"].append(stereotype_type)
                    results["bias_score"] += 0.2
                    results["stereotypes"].append({
                        "type": stereotype_type,
                        "pattern": pattern
                    })
        
        results["bias_score"] = min(results["bias_score"], 1.0)
        return results
    
    def _contains_bias_indicators(self, text: str) -> bool:
        """Check if text contains bias indicators"""
        
        bias_keywords = [
            "gender", "race", "ethnic", "religion", "age", "typical",
            "naturally", "inherently", "biologically", "all women",
            "all men", "people like", "those people"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in bias_keywords)
    
    def _response_reinforces_bias(self, query: str, response: str) -> bool:
        """Check if response reinforces bias present in query"""
        
        # Simple heuristic: if query contains bias indicators and response doesn't
        # challenge or provide balanced view, it might be reinforcing bias
        
        if not self._contains_bias_indicators(query):
            return False
        
        # Check if response provides balanced perspective
        balance_indicators = [
            "however", "on the other hand", "it's important to note",
            "varies greatly", "depends on", "individual differences",
            "stereotype", "generalization", "not all"
        ]
        
        response_lower = response.lower()
        has_balance = any(indicator in response_lower for indicator in balance_indicators)
        
        # If no balance indicators and response agrees with potentially biased query
        agreement_indicators = ["yes", "indeed", "exactly", "that's right", "correct"]
        has_agreement = any(indicator in response_lower for indicator in agreement_indicators)
        
        return has_agreement and not has_balance
    
    def _determine_severity(self, bias_score: float) -> str:
        """Determine bias severity level"""
        
        if bias_score >= 0.7:
            return "high"
        elif bias_score >= 0.4:
            return "medium"
        elif bias_score >= 0.1:
            return "low"
        else:
            return "none"
    
    def _generate_recommendations(self, bias_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on bias detection results"""
        
        recommendations = []
        
        if not bias_results["bias_detected"]:
            recommendations.append("No bias detected. Continue monitoring for fairness.")
            return recommendations
        
        severity = bias_results["severity"]
        bias_types = bias_results["bias_types"]
        
        if severity == "high":
            recommendations.append("HIGH PRIORITY: Significant bias detected. Review and modify response immediately.")
        elif severity == "medium":
            recommendations.append("MEDIUM PRIORITY: Moderate bias detected. Consider revising response.")
        else:
            recommendations.append("LOW PRIORITY: Minor bias indicators detected. Monitor for patterns.")
        
        # Specific recommendations by bias type
        if "gender_bias" in bias_types:
            recommendations.append("Use gender-neutral language and avoid gender stereotypes.")
        
        if "racial_bias" in bias_types:
            recommendations.append("Avoid racial generalizations and promote inclusive language.")
        
        if "overgeneralization" in bias_types:
            recommendations.append("Replace absolute statements with qualified language (e.g., 'some', 'many', 'often').")
        
        if "loaded_language" in bias_types:
            recommendations.append("Use neutral, objective language instead of loaded terms.")
        
        if "stereotype_detection" in bias_types:
            recommendations.append("Challenge stereotypes by highlighting individual differences and diversity.")
        
        if "bias_reinforcement" in bias_types:
            recommendations.append("Provide balanced perspectives and challenge biased assumptions in queries.")
        
        recommendations.append("Consider alternative phrasings that are more inclusive.")
        recommendations.append("Review training data and model responses for systematic bias patterns.")
        
        return recommendations
    
    def calculate_fairness_score(self, bias_results: Dict[str, Any]) -> float:
        """Calculate overall fairness score"""
        
        if not bias_results["bias_detected"]:
            return 1.0
        
        bias_score = bias_results["bias_score"]
        severity = bias_results["severity"]
        
        # Calculate fairness as inverse of bias
        base_fairness = 1.0 - bias_score
        
        # Apply severity penalty
        severity_penalties = {
            "high": 0.3,
            "medium": 0.15,
            "low": 0.05,
            "none": 0.0
        }
        
        penalty = severity_penalties.get(severity, 0.0)
        fairness_score = max(0.0, base_fairness - penalty)
        
        return fairness_score
    
    def _store_bias_result(self, bias_results: Dict[str, Any]):
        """Store bias detection results for historical analysis"""
        
        # Add to history (in production, this would go to database)
        self.response_history.append({
            "timestamp": bias_results["timestamp"],
            "bias_detected": bias_results["bias_detected"],
            "bias_score": bias_results["bias_score"],
            "bias_types": bias_results["bias_types"],
            "severity": bias_results["severity"]
        })
        
        if bias_results["bias_detected"]:
            self.bias_incidents.append(bias_results)
        
        # Keep only recent history (last 1000 responses)
        if len(self.response_history) > 1000:
            self.response_history = self.response_history[-1000:]
        
        if len(self.bias_incidents) > 100:
            self.bias_incidents = self.bias_incidents[-100:]
    
    def get_bias_statistics(self) -> Dict[str, Any]:
        """Get statistics about bias detection over time"""
        
        if not self.response_history:
            return {
                "total_responses": 0,
                "bias_detection_rate": 0.0,
                "bias_types_distribution": {},
                "severity_distribution": {},
                "recent_trend": "no_data"
            }
        
        total_responses = len(self.response_history)
        biased_responses = sum(1 for r in self.response_history if r["bias_detected"])
        bias_rate = biased_responses / total_responses if total_responses > 0 else 0.0
        
        # Bias types distribution
        bias_types_count = Counter()
        for incident in self.bias_incidents:
            bias_types_count.update(incident["bias_types"])
        
        # Severity distribution
        severity_count = Counter(r["severity"] for r in self.response_history if r["bias_detected"])
        
        # Recent trend (last 10 responses vs previous 10)
        recent_trend = "stable"
        if len(self.response_history) >= 20:
            recent_10 = self.response_history[-10:]
            previous_10 = self.response_history[-20:-10]
            
            recent_bias_rate = sum(1 for r in recent_10 if r["bias_detected"]) / 10
            previous_bias_rate = sum(1 for r in previous_10 if r["bias_detected"]) / 10
            
            if recent_bias_rate > previous_bias_rate * 1.2:
                recent_trend = "increasing"
            elif recent_bias_rate < previous_bias_rate * 0.8:
                recent_trend = "decreasing"
        
        return {
            "total_responses": total_responses,
            "bias_detection_rate": bias_rate,
            "bias_types_distribution": dict(bias_types_count),
            "severity_distribution": dict(severity_count),
            "recent_trend": recent_trend,
            "average_bias_score": np.mean([r["bias_score"] for r in self.response_history if r["bias_detected"]]) if self.bias_incidents else 0.0
        }
    
    def generate_bias_report(self) -> str:
        """Generate a human-readable bias assessment report"""
        
        stats = self.get_bias_statistics()
        
        if stats["total_responses"] == 0:
            return "No responses analyzed yet. Bias detection will begin with the first conversation."
        
        report_parts = [
            f"Bias Detection Report",
            f"=" * 25,
            f"Total Responses Analyzed: {stats['total_responses']}",
            f"Bias Detection Rate: {stats['bias_detection_rate']:.1%}",
            f"Recent Trend: {stats['recent_trend']}",
            ""
        ]
        
        if stats["bias_types_distribution"]:
            report_parts.append("Most Common Bias Types:")
            for bias_type, count in sorted(stats["bias_types_distribution"].items(), 
                                         key=lambda x: x[1], reverse=True)[:5]:
                report_parts.append(f"  - {bias_type}: {count} occurrences")
            report_parts.append("")
        
        if stats["severity_distribution"]:
            report_parts.append("Severity Distribution:")
            for severity, count in stats["severity_distribution"].items():
                report_parts.append(f"  - {severity}: {count} cases")
            report_parts.append("")
        
        # Recommendations
        report_parts.append("Recommendations:")
        if stats["bias_detection_rate"] > 0.1:  # More than 10%
            report_parts.append("  - HIGH: Bias detection rate is above 10%. Review training data and model behavior.")
        elif stats["bias_detection_rate"] > 0.05:  # More than 5%
            report_parts.append("  - MEDIUM: Bias detection rate is moderate. Continue monitoring.")
        else:
            report_parts.append("  - LOW: Bias detection rate is low. Maintain current monitoring.")
        
        if stats["recent_trend"] == "increasing":
            report_parts.append("  - ALERT: Bias incidents are increasing. Investigate recent changes.")
        
        report_parts.append("  - Continue regular bias monitoring and assessment.")
        report_parts.append("  - Review and update bias detection patterns regularly.")
        
        return "\n".join(report_parts)


# Global bias detector instance
bias_detector = BiasDetector()
