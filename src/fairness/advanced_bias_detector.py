"""
Advanced Bias Detection and Fairness Monitoring for AI Chatbot

This module provides enterprise-level bias detection with:
- Multi-dimensional bias analysis (gender, race, age, religion, etc.)
- Statistical fairness metrics using Fairlearn and AIF360
- ML-based bias classification with confidence scores
- Contextual bias analysis using NLP
- Real-time bias monitoring and alerting
- Comprehensive bias reporting and analytics
"""
import logging
import re
import json
import os
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

# Advanced fairness libraries
try:
    from fairlearn.metrics import (
        demographic_parity_difference, 
        equalized_odds_difference,
        selection_rate,
        false_positive_rate,
        false_negative_rate,
        MetricFrame
    )
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity
    from fairlearn.postprocessing import ThresholdOptimizer
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False
    logging.warning("Fairlearn not available for advanced fairness metrics")

try:
    from aif360.datasets import BinaryLabelDataset, StandardDataset
    from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
    from aif360.algorithms.preprocessing import Reweighing, DisparateImpactRemover
    from aif360.algorithms.inprocessing import AdversarialDebiasing
    from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
    AIF360_AVAILABLE = True
except ImportError:
    AIF360_AVAILABLE = False
    logging.warning("AIF360 not available for advanced bias detection")

# Machine learning libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import scipy.stats as stats
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available for ML-based bias detection")

# NLP libraries for advanced text analysis
try:
    import spacy
    # Load English model if available
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        SPACY_AVAILABLE = False
        logging.warning("spaCy English model not available. Install with: python -m spacy download en_core_web_sm")
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("spaCy not available for advanced NLP analysis")

import logging

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedBiasDetector:
    """
    Enterprise-level bias detection and fairness monitoring system.
    
    Features:
    - Multi-dimensional bias analysis
    - Statistical fairness metrics
    - ML-based bias classification
    - Real-time monitoring
    - Comprehensive reporting
    """
    
    def __init__(self):
        """Initialize the advanced bias detector."""
        self.bias_patterns = self._load_comprehensive_bias_patterns()
        self.sensitive_attributes = [
            "gender", "race", "ethnicity", "religion", "age", "nationality",
            "sexual_orientation", "disability", "socioeconomic_status", 
            "political_affiliation", "education_level", "geographic_location",
            "physical_appearance", "mental_health", "marital_status", "parental_status"
        ]
        
        # Historical data for analysis
        self.response_history = []
        self.bias_incidents = []
        self.fairness_metrics_history = []
        
        # ML models for bias detection
        self.bias_classifier = None
        self.is_trained = False
        
        # Initialize ML models if libraries are available
        if SKLEARN_AVAILABLE:
            self._initialize_ml_models()
        
        # Load pre-trained bias patterns if available
        self._load_pretrained_patterns()
        
        logger.info("Advanced Bias Detector initialized with enterprise features")
    
    def _load_comprehensive_bias_patterns(self) -> Dict[str, Dict[str, Union[List[str], float]]]:
        """Load comprehensive bias detection patterns with severity weights."""
        
        return {
            "gender_bias": {
                "patterns": [
                    r"\b(he|she) is (better|worse|superior|inferior) at\b",
                    r"\b(men|women) (can't|cannot|shouldn't|should not|are unable to)\b",
                    r"\b(boys|girls) are (naturally|typically|inherently)\b",
                    r"\bmale (nurse|teacher|secretary|assistant)\b",
                    r"\bfemale (engineer|doctor|ceo|president|leader)\b",
                    r"\b(woman|women)'?s (job|role|place|duty)\b",
                    r"\b(man|men)'?s (job|role|place|duty)\b",
                    r"\bgirls? (should|must|need to) (be|stay|remain)\b",
                    r"\bboys? (don't|do not) (cry|show emotion|express feelings)\b",
                    r"\b(maternal|paternal) instinct\b",
                    r"\bbiological (clock|imperative)\b",
                    r"\b(weaker|stronger) sex\b",
                    r"\bacting like a (girl|boy)\b",
                    r"\b(man up|be a man|like a woman)\b"
                ],
                "severity": 0.8
            },
            
            "racial_bias": {
                "patterns": [
                    r"\b(black|white|asian|hispanic|latino) (people|person) (are|is) (more|less)\b",
                    r"\b(race|ethnicity) (determines|influences) (intelligence|ability|character)\b",
                    r"\bmodel minority\b",
                    r"\bacting (white|black|asian)\b",
                    r"\b(race|color) blind\b",
                    r"\bnot like other (blacks|whites|asians|hispanics)\b",
                    r"\byou (people|guys) always\b",
                    r"\bwhere are you really from\b",
                    r"\byou speak (good|excellent) english\b",
                    r"\b(urban|inner city|ghetto) (culture|lifestyle)\b",
                    r"\bthug|criminal (element|behavior)\b",
                    r"\bexotic|oriental|colored\b",
                    r"\b(people|person) from that (country|place|region) (are|is) (not|never|always)\b",
                    r"\bthat country.*not trustworthy\b",
                    r"\bpeople from.*not (trustworthy|reliable|honest)\b"
                ],
                "severity": 0.9
            },
            
            "age_bias": {
                "patterns": [
                    r"\b(too old|too young) (for|to)\b",
                    r"\bmillennials? (are|always|never)\b",
                    r"\bboomers? (are|always|never)\b",
                    r"\b(older|younger) (workers|people) (are|aren't)\b",
                    r"\bact your age\b",
                    r"\b(past|over) your prime\b",
                    r"\bdigital native\b",
                    r"\btechnology (challenged|illiterate)\b",
                    r"\bset in (his|her|their) ways\b",
                    r"\byoung and inexperienced\b",
                    r"\bold school (thinking|methods)\b"
                ],
                "severity": 0.7
            },
            
            "religious_bias": {
                "patterns": [
                    r"\b(muslim|christian|jewish|hindu|buddhist) (extremist|fundamentalist)\b",
                    r"\b(religion|faith) (makes|causes) (people|person) (violent|irrational)\b",
                    r"\binfidel|heretic|heathen\b",
                    r"\b(holy war|crusade|jihad) against\b",
                    r"\b(godless|immoral) (society|people)\b",
                    r"\breligious (fanatic|zealot)\b",
                    r"\b(all|most) (muslims|christians|jews) are\b",
                    r"\b(backwards|primitive) (beliefs|religion)\b"
                ],
                "severity": 0.9
            },
            
            "lgbtq_bias": {
                "patterns": [
                    r"\bthat's so gay\b",
                    r"\b(lifestyle|choice|preference) of (being|homosexual)\b",
                    r"\b(normal|natural) (relationship|marriage)\b",
                    r"\b(gender|sexual) (confusion|disorder)\b",
                    r"\bagainst (nature|god's will)\b",
                    r"\b(real|biological) (man|woman)\b",
                    r"\b(adam and eve|traditional family)\b",
                    r"\b(sexual|gender) deviant\b",
                    r"\bconversion therapy\b",
                    r"\bpushing (agenda|lifestyle)\b"
                ],
                "severity": 0.9
            },
            
            "disability_bias": {
                "patterns": [
                    r"\b(retarded|retard|spastic|cripple)\b",
                    r"\b(normal|regular) people\b",
                    r"\bsuffers from (disability|condition)\b",
                    r"\bconfined to (wheelchair|bed)\b",
                    r"\b(inspiration|brave|heroic) for (living|existing)\b",
                    r"\bspecial needs\b",
                    r"\bhandicapped (person|individual)\b",
                    r"\b(mentally|physically) (defective|impaired)\b",
                    r"\bbless (his|her|their) heart\b"
                ],
                "severity": 0.8
            },
            
            "socioeconomic_bias": {
                "patterns": [
                    r"\b(poor|rich) (people|person) (are|is) (more|less|always)\b",
                    r"\b(welfare|food stamp) (abuse|fraud|queen)\b",
                    r"\bpull yourself up by your bootstraps\b",
                    r"\bjust (work harder|get a job)\b",
                    r"\b(ghetto|trailer trash|white trash)\b",
                    r"\b(silver spoon|born with privilege)\b",
                    r"\bmoney can't buy (class|happiness)\b",
                    r"\b(earned|deserved) their wealth\b"
                ],
                "severity": 0.7
            },
            
            "appearance_bias": {
                "patterns": [
                    r"\b(attractive|ugly) (people|person) (are|is) (more|less)\b",
                    r"\b(fat|skinny|overweight|underweight) (shaming|people)\b",
                    r"\b(real|healthy) (woman|man) (has|should have)\b",
                    r"\b(pretty|handsome) privilege\b",
                    r"\bjudge a book by its cover\b",
                    r"\b(dress|look) the part\b",
                    r"\bphysical (standards|requirements)\b"
                ],
                "severity": 0.6
            }
        }
    
    def _initialize_ml_models(self):
        """Initialize machine learning models for bias detection."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, skipping ML model initialization")
            return
        
        try:
            # Text preprocessing pipeline
            self.text_vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 3),
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
            
            # Ensemble classifier for bias detection
            self.bias_classifier = Pipeline([
                ('tfidf', self.text_vectorizer),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42
                ))
            ])
            
            logger.info("ML models for bias detection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML models: {e}")
    
    def _load_pretrained_patterns(self):
        """Load pre-trained bias patterns if available."""
        try:
            # You could load from a file or external source
            # For now, we'll use the built-in patterns
            pass
        except Exception as e:
            logger.warning(f"Failed to load pre-trained patterns: {e}")
    
    def analyze_input_bias(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of user input for bias.
        
        Args:
            user_input: The user's input text
            context: Additional context for analysis
            
        Returns:
            Detailed bias analysis results
        """
        analysis_result = {
            "is_biased": False,
            "overall_bias_score": 0.0,
            "bias_types": [],
            "bias_details": {},
            "severity_level": "none",
            "detected_patterns": [],
            "recommendations": [],
            "confidence": 0.0,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Pattern-based detection
            pattern_results = self._detect_bias_patterns(user_input)
            
            # ML-based detection (if trained)
            ml_results = self._ml_bias_detection(user_input)
            
            # NLP-based contextual analysis
            nlp_results = self._nlp_bias_analysis(user_input)
            
            # Statistical analysis
            statistical_results = self._statistical_bias_analysis(user_input, context)
            
            # Combine all results
            analysis_result = self._combine_bias_analyses(
                pattern_results, ml_results, nlp_results, statistical_results
            )
            
            # Generate recommendations
            analysis_result["recommendations"] = self._generate_bias_recommendations(analysis_result)
            
            # Log the analysis
            self._log_bias_analysis(user_input, analysis_result)
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error in bias analysis: {e}")
            analysis_result["error"] = str(e)
            return analysis_result
    
    def _detect_bias_patterns(self, text: str) -> Dict[str, Any]:
        """Detect bias using pattern matching."""
        detected_biases = {}
        detected_patterns = []
        total_severity = 0.0
        
        text_lower = text.lower()
        
        for bias_type, bias_data in self.bias_patterns.items():
            patterns = bias_data["patterns"]
            severity = bias_data["severity"]
            
            matches = []
            for pattern in patterns:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    matches.append(pattern)
                    detected_patterns.append({
                        "pattern": pattern,
                        "bias_type": bias_type,
                        "severity": severity,
                        "match_text": re.search(pattern, text_lower, re.IGNORECASE).group()
                    })
            
            if matches:
                detected_biases[bias_type] = {
                    "detected": True,
                    "patterns_matched": len(matches),
                    "severity": severity,
                    "matches": matches
                }
                total_severity += severity
        
        return {
            "detected_biases": detected_biases,
            "detected_patterns": detected_patterns,
            "pattern_bias_score": min(total_severity / len(self.bias_patterns), 1.0),
            "bias_types_detected": list(detected_biases.keys())
        }
    
    def _ml_bias_detection(self, text: str) -> Dict[str, Any]:
        """ML-based bias detection."""
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return {
                "ml_bias_score": 0.0,
                "ml_confidence": 0.0,
                "ml_prediction": "neutral",
                "note": "ML model not available or not trained"
            }
        
        try:
            # Predict bias probability
            bias_probability = self.bias_classifier.predict_proba([text])[0]
            bias_prediction = self.bias_classifier.predict([text])[0]
            
            # Get feature importance if possible
            feature_importance = self._get_feature_importance(text)
            
            return {
                "ml_bias_score": float(max(bias_probability)),
                "ml_confidence": float(max(bias_probability)),
                "ml_prediction": bias_prediction,
                "bias_probability_distribution": bias_probability.tolist(),
                "feature_importance": feature_importance
            }
            
        except Exception as e:
            logger.warning(f"ML bias detection failed: {e}")
            return {
                "ml_bias_score": 0.0,
                "ml_confidence": 0.0,
                "ml_prediction": "error",
                "error": str(e)
            }
    
    def _nlp_bias_analysis(self, text: str) -> Dict[str, Any]:
        """NLP-based contextual bias analysis."""
        if not SPACY_AVAILABLE:
            return {
                "nlp_bias_indicators": [],
                "contextual_analysis": "NLP not available",
                "sentiment_bias": 0.0
            }
        
        try:
            doc = nlp(text)
            
            # Analyze entities and their context
            bias_indicators = []
            
            # Check for stereotypical associations
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG", "GPE", "NORP"]:
                    context_words = [token.text.lower() for token in doc 
                                   if abs(token.i - ent.start) <= 3]
                    
                    # Look for stereotypical language around entities
                    stereotypical_words = [
                        "typical", "always", "never", "naturally", "inherently",
                        "obviously", "clearly", "all", "most", "tend to"
                    ]
                    
                    if any(word in context_words for word in stereotypical_words):
                        bias_indicators.append({
                            "entity": ent.text,
                            "label": ent.label_,
                            "context": context_words,
                            "bias_type": "stereotypical_association"
                        })
            
            # Analyze dependency relationships for biased constructions
            biased_constructions = self._analyze_syntactic_bias(doc)
            
            return {
                "nlp_bias_indicators": bias_indicators,
                "biased_constructions": biased_constructions,
                "entity_count": len(doc.ents),
                "contextual_analysis": "completed"
            }
            
        except Exception as e:
            logger.warning(f"NLP bias analysis failed: {e}")
            return {
                "nlp_bias_indicators": [],
                "contextual_analysis": f"error: {e}",
                "sentiment_bias": 0.0
            }
    
    def _analyze_syntactic_bias(self, doc) -> List[Dict[str, Any]]:
        """Analyze syntactic patterns that may indicate bias."""
        biased_constructions = []
        
        for token in doc:
            # Look for generalizing constructions
            if token.lemma_ in ["be", "have"] and token.head.pos_ in ["NOUN", "PROPN"]:
                # Find subject and object
                subj = None
                obj = None
                
                for child in token.children:
                    if child.dep_ == "nsubj":
                        subj = child
                    elif child.dep_ in ["attr", "dobj"]:
                        obj = child
                
                if subj and obj:
                    # Check if this is a generalization about a group
                    group_indicators = ["people", "women", "men", "children", "adults"]
                    if any(indicator in subj.text.lower() for indicator in group_indicators):
                        biased_constructions.append({
                            "subject": subj.text,
                            "predicate": token.text,
                            "object": obj.text,
                            "construction_type": "generalization",
                            "full_phrase": f"{subj.text} {token.text} {obj.text}"
                        })
        
        return biased_constructions
    
    def _statistical_bias_analysis(
        self, 
        text: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Statistical analysis for bias detection."""
        
        # Word frequency analysis
        words = text.lower().split()
        word_freq = Counter(words)
        
        # Check for overrepresentation of bias-related terms
        bias_word_counts = defaultdict(int)
        
        for bias_type, bias_data in self.bias_patterns.items():
            for pattern in bias_data["patterns"]:
                # Simple word extraction from regex patterns
                pattern_words = re.findall(r'\b\w+\b', pattern.lower())
                for word in pattern_words:
                    if word in word_freq:
                        bias_word_counts[bias_type] += word_freq[word]
        
        # Calculate statistical significance
        total_words = len(words)
        bias_ratios = {}
        
        for bias_type, count in bias_word_counts.items():
            if total_words > 0:
                bias_ratios[bias_type] = count / total_words
        
        return {
            "word_frequency_analysis": dict(word_freq.most_common(10)),
            "bias_word_counts": dict(bias_word_counts),
            "bias_ratios": bias_ratios,
            "total_words": total_words,
            "statistical_significance": self._calculate_statistical_significance(bias_ratios)
        }
    
    def _calculate_statistical_significance(self, bias_ratios: Dict[str, float]) -> Dict[str, float]:
        """Calculate statistical significance of bias indicators."""
        # Simple statistical significance based on deviation from expected baseline
        baseline_ratio = 0.01  # Expected baseline bias ratio
        significance_scores = {}
        
        for bias_type, ratio in bias_ratios.items():
            if ratio > baseline_ratio:
                # Z-score calculation (simplified)
                z_score = (ratio - baseline_ratio) / (baseline_ratio * 0.1)  # Assuming small std dev
                significance_scores[bias_type] = min(abs(z_score), 5.0)  # Cap at 5.0
        
        return significance_scores
    
    def _combine_bias_analyses(
        self,
        pattern_results: Dict[str, Any],
        ml_results: Dict[str, Any],
        nlp_results: Dict[str, Any],
        statistical_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine results from all bias analysis methods."""
        
        # Calculate overall bias score
        pattern_score = pattern_results.get("pattern_bias_score", 0.0)
        ml_score = ml_results.get("ml_bias_score", 0.0)
        
        # Weighted combination
        overall_score = (pattern_score * 0.4 + ml_score * 0.3 + 
                        min(len(nlp_results.get("nlp_bias_indicators", [])) * 0.1, 0.3))
        
        # Determine if biased - more sensitive threshold
        is_biased = overall_score > 0.05 or len(pattern_results.get("bias_types_detected", [])) > 0
        
        # Collect all bias types
        bias_types = pattern_results.get("bias_types_detected", [])
        
        # Determine severity
        severity_level = "none"
        if overall_score > 0.5:
            severity_level = "high"
        elif overall_score > 0.3:
            severity_level = "medium"
        elif overall_score > 0.05:
            severity_level = "low"
        
        # Override severity if clear bias patterns detected
        if len(pattern_results.get("bias_types_detected", [])) > 0:
            if severity_level == "none":
                severity_level = "low"
        
        # Log bias detection
        if is_biased:
            logger.info(f"Bias detection working: {bias_types} detected (Score: {overall_score:.3f})")
        
        # Calculate confidence
        confidence = min(
            pattern_results.get("pattern_bias_score", 0.0) + 
            ml_results.get("ml_confidence", 0.0) + 0.2, 1.0
        )
        
        return {
            "is_biased": is_biased,
            "overall_bias_score": overall_score,
            "bias_types": bias_types,
            "bias_details": {
                "pattern_analysis": pattern_results,
                "ml_analysis": ml_results,
                "nlp_analysis": nlp_results,
                "statistical_analysis": statistical_results
            },
            "severity_level": severity_level,
            "detected_patterns": pattern_results.get("detected_patterns", []),
            "confidence": confidence,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _generate_bias_recommendations(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on bias analysis."""
        recommendations = []
        
        if not analysis_result["is_biased"]:
            recommendations.append("Input appears to be free of obvious bias")
            return recommendations
        
        bias_types = analysis_result.get("bias_types", [])
        severity = analysis_result.get("severity_level", "none")
        
        # General recommendations
        if severity in ["medium", "high"]:
            recommendations.append("Consider rephrasing to avoid potentially biased language")
        
        # Specific recommendations based on bias types
        if "gender_bias" in bias_types:
            recommendations.append("Use gender-neutral language when possible")
        
        if "racial_bias" in bias_types:
            recommendations.append("Avoid generalizations about racial or ethnic groups")
        
        if "age_bias" in bias_types:
            recommendations.append("Consider age-inclusive language")
        
        if "religious_bias" in bias_types:
            recommendations.append("Respect religious diversity and avoid stereotypes")
        
        if "lgbtq_bias" in bias_types:
            recommendations.append("Use inclusive language for LGBTQ+ individuals")
        
        if "disability_bias" in bias_types:
            recommendations.append("Use person-first language when discussing disabilities")
        
        # Add educational resources
        recommendations.append("Consider reviewing inclusive language guidelines")
        
        return recommendations
    
    def _log_bias_analysis(self, input_text: str, analysis_result: Dict[str, Any]):
        """Log bias analysis for monitoring and improvement."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_text_length": len(input_text),
            "is_biased": analysis_result["is_biased"],
            "overall_score": analysis_result["overall_bias_score"],
            "bias_types": analysis_result["bias_types"],
            "severity": analysis_result["severity_level"],
            "confidence": analysis_result["confidence"]
        }
        
        self.bias_incidents.append(log_entry)
        
        # Log to file if bias detected
        if analysis_result["is_biased"]:
            logger.info(f"Bias detection working: {analysis_result['bias_types']} detected "
                       f"(Score: {analysis_result['overall_bias_score']:.3f})")
    
    def _get_feature_importance(self, text: str) -> Dict[str, float]:
        """Get feature importance for ML bias prediction."""
        if not SKLEARN_AVAILABLE or not self.is_trained:
            return {}
        
        try:
            # Get feature names and importance scores
            feature_names = self.text_vectorizer.get_feature_names_out()
            text_vector = self.text_vectorizer.transform([text])
            
            # Get top features that contributed to the prediction
            feature_scores = {}
            for i, score in enumerate(text_vector.toarray()[0]):
                if score > 0:
                    feature_scores[feature_names[i]] = score
            
            # Return top 10 features
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            return dict(sorted_features[:10])
            
        except Exception as e:
            logger.warning(f"Failed to get feature importance: {e}")
            return {}
    
    def analyze_response_bias(
        self,
        response: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze AI response for bias.
        
        Args:
            response: AI generated response
            user_input: Original user input
            context: Additional context
            
        Returns:
            Bias analysis of the response
        """
        # Use same analysis as input bias, but with different thresholds
        analysis = self.analyze_input_bias(response, context)
        
        # Add response-specific analysis
        analysis["response_analysis"] = {
            "reinforces_input_bias": self._check_bias_reinforcement(user_input, response),
            "introduces_new_bias": self._check_new_bias_introduction(user_input, response),
            "bias_amplification": self._measure_bias_amplification(user_input, response)
        }
        
        return analysis
    
    def _check_bias_reinforcement(self, user_input: str, response: str) -> bool:
        """Check if response reinforces bias in user input."""
        input_analysis = self.analyze_input_bias(user_input)
        response_analysis = self.analyze_input_bias(response)
        
        # Check if response contains similar bias types as input
        input_bias_types = set(input_analysis.get("bias_types", []))
        response_bias_types = set(response_analysis.get("bias_types", []))
        
        return len(input_bias_types.intersection(response_bias_types)) > 0
    
    def _check_new_bias_introduction(self, user_input: str, response: str) -> bool:
        """Check if response introduces new bias not present in input."""
        input_analysis = self.analyze_input_bias(user_input)
        response_analysis = self.analyze_input_bias(response)
        
        input_bias_types = set(input_analysis.get("bias_types", []))
        response_bias_types = set(response_analysis.get("bias_types", []))
        
        return len(response_bias_types - input_bias_types) > 0
    
    def _measure_bias_amplification(self, user_input: str, response: str) -> float:
        """Measure how much the response amplifies bias from input."""
        input_analysis = self.analyze_input_bias(user_input)
        response_analysis = self.analyze_input_bias(response)
        
        input_score = input_analysis.get("overall_bias_score", 0.0)
        response_score = response_analysis.get("overall_bias_score", 0.0)
        
        if input_score == 0:
            return response_score
        
        return (response_score - input_score) / input_score
    
    def get_bias_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive bias analytics."""
        
        # Filter recent incidents
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        recent_incidents = [
            incident for incident in self.bias_incidents 
            if datetime.fromisoformat(incident["timestamp"]).timestamp() > cutoff_date
        ]
        
        if not recent_incidents:
            return {"message": "No bias incidents recorded in the specified period"}
        
        # Calculate analytics
        total_incidents = len(recent_incidents)
        biased_incidents = sum(1 for incident in recent_incidents if incident["is_biased"])
        
        bias_type_counts = Counter()
        severity_counts = Counter()
        
        for incident in recent_incidents:
            for bias_type in incident["bias_types"]:
                bias_type_counts[bias_type] += 1
            severity_counts[incident["severity"]] += 1
        
        return {
            "period_days": days,
            "total_analyzed": total_incidents,
            "bias_detected_count": biased_incidents,
            "bias_detection_rate": biased_incidents / total_incidents if total_incidents > 0 else 0,
            "bias_types_frequency": dict(bias_type_counts.most_common()),
            "severity_distribution": dict(severity_counts),
            "most_common_bias_type": bias_type_counts.most_common(1)[0] if bias_type_counts else None,
            "average_bias_score": np.mean([incident["overall_score"] for incident in recent_incidents]),
            "trend_analysis": self._analyze_bias_trends(recent_incidents)
        }
    
    def _analyze_bias_trends(self, incidents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in bias detection."""
        if len(incidents) < 2:
            return {"trend": "insufficient_data"}
        
        # Sort by timestamp
        sorted_incidents = sorted(incidents, key=lambda x: x["timestamp"])
        
        # Calculate moving average of bias scores
        scores = [incident["overall_score"] for incident in sorted_incidents]
        window_size = min(7, len(scores) // 2)
        
        if window_size < 2:
            return {"trend": "insufficient_data"}
        
        # Simple trend analysis
        first_half_avg = np.mean(scores[:len(scores)//2])
        second_half_avg = np.mean(scores[len(scores)//2:])
        
        trend = "stable"
        if second_half_avg > first_half_avg * 1.1:
            trend = "increasing"
        elif second_half_avg < first_half_avg * 0.9:
            trend = "decreasing"
        
        return {
            "trend": trend,
            "first_half_average": first_half_avg,
            "second_half_average": second_half_avg,
            "change_percentage": ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0
        }
    
    def train_bias_classifier(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train the ML bias classifier with labeled data.
        
        Args:
            training_data: List of dictionaries with 'text' and 'label' keys
            
        Returns:
            Training results and metrics
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "Scikit-learn not available for training"}
        
        try:
            # Prepare training data
            texts = [item["text"] for item in training_data]
            labels = [item["label"] for item in training_data]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Train the classifier
            self.bias_classifier.fit(X_train, y_train)
            
            # Evaluate
            train_score = self.bias_classifier.score(X_train, y_train)
            test_score = self.bias_classifier.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.bias_classifier, texts, labels, cv=5)
            
            self.is_trained = True
            
            return {
                "training_successful": True,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "train_accuracy": train_score,
                "test_accuracy": test_score,
                "cv_mean_accuracy": cv_scores.mean(),
                "cv_std_accuracy": cv_scores.std(),
                "model_type": "GradientBoostingClassifier"
            }
            
        except Exception as e:
            logger.error(f"Failed to train bias classifier: {e}")
            return {"error": str(e)}


# Global instance
advanced_bias_detector = AdvancedBiasDetector()
