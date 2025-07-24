"""
LIME/SHAP Integration with Opik Evaluation Data

This module synthesizes explainability insights from LIME and SHAP with 
Opik evaluation results to provide comprehensive response explanations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import json
import re
import asyncio

# Set up basic logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import lime
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
    logger.info("LIME successfully imported")
except ImportError as e:
    LIME_AVAILABLE = False
    logger.warning(f"LIME not available: {e}")

try:
    import shap
    SHAP_AVAILABLE = True
    logger.info("SHAP successfully imported")
except ImportError as e:
    SHAP_AVAILABLE = False
    logger.warning(f"SHAP not available: {e}")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
    logger.info("Scikit-learn successfully imported")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    logger.warning(f"Scikit-learn not available: {e}")

from .enhanced_opik_evaluator import enhanced_opik_evaluator, EvaluationResult

class ExplainabilityEngine:
    """
    Comprehensive explainability engine that combines LIME/SHAP with Opik evaluation data.
    """
    
    def __init__(self):
        """Initialize the explainability engine."""
        self.evaluator = enhanced_opik_evaluator
        
        # Initialize LIME explainer if available
        self.lime_explainer = None
        if LIME_AVAILABLE:
            self.lime_explainer = LimeTextExplainer(
                class_names=['low_quality', 'high_quality'],
                feature_selection='auto'
            )
        
        # Initialize mock models for demonstration
        self._init_surrogate_models()
        
        logger.info("Explainability engine initialized")
    
    def _init_surrogate_models(self):
        """Initialize surrogate models for explanation generation."""
        if SKLEARN_AVAILABLE:
            # Create simple surrogate models for different quality aspects
            self.quality_model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000, stop_words='english')),
                ('classifier', LogisticRegression())
            ])
            
            self.bias_model = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=500, stop_words='english')),
                ('classifier', LogisticRegression())
            ])
            
            # Train with synthetic data (in production, use real evaluation data)
            self._train_surrogate_models()
        else:
            self.quality_model = None
            self.bias_model = None
    
    def _train_surrogate_models(self):
        """Train surrogate models with synthetic data."""
        try:
            # In production, this would use real evaluation data from the database
            # For now, using synthetic training data
            
            # Quality training data
            quality_texts = [
                "This is a helpful and accurate response that addresses the question directly.",
                "The response provides useful information with relevant examples.",
                "This answer is vague and doesn't really help with the question.",
                "The response is confusing and contains irrelevant information.",
                "Excellent explanation with clear steps and practical advice.",
                "Poor quality response with potential inaccuracies.",
            ]
            quality_labels = [1, 1, 0, 0, 1, 0]  # 1 for high quality, 0 for low quality
            
            # Bias training data
            bias_texts = [
                "People from all backgrounds can succeed in this field.",
                "This approach works well regardless of gender or ethnicity.",
                "Men are naturally better at this type of work than women.",
                "This is typically something that young people understand better.",
                "Everyone has equal potential to learn these skills.",
                "Older workers struggle more with new technology.",
            ]
            bias_labels = [0, 0, 1, 1, 0, 1]  # 1 for biased, 0 for unbiased
            
            # Train models
            self.quality_model.fit(quality_texts, quality_labels)
            self.bias_model.fit(bias_texts, bias_labels)
            
            logger.info("Surrogate models trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train surrogate models: {e}")
    
    async def generate_comprehensive_explanation(
        self,
        input_text: str,
        output_text: str,
        evaluation_result: EvaluationResult,
        explanation_types: List[str] = ['lime', 'shap', 'opik', 'combined']
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation combining multiple approaches.
        
        Args:
            input_text: Original user input
            output_text: Model response
            evaluation_result: Opik evaluation results
            explanation_types: Types of explanations to generate
        """
        try:
            explanations = {
                'metadata': {
                    'input_text': input_text,
                    'output_text': output_text,
                    'evaluation_id': evaluation_result.id,
                    'timestamp': evaluation_result.timestamp.isoformat(),
                    'explanation_types': explanation_types
                },
                'opik_evaluation': self._extract_opik_insights(evaluation_result),
                'explanations': {}
            }
            
            # Generate LIME explanations
            if 'lime' in explanation_types and LIME_AVAILABLE and self.quality_model:
                explanations['explanations']['lime'] = await self._generate_lime_explanations(
                    input_text, output_text, evaluation_result
                )
            
            # Generate SHAP explanations
            if 'shap' in explanation_types:
                explanations['explanations']['shap'] = await self._generate_shap_explanations(
                    input_text, output_text, evaluation_result
                )
            
            # Generate Opik-specific insights
            if 'opik' in explanation_types:
                explanations['explanations']['opik_insights'] = await self._generate_opik_insights(
                    input_text, output_text, evaluation_result
                )
            
            # Generate combined explanation
            if 'combined' in explanation_types:
                explanations['explanations']['combined'] = await self._generate_combined_explanation(
                    explanations['explanations'], evaluation_result
                )
            
            # Generate user-friendly summary
            explanations['user_summary'] = await self._generate_user_friendly_summary(
                explanations['explanations'], evaluation_result
            )
            
            return explanations
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive explanation: {e}")
            return {'error': str(e)}
    
    async def _generate_lime_explanations(
        self,
        input_text: str,
        output_text: str,
        evaluation_result: EvaluationResult
    ) -> Dict[str, Any]:
        """Generate LIME-based explanations."""
        try:
            if not self.lime_explainer or not self.quality_model:
                return {'error': 'LIME not available or models not trained'}
            
            # Generate explanation for response quality
            def quality_prediction_fn(texts):
                """Prediction function for LIME."""
                try:
                    probs = self.quality_model.predict_proba(texts)
                    return probs
                except Exception as e:
                    logger.error(f"Quality prediction failed: {e}")
                    # Return default probabilities
                    return np.array([[0.5, 0.5]] * len(texts))
            
            # Generate LIME explanation
            lime_explanation = self.lime_explainer.explain_instance(
                output_text,
                quality_prediction_fn,
                num_features=10,
                num_samples=100
            )
            
            # Extract feature importance
            features = lime_explanation.as_list()
            
            # Generate bias explanation if model available
            bias_explanation = None
            if self.bias_model:
                def bias_prediction_fn(texts):
                    """Prediction function for bias detection."""
                    try:
                        probs = self.bias_model.predict_proba(texts)
                        return probs
                    except Exception as e:
                        logger.error(f"Bias prediction failed: {e}")
                        return np.array([[0.8, 0.2]] * len(texts))  # Default: low bias
                
                bias_lime = LimeTextExplainer(
                    class_names=['unbiased', 'biased'],
                    mode='classification'
                )
                
                bias_explanation = bias_lime.explain_instance(
                    output_text,
                    bias_prediction_fn,
                    num_features=8,
                    num_samples=50
                )
            
            return {
                'quality_explanation': {
                    'features': features,
                    'prediction_probability': float(quality_prediction_fn([output_text])[0][1]),
                    'top_positive_features': [f for f in features if f[1] > 0][:5],
                    'top_negative_features': [f for f in features if f[1] < 0][:5]
                },
                'bias_explanation': {
                    'features': bias_explanation.as_list() if bias_explanation else [],
                    'bias_indicators': self._extract_bias_indicators(bias_explanation) if bias_explanation else []
                },
                'lime_insights': {
                    'most_influential_words': self._get_most_influential_words(features),
                    'quality_drivers': self._identify_quality_drivers(features),
                    'potential_issues': self._identify_potential_issues(features)
                }
            }
            
        except Exception as e:
            logger.error(f"LIME explanation generation failed: {e}")
            return {'error': str(e)}
    
    async def _generate_shap_explanations(
        self,
        input_text: str,
        output_text: str,
        evaluation_result: EvaluationResult
    ) -> Dict[str, Any]:
        """Generate SHAP-based explanations."""
        try:
            # Since SHAP requires more complex model integration, 
            # we'll provide a structured analysis based on text features
            
            # Analyze text features
            text_features = self._extract_text_features(output_text)
            
            # Mock SHAP values based on evaluation results and text features
            shap_values = self._calculate_mock_shap_values(text_features, evaluation_result)
            
            return {
                'feature_importance': shap_values,
                'text_analysis': {
                    'sentence_contributions': self._analyze_sentence_contributions(output_text, evaluation_result),
                    'word_importance': self._analyze_word_importance(output_text, evaluation_result),
                    'structural_factors': self._analyze_structural_factors(output_text)
                },
                'shap_insights': {
                    'top_contributing_factors': self._get_top_contributing_factors(shap_values),
                    'quality_attribution': self._calculate_quality_attribution(shap_values),
                    'improvement_suggestions': self._generate_improvement_suggestions(shap_values)
                }
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation generation failed: {e}")
            return {'error': str(e)}
    
    async def _generate_opik_insights(
        self,
        input_text: str,
        output_text: str,
        evaluation_result: EvaluationResult
    ) -> Dict[str, Any]:
        """Generate Opik-specific insights and explanations."""
        try:
            return {
                'quality_breakdown': {
                    'accuracy': {
                        'score': evaluation_result.accuracy_score,
                        'explanation': self._explain_accuracy_score(evaluation_result.accuracy_score),
                        'contributing_factors': self._identify_accuracy_factors(input_text, output_text)
                    },
                    'bias': {
                        'score': evaluation_result.bias_score,
                        'explanation': self._explain_bias_score(evaluation_result.bias_score),
                        'detected_patterns': self._identify_bias_patterns(output_text)
                    },
                    'hallucination': {
                        'score': evaluation_result.hallucination_score,
                        'explanation': self._explain_hallucination_score(evaluation_result.hallucination_score),
                        'risk_factors': self._identify_hallucination_risks(output_text)
                    },
                    'relevance': {
                        'score': evaluation_result.relevance_score,
                        'explanation': self._explain_relevance_score(evaluation_result.relevance_score),
                        'alignment_analysis': self._analyze_question_alignment(input_text, output_text)
                    },
                    'usefulness': {
                        'score': evaluation_result.usefulness_score,
                        'explanation': self._explain_usefulness_score(evaluation_result.usefulness_score),
                        'practical_value': self._assess_practical_value(output_text)
                    }
                },
                'opik_recommendations': {
                    'immediate_improvements': self._generate_immediate_improvements(evaluation_result),
                    'long_term_suggestions': self._generate_long_term_suggestions(evaluation_result),
                    'monitoring_points': self._identify_monitoring_points(evaluation_result)
                }
            }
            
        except Exception as e:
            logger.error(f"Opik insights generation failed: {e}")
            return {'error': str(e)}
    
    async def _generate_combined_explanation(
        self,
        explanations: Dict[str, Any],
        evaluation_result: EvaluationResult
    ) -> Dict[str, Any]:
        """Generate a combined explanation synthesizing all approaches."""
        try:
            # Synthesize insights from all explanation methods
            combined_insights = {
                'overall_assessment': {
                    'quality_level': evaluation_result.confidence_level,
                    'main_strengths': [],
                    'main_weaknesses': [],
                    'critical_issues': []
                },
                'detailed_analysis': {
                    'content_quality': self._synthesize_content_quality(explanations),
                    'bias_assessment': self._synthesize_bias_assessment(explanations),
                    'safety_evaluation': self._synthesize_safety_evaluation(explanations),
                    'user_experience': self._synthesize_user_experience(explanations)
                },
                'actionable_insights': {
                    'immediate_actions': [],
                    'improvement_priorities': [],
                    'monitoring_recommendations': []
                }
            }
            
            # Extract strengths and weaknesses
            if evaluation_result.accuracy_score > 0.8:
                combined_insights['overall_assessment']['main_strengths'].append('High accuracy and factual correctness')
            
            if evaluation_result.bias_score < 0.2:
                combined_insights['overall_assessment']['main_strengths'].append('Low bias and inclusive language')
            
            if evaluation_result.relevance_score > 0.8:
                combined_insights['overall_assessment']['main_strengths'].append('Highly relevant to user query')
            
            if evaluation_result.bias_score > 0.6:
                combined_insights['overall_assessment']['main_weaknesses'].append('Detected bias in language or content')
                combined_insights['overall_assessment']['critical_issues'].append('High bias level requires immediate attention')
            
            if evaluation_result.hallucination_score > 0.5:
                combined_insights['overall_assessment']['main_weaknesses'].append('Potential factual inaccuracies')
                combined_insights['overall_assessment']['critical_issues'].append('Hallucination risk needs verification')
            
            # Generate actionable insights
            if evaluation_result.requires_correction:
                combined_insights['actionable_insights']['immediate_actions'].append(
                    f"Address correction need: {evaluation_result.correction_reason}"
                )
            
            # Prioritize improvements based on scores
            improvement_priorities = []
            if evaluation_result.bias_score > 0.4:
                improvement_priorities.append(('Bias reduction', 'high'))
            if evaluation_result.hallucination_score > 0.4:
                improvement_priorities.append(('Fact verification', 'high'))
            if evaluation_result.relevance_score < 0.6:
                improvement_priorities.append(('Relevance improvement', 'medium'))
            if evaluation_result.usefulness_score < 0.6:
                improvement_priorities.append(('Usefulness enhancement', 'medium'))
            
            combined_insights['actionable_insights']['improvement_priorities'] = improvement_priorities
            
            return combined_insights
            
        except Exception as e:
            logger.error(f"Combined explanation generation failed: {e}")
            return {'error': str(e)}
    
    async def _generate_user_friendly_summary(
        self,
        explanations: Dict[str, Any],
        evaluation_result: EvaluationResult
    ) -> Dict[str, Any]:
        """Generate a user-friendly summary of all explanations."""
        try:
            # Create simple, accessible explanations
            summary = {
                'overall_quality': {
                    'rating': evaluation_result.confidence_level,
                    'score': f"{evaluation_result.overall_score * 100:.1f}%",
                    'simple_explanation': self._get_simple_quality_explanation(evaluation_result.overall_score)
                },
                'key_findings': [],
                'what_went_well': [],
                'areas_for_improvement': [],
                'recommended_actions': []
            }
            
            # Add key findings
            if evaluation_result.accuracy_score > 0.8:
                summary['what_went_well'].append("The response is accurate and factually correct")
            
            if evaluation_result.bias_score < 0.2:
                summary['what_went_well'].append("The language used is inclusive and unbiased")
            
            if evaluation_result.relevance_score > 0.8:
                summary['what_went_well'].append("The response directly addresses your question")
            
            if evaluation_result.usefulness_score > 0.8:
                summary['what_went_well'].append("The response provides practical, helpful information")
            
            # Add areas for improvement
            if evaluation_result.bias_score > 0.4:
                summary['areas_for_improvement'].append("The response contains some biased language that should be addressed")
                summary['recommended_actions'].append("Review and revise language to be more inclusive")
            
            if evaluation_result.hallucination_score > 0.4:
                summary['areas_for_improvement'].append("Some information may not be factually accurate")
                summary['recommended_actions'].append("Verify factual claims with reliable sources")
            
            if evaluation_result.relevance_score < 0.6:
                summary['areas_for_improvement'].append("The response doesn't fully address your specific question")
                summary['recommended_actions'].append("Provide a more targeted answer to the specific question asked")
            
            if evaluation_result.usefulness_score < 0.6:
                summary['areas_for_improvement'].append("The response could be more helpful and practical")
                summary['recommended_actions'].append("Add specific examples or actionable advice")
            
            # Add overall recommendation
            if evaluation_result.requires_correction:
                summary['key_findings'].append(f"This response should be improved because: {evaluation_result.correction_reason}")
            else:
                summary['key_findings'].append("This response meets quality standards")
            
            return summary
            
        except Exception as e:
            logger.error(f"User-friendly summary generation failed: {e}")
            return {'error': str(e)}
    
    def _extract_opik_insights(self, evaluation_result: EvaluationResult) -> Dict[str, Any]:
        """Extract key insights from Opik evaluation."""
        return {
            'overall_score': evaluation_result.overall_score,
            'confidence_level': evaluation_result.confidence_level,
            'requires_correction': evaluation_result.requires_correction,
            'correction_reason': evaluation_result.correction_reason,
            'individual_scores': {
                'accuracy': evaluation_result.accuracy_score,
                'bias': evaluation_result.bias_score,
                'hallucination': evaluation_result.hallucination_score,
                'relevance': evaluation_result.relevance_score,
                'usefulness': evaluation_result.usefulness_score
            }
        }
    
    def _extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract features from text for analysis."""
        return {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(text.split('.')),
            'avg_word_length': np.mean([len(word) for word in text.split()]),
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if text else 0,
            'has_numbers': bool(re.search(r'\d', text)),
            'has_urls': bool(re.search(r'http[s]?://', text)),
            'politeness_indicators': sum(1 for word in ['please', 'thank', 'sorry'] if word in text.lower())
        }
    
    def _calculate_mock_shap_values(self, features: Dict[str, Any], evaluation_result: EvaluationResult) -> Dict[str, float]:
        """Calculate mock SHAP values based on features and evaluation."""
        # This would be replaced with actual SHAP calculations in production
        return {
            'text_length': 0.1 * (features['length'] / 1000),
            'word_complexity': 0.05 * features['avg_word_length'],
            'structure_clarity': 0.15 * (1 / max(features['sentence_count'], 1)),
            'politeness': 0.1 * features['politeness_indicators'],
            'accuracy_contribution': 0.3 * evaluation_result.accuracy_score,
            'relevance_contribution': 0.25 * evaluation_result.relevance_score,
            'bias_penalty': -0.2 * evaluation_result.bias_score,
            'hallucination_penalty': -0.15 * evaluation_result.hallucination_score
        }
    
    # Additional helper methods for explanation generation...
    def _get_most_influential_words(self, features: List[Tuple[str, float]]) -> List[str]:
        """Get most influential words from LIME features."""
        return [word for word, importance in sorted(features, key=lambda x: abs(x[1]), reverse=True)[:5]]
    
    def _identify_quality_drivers(self, features: List[Tuple[str, float]]) -> List[str]:
        """Identify factors that drive quality up."""
        positive_features = [word for word, importance in features if importance > 0.1]
        return positive_features[:3]
    
    def _identify_potential_issues(self, features: List[Tuple[str, float]]) -> List[str]:
        """Identify potential issues from features."""
        negative_features = [word for word, importance in features if importance < -0.1]
        return negative_features[:3]
    
    def _extract_bias_indicators(self, bias_explanation) -> List[str]:
        """Extract bias indicators from LIME explanation."""
        if not bias_explanation:
            return []
        
        bias_words = [word for word, importance in bias_explanation.as_list() if importance > 0.1]
        return bias_words[:5]
    
    def _explain_accuracy_score(self, score: float) -> str:
        """Provide explanation for accuracy score."""
        if score > 0.8:
            return "The response demonstrates high accuracy with factual correctness and reliable information."
        elif score > 0.6:
            return "The response is generally accurate but may have some minor factual issues."
        else:
            return "The response has accuracy concerns that need to be addressed."
    
    def _explain_bias_score(self, score: float) -> str:
        """Provide explanation for bias score."""
        if score < 0.2:
            return "The response shows minimal bias and uses inclusive language."
        elif score < 0.5:
            return "The response has some bias indicators that should be reviewed."
        else:
            return "The response contains significant bias that requires correction."
    
    def _explain_hallucination_score(self, score: float) -> str:
        """Provide explanation for hallucination score."""
        if score < 0.2:
            return "The response has low risk of containing false or fabricated information."
        elif score < 0.5:
            return "The response may contain some unverified information that should be checked."
        else:
            return "The response has high risk of containing hallucinated or false information."
    
    def _explain_relevance_score(self, score: float) -> str:
        """Provide explanation for relevance score."""
        if score > 0.8:
            return "The response directly and comprehensively addresses the user's question."
        elif score > 0.6:
            return "The response addresses most aspects of the question but could be more focused."
        else:
            return "The response doesn't adequately address the user's specific question."
    
    def _explain_usefulness_score(self, score: float) -> str:
        """Provide explanation for usefulness score."""
        if score > 0.8:
            return "The response provides practical, actionable information that helps the user."
        elif score > 0.6:
            return "The response is somewhat helpful but could provide more practical value."
        else:
            return "The response lacks practical value and actionable information."
    
    def _get_simple_quality_explanation(self, score: float) -> str:
        """Get simple explanation of overall quality."""
        if score > 0.8:
            return "This is a high-quality response that meets all standards."
        elif score > 0.6:
            return "This is a good response with room for minor improvements."
        elif score > 0.4:
            return "This response is acceptable but has several areas that need improvement."
        else:
            return "This response needs significant improvement before it meets quality standards."
    
    # Placeholder methods for more complex analysis
    def _identify_accuracy_factors(self, input_text: str, output_text: str) -> List[str]:
        """Identify factors contributing to accuracy."""
        return ["Clear factual statements", "Appropriate caveats", "Verifiable claims"]
    
    def _identify_bias_patterns(self, output_text: str) -> List[str]:
        """Identify bias patterns in text."""
        patterns = []
        if any(word in output_text.lower() for word in ['guys', 'mankind']):
            patterns.append("Gender-exclusive language")
        return patterns
    
    def _identify_hallucination_risks(self, output_text: str) -> List[str]:
        """Identify hallucination risk factors."""
        risks = []
        if re.search(r'\d{4}', output_text):  # Years
            risks.append("Specific dates that may be inaccurate")
        if "studies show" in output_text.lower():
            risks.append("Unverified research claims")
        return risks
    
    def _analyze_question_alignment(self, input_text: str, output_text: str) -> Dict[str, Any]:
        """Analyze how well response aligns with question."""
        return {
            "direct_answer": "yes" if any(word in output_text.lower() for word in input_text.lower().split()[:3]) else "partial",
            "completeness": "complete" if len(output_text) > len(input_text) * 2 else "partial"
        }
    
    def _assess_practical_value(self, output_text: str) -> Dict[str, Any]:
        """Assess practical value of response."""
        return {
            "actionable_advice": bool(re.search(r'(should|can|try|consider)', output_text, re.IGNORECASE)),
            "specific_examples": bool(re.search(r'(for example|such as|like)', output_text, re.IGNORECASE)),
            "step_by_step": bool(re.search(r'(first|second|next|then|finally)', output_text, re.IGNORECASE))
        }
    
    # Synthesis methods for combined explanation
    def _synthesize_content_quality(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize content quality assessment."""
        return {"assessment": "Content quality analysis from multiple sources"}
    
    def _synthesize_bias_assessment(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize bias assessment."""
        return {"assessment": "Bias analysis from LIME, SHAP, and Opik"}
    
    def _synthesize_safety_evaluation(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize safety evaluation."""
        return {"assessment": "Safety evaluation from multiple approaches"}
    
    def _synthesize_user_experience(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize user experience assessment."""
        return {"assessment": "User experience analysis"}
    
    # Additional helper methods would be implemented here...
    def _analyze_sentence_contributions(self, text: str, evaluation_result: EvaluationResult) -> List[Dict[str, Any]]:
        """Analyze contribution of each sentence to quality."""
        sentences = text.split('.')
        contributions = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                contributions.append({
                    'sentence': sentence.strip(),
                    'quality_contribution': 0.1 * (i + 1),  # Mock calculation
                    'bias_risk': 0.05,  # Mock calculation
                    'relevance': 0.8  # Mock calculation
                })
        return contributions
    
    def _analyze_word_importance(self, text: str, evaluation_result: EvaluationResult) -> Dict[str, float]:
        """Analyze importance of individual words."""
        words = text.split()
        importance = {}
        for word in set(words):
            # Mock importance calculation
            importance[word] = len(word) / 100.0
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
    
    def _analyze_structural_factors(self, text: str) -> Dict[str, Any]:
        """Analyze structural factors of the text."""
        return {
            'paragraph_structure': 'well_organized' if '\n\n' in text else 'single_block',
            'sentence_variety': 'varied' if len(set(len(s.split()) for s in text.split('.'))) > 2 else 'uniform',
            'formatting': 'basic' if not any(char in text for char in ['*', '-', '1.']) else 'enhanced'
        }
    
    def _get_top_contributing_factors(self, shap_values: Dict[str, float]) -> List[Tuple[str, float]]:
        """Get top contributing factors from SHAP values."""
        return sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    def _calculate_quality_attribution(self, shap_values: Dict[str, float]) -> Dict[str, float]:
        """Calculate quality attribution from SHAP values."""
        positive_contribution = sum(v for v in shap_values.values() if v > 0)
        negative_contribution = sum(v for v in shap_values.values() if v < 0)
        
        return {
            'positive_factors': positive_contribution,
            'negative_factors': abs(negative_contribution),
            'net_contribution': positive_contribution + negative_contribution
        }
    
    def _generate_improvement_suggestions(self, shap_values: Dict[str, float]) -> List[str]:
        """Generate improvement suggestions based on SHAP values."""
        suggestions = []
        
        if shap_values.get('bias_penalty', 0) < -0.1:
            suggestions.append("Review language for potential bias and use more inclusive terms")
        
        if shap_values.get('hallucination_penalty', 0) < -0.1:
            suggestions.append("Verify factual claims and add appropriate disclaimers")
        
        if shap_values.get('structure_clarity', 0) < 0.1:
            suggestions.append("Improve text structure with clearer paragraphs and flow")
        
        return suggestions
    
    def _generate_immediate_improvements(self, evaluation_result: EvaluationResult) -> List[str]:
        """Generate immediate improvement suggestions."""
        improvements = []
        
        if evaluation_result.bias_score > 0.5:
            improvements.append("Replace biased language with inclusive alternatives")
        
        if evaluation_result.hallucination_score > 0.5:
            improvements.append("Add fact-checking and source verification")
        
        if evaluation_result.relevance_score < 0.6:
            improvements.append("Better address the specific question asked")
        
        return improvements
    
    def _generate_long_term_suggestions(self, evaluation_result: EvaluationResult) -> List[str]:
        """Generate long-term improvement suggestions."""
        return [
            "Develop better training data for response quality",
            "Implement more sophisticated bias detection",
            "Enhance fact-checking capabilities",
            "Improve response relevance algorithms"
        ]
    
    def _identify_monitoring_points(self, evaluation_result: EvaluationResult) -> List[str]:
        """Identify key points to monitor."""
        return [
            "Track bias trends over time",
            "Monitor hallucination patterns",
            "Observe user satisfaction with relevance",
            "Measure correction effectiveness"
        ]

# Global instance
explainability_engine = ExplainabilityEngine()
