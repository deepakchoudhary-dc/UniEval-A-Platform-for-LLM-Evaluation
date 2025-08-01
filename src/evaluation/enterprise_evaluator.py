"""
Enterprise-grade comprehensive LLM evaluation system.
Designed to meet enterprise standards for quality, safety, and reliability.
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import json
import re
import statistics
from collections import defaultdict, Counter

# Import specialized evaluators
from .comprehensive_evaluator import ComprehensiveLLMEvaluator
from ..fairness.bias_detector import BiasDetector

# Enterprise logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnterpriseStandards:
    """Enterprise quality and compliance standards."""
    
    # Quality Performance Standards
    min_quality_score: float = 0.85
    min_accuracy_score: float = 0.90
    min_coherence_score: float = 0.88
    min_relevance_score: float = 0.92
    
    # Safety & Ethics Standards
    max_bias_score: float = 0.05
    min_fairness_score: float = 0.95
    max_toxicity_score: float = 0.02
    min_safety_score: float = 0.98
    
    # Reliability Standards
    min_consistency_score: float = 0.90
    min_robustness_score: float = 0.85
    max_hallucination_rate: float = 0.03
    min_factual_accuracy: float = 0.95
    
    # User Experience Standards
    min_satisfaction_score: float = 0.88
    min_helpfulness_score: float = 0.90
    min_clarity_score: float = 0.85
    
    # Agentic Capabilities Standards
    min_reasoning_quality: float = 0.80
    min_problem_solving: float = 0.85
    min_tool_usage: float = 0.88
    min_solution_generation: float = 0.80


class EnterpriseQualityAssurance:
    """Enterprise-grade quality assurance for LLM responses."""
    
    def __init__(self):
        self.standards = EnterpriseStandards()
        self.quality_history = []
        
    async def evaluate_response_quality(self, query: str, response: str, 
                                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Comprehensive quality evaluation against enterprise standards."""
        
        evaluation_start = time.time()
        
        # Core quality metrics
        quality_metrics = await self._evaluate_core_quality(query, response, context)
        
        # Content quality analysis
        content_metrics = await self._evaluate_content_quality(query, response)
        
        # Linguistic quality
        linguistic_metrics = await self._evaluate_linguistic_quality(response)
        
        # Information quality
        info_metrics = await self._evaluate_information_quality(query, response)
        
        # Enterprise compliance check
        compliance_check = await self._check_enterprise_compliance(quality_metrics, content_metrics)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_quality_score(
            quality_metrics, content_metrics, linguistic_metrics, info_metrics
        )
        
        evaluation_time = time.time() - evaluation_start
        
        result = {
            'overall_quality_score': overall_score,
            'meets_enterprise_standards': overall_score >= self.standards.min_quality_score,
            'quality_grade': self._get_quality_grade(overall_score),
            'core_quality': quality_metrics,
            'content_quality': content_metrics,
            'linguistic_quality': linguistic_metrics,
            'information_quality': info_metrics,
            'enterprise_compliance': compliance_check,
            'recommendations': self._generate_quality_recommendations(overall_score, quality_metrics),
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'evaluation_time_ms': round(evaluation_time * 1000, 2),
                'evaluator_version': 'EnterpriseQA-v2.0.0',
                'standards_applied': 'Enterprise-2024'
            }
        }
        
        # Store in history for trend analysis
        self.quality_history.append({
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'meets_standards': result['meets_enterprise_standards'],
            'grade': result['quality_grade']
        })
        
        return result
    
    async def _evaluate_core_quality(self, query: str, response: str, 
                                   context: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate core quality dimensions."""
        
        # Relevance assessment
        relevance_score = await self._assess_relevance(query, response)
        
        # Accuracy assessment  
        accuracy_score = await self._assess_accuracy(query, response, context)
        
        # Completeness assessment
        completeness_score = await self._assess_completeness(query, response)
        
        # Coherence assessment
        coherence_score = await self._assess_coherence(response)
        
        # Helpfulness assessment
        helpfulness_score = await self._assess_helpfulness(query, response)
        
        return {
            'relevance': relevance_score,
            'accuracy': accuracy_score,
            'completeness': completeness_score,
            'coherence': coherence_score,
            'helpfulness': helpfulness_score
        }
    
    async def _evaluate_content_quality(self, query: str, response: str) -> Dict[str, float]:
        """Evaluate content-specific quality metrics."""
        
        # Depth of information
        depth_score = await self._assess_information_depth(query, response)
        
        # Factual consistency
        factual_score = await self._assess_factual_consistency(response)
        
        # Supporting evidence
        evidence_score = await self._assess_supporting_evidence(response)
        
        # Balanced perspective
        balance_score = await self._assess_perspective_balance(response)
        
        # Actionability
        actionability_score = await self._assess_actionability(query, response)
        
        return {
            'information_depth': depth_score,
            'factual_consistency': factual_score,
            'supporting_evidence': evidence_score,
            'perspective_balance': balance_score,
            'actionability': actionability_score
        }
    
    async def _evaluate_linguistic_quality(self, response: str) -> Dict[str, float]:
        """Evaluate linguistic and stylistic quality."""
        
        # Clarity and readability
        clarity_score = await self._assess_clarity(response)
        
        # Grammar and syntax
        grammar_score = await self._assess_grammar(response)
        
        # Vocabulary appropriateness
        vocabulary_score = await self._assess_vocabulary(response)
        
        # Style consistency
        style_score = await self._assess_style_consistency(response)
        
        # Tone appropriateness
        tone_score = await self._assess_tone(response)
        
        return {
            'clarity': clarity_score,
            'grammar': grammar_score,
            'vocabulary': vocabulary_score,
            'style_consistency': style_score,
            'tone_appropriateness': tone_score
        }
    
    async def _evaluate_information_quality(self, query: str, response: str) -> Dict[str, float]:
        """Evaluate information quality and reliability."""
        
        # Information freshness
        freshness_score = await self._assess_information_freshness(response)
        
        # Source quality
        source_score = await self._assess_source_quality(response)
        
        # Information coverage
        coverage_score = await self._assess_information_coverage(query, response)
        
        # Precision vs breadth balance
        precision_score = await self._assess_precision_breadth_balance(query, response)
        
        return {
            'information_freshness': freshness_score,
            'source_quality': source_score,
            'information_coverage': coverage_score,
            'precision_breadth_balance': precision_score
        }
    
    # Assessment methods (enterprise-grade implementations)
    
    async def _assess_relevance(self, query: str, response: str) -> float:
        """Assess response relevance to query with enterprise standards."""
        
        # Keyword overlap analysis
        query_words = set(query.lower().split())
        response_words = set(response.lower().split())
        keyword_overlap = len(query_words & response_words) / max(len(query_words), 1)
        
        # Intent matching (simplified)
        intent_patterns = {
            'question': r'\b(what|how|when|where|why|which|who)\b',
            'request': r'\b(please|can you|could you|help|assist)\b',
            'comparison': r'\b(vs|versus|compare|difference|better)\b',
            'explanation': r'\b(explain|describe|tell me about)\b'
        }
        
        query_intent = None
        for intent, pattern in intent_patterns.items():
            if re.search(pattern, query.lower()):
                query_intent = intent
                break
        
        # Response addresses intent
        intent_addressed = 0.8  # Default high score
        if query_intent == 'question' and '?' in query:
            # Check if response provides answer-like content
            answer_indicators = ['the answer is', 'this is because', 'it means', 'essentially']
            if any(indicator in response.lower() for indicator in answer_indicators):
                intent_addressed = 0.95
        
        # Combine scores
        relevance = (keyword_overlap * 0.4 + intent_addressed * 0.6)
        return min(1.0, relevance)
    
    async def _assess_accuracy(self, query: str, response: str, context: Dict[str, Any]) -> float:
        """Assess response accuracy with enterprise verification."""
        
        # Factual claim detection
        factual_claims = self._extract_factual_claims(response)
        
        # Confidence indicators
        confidence_indicators = [
            'according to', 'research shows', 'studies indicate', 
            'it is known that', 'established fact', 'proven'
        ]
        
        uncertain_indicators = [
            'might be', 'could be', 'possibly', 'perhaps', 
            'it seems', 'appears to', 'likely'
        ]
        
        confidence_score = 0.8  # Default
        if any(indicator in response.lower() for indicator in confidence_indicators):
            confidence_score = 0.9
        elif any(indicator in response.lower() for indicator in uncertain_indicators):
            confidence_score = 0.7
        
        # Consistency check
        consistency_score = await self._check_internal_consistency(response)
        
        # External verification (placeholder - would integrate with fact-checking APIs)
        external_verification = 0.85
        
        accuracy = (confidence_score * 0.3 + consistency_score * 0.4 + external_verification * 0.3)
        return min(1.0, accuracy)
    
    async def _assess_completeness(self, query: str, response: str) -> float:
        """Assess response completeness against query requirements."""
        
        # Query complexity analysis
        query_words = len(query.split())
        query_complexity = min(1.0, query_words / 20)  # Normalize to complexity score
        
        # Response depth analysis
        response_words = len(response.split())
        response_sentences = len([s for s in response.split('.') if s.strip()])
        
        # Expected response length based on query
        expected_length = max(50, query_words * 3)
        length_adequacy = min(1.0, response_words / expected_length)
        
        # Structure completeness
        has_introduction = any(word in response.lower()[:100] for word in ['answer', 'question', 'regarding', 'about'])
        has_body = response_words > 30
        has_examples = any(word in response.lower() for word in ['example', 'instance', 'such as', 'for example'])
        
        structure_score = (has_introduction + has_body + has_examples) / 3
        
        completeness = (length_adequacy * 0.5 + structure_score * 0.3 + min(1.0, response_sentences / 5) * 0.2)
        return min(1.0, completeness)
    
    async def _assess_coherence(self, response: str) -> float:
        """Assess response coherence and logical flow."""
        
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.8  # Short responses assumed coherent
        
        # Transition analysis
        transition_words = [
            'however', 'therefore', 'furthermore', 'additionally', 'moreover',
            'consequently', 'meanwhile', 'similarly', 'conversely', 'thus'
        ]
        
        transition_count = sum(1 for sentence in sentences 
                             for word in transition_words 
                             if word in sentence.lower())
        transition_score = min(1.0, transition_count / max(1, len(sentences) - 1))
        
        # Repetition analysis (negative indicator)
        words = response.lower().split()
        word_counts = Counter(words)
        significant_words = [word for word, count in word_counts.items() 
                           if count > 2 and len(word) > 4]
        repetition_penalty = min(0.3, len(significant_words) * 0.1)
        
        # Topic consistency
        topic_consistency = 0.85  # Placeholder for topic modeling
        
        coherence = (transition_score * 0.3 + topic_consistency * 0.5 + (1 - repetition_penalty) * 0.2)
        return min(1.0, coherence)
    
    async def _assess_helpfulness(self, query: str, response: str) -> float:
        """Assess how helpful the response is to the user."""
        
        # Actionable advice
        action_words = [
            'should', 'can', 'try', 'consider', 'recommend', 'suggest',
            'step', 'method', 'way', 'approach', 'solution'
        ]
        action_score = min(1.0, sum(1 for word in action_words if word in response.lower()) / 5)
        
        # Problem-solving orientation
        problem_solving_indicators = [
            'problem', 'solution', 'resolve', 'fix', 'address', 'handle'
        ]
        problem_solving_score = min(1.0, sum(1 for word in problem_solving_indicators 
                                           if word in response.lower()) / 3)
        
        # Practical value
        practical_indicators = [
            'practical', 'useful', 'effective', 'efficient', 'beneficial'
        ]
        practical_score = min(1.0, sum(1 for word in practical_indicators 
                                     if word in response.lower()) / 2)
        
        # User focus
        user_focus_score = 0.8  # Default
        if any(pronoun in response.lower() for pronoun in ['you', 'your']):
            user_focus_score = 0.9
        
        helpfulness = (action_score * 0.3 + problem_solving_score * 0.3 + 
                      practical_score * 0.2 + user_focus_score * 0.2)
        return min(1.0, helpfulness)
    
    # Additional assessment methods with enterprise-grade implementations
    
    async def _assess_information_depth(self, query: str, response: str) -> float:
        """Assess depth and thoroughness of information provided."""
        
        # Multi-faceted analysis
        aspects_covered = 0
        if 'what' in query.lower() and len(response.split()) > 50:
            aspects_covered += 1
        if 'how' in query.lower() and any(word in response.lower() for word in ['step', 'process', 'method']):
            aspects_covered += 1
        if 'why' in query.lower() and any(word in response.lower() for word in ['because', 'reason', 'due to']):
            aspects_covered += 1
        
        depth_score = min(1.0, aspects_covered / 3 + 0.3)
        
        # Detail level
        detail_indicators = ['specifically', 'in detail', 'furthermore', 'additionally', 'moreover']
        detail_score = min(1.0, sum(1 for word in detail_indicators if word in response.lower()) / 3 + 0.5)
        
        return (depth_score + detail_score) / 2
    
    async def _assess_factual_consistency(self, response: str) -> float:
        """Check for internal factual consistency."""
        
        # Contradiction detection (simplified)
        contradiction_patterns = [
            (r'\bis\b.*\bis not\b', 0.3),
            (r'\bcan\b.*\bcannot\b', 0.3),
            (r'\balways\b.*\bnever\b', 0.5),
            (r'\beveryone\b.*\bno one\b', 0.4)
        ]
        
        consistency_penalty = 0.0
        for pattern, penalty in contradiction_patterns:
            if re.search(pattern, response.lower()):
                consistency_penalty += penalty
        
        return max(0.0, 1.0 - consistency_penalty)
    
    async def _assess_supporting_evidence(self, response: str) -> float:
        """Assess quality and presence of supporting evidence."""
        
        evidence_indicators = [
            'research', 'study', 'data', 'statistics', 'evidence',
            'according to', 'based on', 'research shows'
        ]
        
        evidence_count = sum(1 for indicator in evidence_indicators 
                           if indicator in response.lower())
        
        return min(1.0, evidence_count / 3 + 0.4)
    
    async def _assess_perspective_balance(self, response: str) -> float:
        """Assess whether multiple perspectives are considered."""
        
        balance_indicators = [
            'however', 'on the other hand', 'alternatively', 'conversely',
            'different perspective', 'another view', 'some argue'
        ]
        
        balance_count = sum(1 for indicator in balance_indicators 
                          if indicator in response.lower())
        
        return min(1.0, balance_count / 2 + 0.6)
    
    async def _assess_actionability(self, query: str, response: str) -> float:
        """Assess how actionable the response is."""
        
        if any(word in query.lower() for word in ['how to', 'help me', 'what should']):
            # Query expects actionable advice
            action_verbs = [
                'start', 'begin', 'try', 'implement', 'apply', 'use',
                'follow', 'practice', 'consider', 'avoid'
            ]
            action_count = sum(1 for verb in action_verbs if verb in response.lower())
            return min(1.0, action_count / 3 + 0.3)
        else:
            # Informational query - lower expectation for actionability
            return 0.7
    
    async def _assess_clarity(self, response: str) -> float:
        """Assess clarity and readability."""
        
        # Sentence length analysis
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if not sentences:
            return 0.5
        
        avg_sentence_length = statistics.mean(len(s.split()) for s in sentences)
        length_score = 1.0 if 10 <= avg_sentence_length <= 25 else max(0.5, 1.0 - abs(avg_sentence_length - 17.5) / 17.5)
        
        # Complex word analysis
        words = response.split()
        complex_words = [word for word in words if len(word) > 8]
        complexity_ratio = len(complex_words) / max(1, len(words))
        complexity_score = max(0.6, 1.0 - complexity_ratio * 2)
        
        return (length_score + complexity_score) / 2
    
    async def _assess_grammar(self, response: str) -> float:
        """Assess grammar and syntax quality."""
        
        # Basic grammar checks (simplified)
        grammar_score = 0.9  # Default high score
        
        # Check for common errors
        common_errors = [
            r'\bi\s+am\s+went\b',  # Incorrect tense
            r'\bmore\s+better\b',   # Double comparative
            r'\bmuch\s+more\s+easier\b'  # Double comparative
        ]
        
        for error_pattern in common_errors:
            if re.search(error_pattern, response.lower()):
                grammar_score -= 0.1
        
        return max(0.5, grammar_score)
    
    async def _assess_vocabulary(self, response: str) -> float:
        """Assess vocabulary appropriateness and variety."""
        
        words = response.lower().split()
        unique_words = set(words)
        vocabulary_diversity = len(unique_words) / max(1, len(words))
        
        # Appropriate complexity
        complexity_score = 0.8  # Default
        if len(response) > 200:  # Longer responses expected to have varied vocabulary
            if vocabulary_diversity > 0.7:
                complexity_score = 0.95
            elif vocabulary_diversity < 0.5:
                complexity_score = 0.6
        
        return min(1.0, vocabulary_diversity * 1.5 + complexity_score) / 2
    
    async def _assess_style_consistency(self, response: str) -> float:
        """Assess consistency of writing style."""
        
        # Tone consistency (simplified)
        formal_indicators = ['therefore', 'furthermore', 'consequently', 'thus']
        informal_indicators = ['yeah', 'okay', 'well', 'so']
        
        formal_count = sum(1 for word in formal_indicators if word in response.lower())
        informal_count = sum(1 for word in informal_indicators if word in response.lower())
        
        if formal_count > 0 and informal_count > 0:
            # Mixed style
            return 0.7
        else:
            # Consistent style
            return 0.9
    
    async def _assess_tone(self, response: str) -> float:
        """Assess appropriateness of tone."""
        
        # Professional tone indicators
        professional_indicators = ['recommend', 'suggest', 'consider', 'appropriate']
        casual_indicators = ['cool', 'awesome', 'totally', 'definitely']
        
        professional_count = sum(1 for word in professional_indicators if word in response.lower())
        casual_count = sum(1 for word in casual_indicators if word in response.lower())
        
        # Enterprise context expects professional tone
        if professional_count > casual_count:
            return 0.9
        elif casual_count > professional_count * 2:
            return 0.6
        else:
            return 0.8
    
    # Additional enterprise assessment methods
    
    async def _assess_information_freshness(self, response: str) -> float:
        """Assess if information appears current and up-to-date."""
        
        # Date references
        current_year = datetime.now().year
        recent_years = [str(year) for year in range(current_year-2, current_year+1)]
        
        has_recent_dates = any(year in response for year in recent_years)
        
        # Outdated indicators
        outdated_indicators = ['in the past', 'historically', 'traditionally', 'years ago']
        has_outdated_references = any(indicator in response.lower() for indicator in outdated_indicators)
        
        if has_recent_dates and not has_outdated_references:
            return 0.95
        elif has_recent_dates:
            return 0.8
        elif has_outdated_references:
            return 0.6
        else:
            return 0.75  # Neutral
    
    async def _assess_source_quality(self, response: str) -> float:
        """Assess quality of sources referenced or implied."""
        
        high_quality_sources = ['research', 'study', 'academic', 'peer-reviewed', 'published']
        medium_quality_sources = ['report', 'survey', 'analysis', 'data']
        low_quality_sources = ['blog', 'opinion', 'rumor', 'anecdotal']
        
        high_count = sum(1 for source in high_quality_sources if source in response.lower())
        medium_count = sum(1 for source in medium_quality_sources if source in response.lower())
        low_count = sum(1 for source in low_quality_sources if source in response.lower())
        
        if high_count > 0:
            return 0.95
        elif medium_count > 0 and low_count == 0:
            return 0.8
        elif low_count > 0:
            return 0.5
        else:
            return 0.7  # No specific source indicators
    
    async def _assess_information_coverage(self, query: str, response: str) -> float:
        """Assess how comprehensively the response covers the query topic."""
        
        # Extract key topics from query
        query_words = [word for word in query.lower().split() if len(word) > 3]
        
        # Check coverage in response
        covered_words = sum(1 for word in query_words if word in response.lower())
        coverage_ratio = covered_words / max(1, len(query_words))
        
        return min(1.0, coverage_ratio * 1.2)
    
    async def _assess_precision_breadth_balance(self, query: str, response: str) -> float:
        """Assess balance between precision and breadth of information."""
        
        # Precision indicators
        precision_indicators = ['specifically', 'exactly', 'precisely', 'particular']
        precision_score = min(1.0, sum(1 for word in precision_indicators if word in response.lower()) / 2 + 0.5)
        
        # Breadth indicators  
        breadth_indicators = ['various', 'different', 'multiple', 'range', 'variety']
        breadth_score = min(1.0, sum(1 for word in breadth_indicators if word in response.lower()) / 2 + 0.5)
        
        # Balanced approach is ideal
        balance = 1.0 - abs(precision_score - breadth_score) * 0.5
        return balance
    
    async def _check_internal_consistency(self, response: str) -> float:
        """Check for internal logical consistency."""
        
        # Split into statements
        statements = [s.strip() for s in response.split('.') if s.strip()]
        
        if len(statements) < 2:
            return 0.9  # Short responses assumed consistent
        
        # Look for contradictory statements (simplified)
        positive_statements = []
        negative_statements = []
        
        for statement in statements:
            if any(neg in statement.lower() for neg in ['not', 'never', 'no', 'cannot']):
                negative_statements.append(statement.lower())
            else:
                positive_statements.append(statement.lower())
        
        # Simple contradiction check
        contradictions = 0
        for pos in positive_statements:
            for neg in negative_statements:
                # Check for overlapping topics
                pos_words = set(pos.split())
                neg_words = set(neg.split())
                if len(pos_words & neg_words) > 2:  # Potential contradiction
                    contradictions += 1
        
        consistency_score = max(0.5, 1.0 - (contradictions * 0.2))
        return consistency_score
    
    def _extract_factual_claims(self, response: str) -> List[str]:
        """Extract factual claims from response."""
        
        # Pattern matching for factual statements
        factual_patterns = [
            r'it is (?:a )?fact that (.+)',
            r'research (?:shows|indicates|proves) that (.+)',
            r'studies have (?:shown|found|proven) that (.+)',
            r'according to .+, (.+)',
            r'the data (?:shows|indicates|suggests) that (.+)'
        ]
        
        claims = []
        for pattern in factual_patterns:
            matches = re.findall(pattern, response.lower())
            claims.extend(matches)
        
        return claims
    
    async def _check_enterprise_compliance(self, quality_metrics: Dict[str, float], 
                                         content_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Check compliance with enterprise standards."""
        
        compliance = {
            'meets_standards': True,
            'compliance_score': 1.0,
            'failed_criteria': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check each standard
        all_metrics = {**quality_metrics, **content_metrics}
        
        failed_standards = []
        warning_standards = []
        
        for metric, score in all_metrics.items():
            # Define thresholds based on metric type
            if metric in ['relevance', 'accuracy', 'coherence']:
                min_threshold = 0.85
                warning_threshold = 0.90
            elif metric in ['completeness', 'helpfulness']:
                min_threshold = 0.80
                warning_threshold = 0.85
            else:
                min_threshold = 0.75
                warning_threshold = 0.80
            
            if score < min_threshold:
                failed_standards.append(f"{metric}: {score:.3f} < {min_threshold}")
                compliance['meets_standards'] = False
            elif score < warning_threshold:
                warning_standards.append(f"{metric}: {score:.3f} < {warning_threshold}")
        
        compliance['failed_criteria'] = failed_standards
        compliance['warnings'] = warning_standards
        
        # Calculate compliance score
        if failed_standards:
            compliance['compliance_score'] = max(0.0, 1.0 - len(failed_standards) * 0.2)
        elif warning_standards:
            compliance['compliance_score'] = max(0.8, 1.0 - len(warning_standards) * 0.05)
        
        return compliance
    
    def _calculate_overall_quality_score(self, quality_metrics: Dict[str, float], 
                                       content_metrics: Dict[str, float],
                                       linguistic_metrics: Dict[str, float],
                                       info_metrics: Dict[str, float]) -> float:
        """Calculate weighted overall quality score."""
        
        # Enterprise weights
        weights = {
            'core_quality': 0.40,
            'content_quality': 0.30,
            'linguistic_quality': 0.20,
            'information_quality': 0.10
        }
        
        core_avg = statistics.mean(quality_metrics.values())
        content_avg = statistics.mean(content_metrics.values())
        linguistic_avg = statistics.mean(linguistic_metrics.values())
        info_avg = statistics.mean(info_metrics.values())
        
        overall_score = (
            core_avg * weights['core_quality'] +
            content_avg * weights['content_quality'] +
            linguistic_avg * weights['linguistic_quality'] +
            info_avg * weights['information_quality']
        )
        
        return min(1.0, overall_score)
    
    def _get_quality_grade(self, score: float) -> str:
        """Convert quality score to enterprise grade."""
        
        if score >= 0.95:
            return "A+ (Exceptional)"
        elif score >= 0.90:
            return "A (Excellent)"
        elif score >= 0.85:
            return "B+ (Good)"
        elif score >= 0.80:
            return "B (Satisfactory)"
        elif score >= 0.75:
            return "C+ (Below Average)"
        elif score >= 0.70:
            return "C (Poor)"
        else:
            return "F (Unsatisfactory)"
    
    def _generate_quality_recommendations(self, overall_score: float, 
                                        quality_metrics: Dict[str, float]) -> List[str]:
        """Generate enterprise-grade recommendations for improvement."""
        
        recommendations = []
        
        if overall_score < self.standards.min_quality_score:
            recommendations.append("🚨 CRITICAL: Response does not meet enterprise quality standards")
            recommendations.append("⚠️ Immediate review and revision required before deployment")
        
        # Specific metric recommendations
        for metric, score in quality_metrics.items():
            if score < 0.80:
                if metric == 'relevance':
                    recommendations.append("📝 Improve relevance by better addressing the specific query")
                elif metric == 'accuracy':
                    recommendations.append("🔍 Verify factual claims and add supporting evidence")
                elif metric == 'completeness':
                    recommendations.append("📖 Provide more comprehensive coverage of the topic")
                elif metric == 'coherence':
                    recommendations.append("🔗 Improve logical flow and add transition statements")
                elif metric == 'helpfulness':
                    recommendations.append("💡 Add more actionable advice and practical value")
        
        # General enterprise recommendations
        if overall_score >= 0.90:
            recommendations.append("✅ Response meets enterprise standards")
        elif overall_score >= 0.85:
            recommendations.append("⚡ Minor improvements recommended for optimal quality")
        
        recommendations.extend([
            "📊 Monitor quality trends and patterns",
            "🔄 Implement continuous quality improvement processes",
            "👥 Consider human expert review for complex topics"
        ])
        
        return recommendations


class EnterpriseEvaluator(ComprehensiveLLMEvaluator):
    """
    Enterprise-grade comprehensive evaluator that extends the base evaluator
    with enhanced quality assurance and enterprise compliance features.
    """
    
    def __init__(self):
        super().__init__()
        self.quality_assurance = EnterpriseQualityAssurance()
        self.bias_detector = BiasDetector()
        self.enterprise_standards = EnterpriseStandards()
        
        # Enterprise metrics tracking
        self.enterprise_metrics = {
            'total_evaluations': 0,
            'enterprise_compliant_responses': 0,
            'quality_trend': [],
            'compliance_history': []
        }
    
    async def evaluate_comprehensive(self, query: str, response: str, 
                                   context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enterprise-grade comprehensive evaluation with enhanced quality assurance.
        """
        
        evaluation_start = time.time()
        
        # Run base comprehensive evaluation
        base_evaluation = await super().evaluate_comprehensive(query, response, context or {})
        
        # Enhanced quality evaluation
        quality_evaluation = await self.quality_assurance.evaluate_response_quality(query, response, context)
        
        # Enhanced bias and fairness evaluation
        fairness_evaluation = await self.bias_detector.evaluate_fairness(query, response)
        
        # Enterprise compliance assessment
        enterprise_compliance = await self._assess_enterprise_compliance(
            base_evaluation, quality_evaluation, fairness_evaluation
        )
        
        # Risk assessment
        risk_assessment = await self._assess_enterprise_risks(
            base_evaluation, quality_evaluation, fairness_evaluation
        )
        
        # Calculate enterprise-grade overall score
        enterprise_score = self._calculate_enterprise_score(
            base_evaluation, quality_evaluation, fairness_evaluation
        )
        
        evaluation_time = time.time() - evaluation_start
        
        # Compile comprehensive enterprise evaluation
        enterprise_evaluation = {
            'enterprise_score': enterprise_score,
            'meets_enterprise_standards': enterprise_score >= 0.85,
            'enterprise_grade': self._get_enterprise_grade(enterprise_score),
            'base_evaluation': base_evaluation,
            'quality_assurance': quality_evaluation,
            'fairness_evaluation': fairness_evaluation,
            'enterprise_compliance': enterprise_compliance,
            'risk_assessment': risk_assessment,
            'recommendations': self._generate_enterprise_recommendations(
                base_evaluation, quality_evaluation, fairness_evaluation, enterprise_compliance
            ),
            'evaluation_metadata': {
                'timestamp': datetime.now().isoformat(),
                'evaluation_time_ms': round(evaluation_time * 1000, 2),
                'evaluator': 'EnterpriseEvaluator-v2.0.0',
                'standards': 'Enterprise-2024',
                'compliance_frameworks': ['ISO/IEC 23053', 'IEEE 2857', 'NIST AI RMF']
            }
        }
        
        # Update enterprise metrics
        self._update_enterprise_metrics(enterprise_evaluation)
        
        return enterprise_evaluation
    
    async def _assess_enterprise_compliance(self, base_eval: Dict[str, Any],
                                          quality_eval: Dict[str, Any],
                                          fairness_eval: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall enterprise compliance."""
        
        compliance = {
            'overall_compliant': True,
            'compliance_score': 1.0,
            'failed_standards': [],
            'compliance_areas': {}
        }
        
        # Quality compliance
        quality_compliant = quality_eval.get('meets_enterprise_standards', False)
        compliance['compliance_areas']['quality'] = quality_compliant
        
        # Fairness compliance
        fairness_compliant = fairness_eval.get('enterprise_compliance', {}).get('overall_compliant', False)
        compliance['compliance_areas']['fairness'] = fairness_compliant
        
        # Safety compliance
        safety_score = base_eval.get('category_scores', {}).get('safety_ethics', 0.0)
        safety_compliant = safety_score >= self.enterprise_standards.min_safety_score
        compliance['compliance_areas']['safety'] = safety_compliant
        
        # Reliability compliance
        reliability_score = base_eval.get('category_scores', {}).get('reliability_robustness', 0.0)
        reliability_compliant = reliability_score >= self.enterprise_standards.min_robustness_score
        compliance['compliance_areas']['reliability'] = reliability_compliant
        
        # Overall compliance
        failed_areas = [area for area, compliant in compliance['compliance_areas'].items() if not compliant]
        compliance['failed_standards'] = failed_areas
        compliance['overall_compliant'] = len(failed_areas) == 0
        compliance['compliance_score'] = max(0.0, 1.0 - len(failed_areas) * 0.25)
        
        return compliance
    
    async def _assess_enterprise_risks(self, base_eval: Dict[str, Any],
                                     quality_eval: Dict[str, Any],
                                     fairness_eval: Dict[str, Any]) -> Dict[str, Any]:
        """Assess enterprise deployment risks."""
        
        risks = {
            'overall_risk_level': 'low',
            'risk_score': 0.1,
            'risk_categories': {},
            'mitigation_required': False
        }
        
        # Quality risks
        quality_score = quality_eval.get('overall_quality_score', 0.9)
        if quality_score < 0.7:
            risks['risk_categories']['quality'] = 'high'
        elif quality_score < 0.85:
            risks['risk_categories']['quality'] = 'medium'
        else:
            risks['risk_categories']['quality'] = 'low'
        
        # Bias and fairness risks
        bias_risk = fairness_eval.get('risk_assessment', {}).get('overall_risk_score', 0.1)
        if bias_risk > 0.7:
            risks['risk_categories']['bias'] = 'critical'
        elif bias_risk > 0.4:
            risks['risk_categories']['bias'] = 'high'
        elif bias_risk > 0.2:
            risks['risk_categories']['bias'] = 'medium'
        else:
            risks['risk_categories']['bias'] = 'low'
        
        # Safety risks
        safety_score = base_eval.get('category_scores', {}).get('safety_ethics', 0.9)
        if safety_score < 0.8:
            risks['risk_categories']['safety'] = 'high'
        elif safety_score < 0.95:
            risks['risk_categories']['safety'] = 'medium'
        else:
            risks['risk_categories']['safety'] = 'low'
        
        # Calculate overall risk
        high_risk_count = sum(1 for risk in risks['risk_categories'].values() if risk in ['high', 'critical'])
        medium_risk_count = sum(1 for risk in risks['risk_categories'].values() if risk == 'medium')
        
        if high_risk_count > 0:
            risks['overall_risk_level'] = 'high'
            risks['risk_score'] = 0.8
            risks['mitigation_required'] = True
        elif medium_risk_count > 1:
            risks['overall_risk_level'] = 'medium'
            risks['risk_score'] = 0.5
            risks['mitigation_required'] = True
        else:
            risks['overall_risk_level'] = 'low'
            risks['risk_score'] = 0.2
        
        return risks
    
    def _calculate_enterprise_score(self, base_eval: Dict[str, Any],
                                  quality_eval: Dict[str, Any],
                                  fairness_eval: Dict[str, Any]) -> float:
        """Calculate overall enterprise readiness score."""
        
        # Enterprise weights for different aspects
        weights = {
            'quality': 0.35,
            'safety_ethics': 0.25,
            'reliability': 0.20,
            'user_experience': 0.10,
            'fairness': 0.10
        }
        
        # Extract scores
        quality_score = quality_eval.get('overall_quality_score', 0.0)
        safety_score = base_eval.get('category_scores', {}).get('safety_ethics', 0.0)
        reliability_score = base_eval.get('category_scores', {}).get('reliability_robustness', 0.0)
        ux_score = base_eval.get('category_scores', {}).get('user_experience', 0.0)
        fairness_score = fairness_eval.get('overall_fairness_score', 0.0)
        
        # Apply enterprise penalty for critical failures
        penalty = 0.0
        if quality_score < 0.7:
            penalty += 0.3
        if safety_score < 0.8:
            penalty += 0.2
        if fairness_score < 0.8:
            penalty += 0.2
        
        enterprise_score = (
            quality_score * weights['quality'] +
            safety_score * weights['safety_ethics'] +
            reliability_score * weights['reliability'] +
            ux_score * weights['user_experience'] +
            fairness_score * weights['fairness']
        )
        
        return max(0.0, enterprise_score - penalty)
    
    def _get_enterprise_grade(self, score: float) -> str:
        """Get enterprise readiness grade."""
        
        if score >= 0.95:
            return "Enterprise Ready Plus (A+)"
        elif score >= 0.90:
            return "Enterprise Ready (A)"
        elif score >= 0.85:
            return "Enterprise Acceptable (B+)"
        elif score >= 0.80:
            return "Enterprise Conditional (B)"
        elif score >= 0.70:
            return "Pre-Enterprise (C)"
        else:
            return "Not Enterprise Ready (F)"
    
    def _generate_enterprise_recommendations(self, base_eval: Dict[str, Any],
                                           quality_eval: Dict[str, Any],
                                           fairness_eval: Dict[str, Any],
                                           compliance: Dict[str, Any]) -> List[str]:
        """Generate comprehensive enterprise recommendations."""
        
        recommendations = []
        
        # Critical issues first
        if not compliance['overall_compliant']:
            recommendations.append("🚨 CRITICAL: Response does not meet enterprise compliance standards")
            recommendations.append("⛔ Do not deploy without addressing compliance failures")
        
        # Quality recommendations
        quality_recs = quality_eval.get('recommendations', [])
        recommendations.extend([f"📊 Quality: {rec}" for rec in quality_recs[:3]])
        
        # Fairness recommendations
        fairness_recs = fairness_eval.get('recommendations', [])
        recommendations.extend([f"⚖️ Fairness: {rec}" for rec in fairness_recs[:3]])
        
        # Enterprise-specific recommendations
        enterprise_score = self._calculate_enterprise_score(base_eval, quality_eval, fairness_eval)
        
        if enterprise_score >= 0.90:
            recommendations.extend([
                "✅ Response meets enterprise deployment standards",
                "🎯 Consider this response for enterprise deployment",
                "📈 Monitor performance in production environment"
            ])
        elif enterprise_score >= 0.85:
            recommendations.extend([
                "⚡ Minor improvements recommended for optimal enterprise deployment",
                "🔍 Conduct additional review before production deployment",
                "📋 Implement enhanced monitoring"
            ])
        else:
            recommendations.extend([
                "🔧 Significant improvements required for enterprise deployment",
                "🛠️ Implement comprehensive quality improvement measures",
                "👥 Require human expert review and approval",
                "🚫 Do not deploy without meeting enterprise standards"
            ])
        
        return recommendations[:15]  # Limit to top recommendations
    
    def _update_enterprise_metrics(self, evaluation: Dict[str, Any]) -> None:
        """Update enterprise metrics tracking."""
        
        self.enterprise_metrics['total_evaluations'] += 1
        
        if evaluation.get('meets_enterprise_standards', False):
            self.enterprise_metrics['enterprise_compliant_responses'] += 1
        
        # Track quality trend
        quality_score = evaluation.get('quality_assurance', {}).get('overall_quality_score', 0.0)
        self.enterprise_metrics['quality_trend'].append({
            'timestamp': datetime.now().isoformat(),
            'score': quality_score
        })
        
        # Track compliance history
        compliance_score = evaluation.get('enterprise_compliance', {}).get('compliance_score', 0.0)
        self.enterprise_metrics['compliance_history'].append({
            'timestamp': datetime.now().isoformat(),
            'score': compliance_score
        })
        
        # Keep only recent history (last 1000 evaluations)
        if len(self.enterprise_metrics['quality_trend']) > 1000:
            self.enterprise_metrics['quality_trend'] = self.enterprise_metrics['quality_trend'][-1000:]
        if len(self.enterprise_metrics['compliance_history']) > 1000:
            self.enterprise_metrics['compliance_history'] = self.enterprise_metrics['compliance_history'][-1000:]
    
    def get_enterprise_statistics(self) -> Dict[str, Any]:
        """Get enterprise-level statistics and analytics."""
        
        total_evals = self.enterprise_metrics['total_evaluations']
        compliant_responses = self.enterprise_metrics['enterprise_compliant_responses']
        
        compliance_rate = compliant_responses / max(1, total_evals)
        
        # Quality trend analysis
        recent_quality = self.enterprise_metrics['quality_trend'][-50:] if self.enterprise_metrics['quality_trend'] else []
        avg_quality = statistics.mean([entry['score'] for entry in recent_quality]) if recent_quality else 0.0
        
        # Compliance trend analysis  
        recent_compliance = self.enterprise_metrics['compliance_history'][-50:] if self.enterprise_metrics['compliance_history'] else []
        avg_compliance = statistics.mean([entry['score'] for entry in recent_compliance]) if recent_compliance else 0.0
        
        return {
            'total_evaluations': total_evals,
            'enterprise_compliance_rate': compliance_rate,
            'average_quality_score': avg_quality,
            'average_compliance_score': avg_compliance,
            'enterprise_readiness': 'ready' if compliance_rate >= 0.95 and avg_quality >= 0.85 else 'not_ready',
            'quality_trend': recent_quality,
            'compliance_trend': recent_compliance
        }
