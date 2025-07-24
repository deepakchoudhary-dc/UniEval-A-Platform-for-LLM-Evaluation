"""
Self-Correction Module with Opik Integration

This module provides intelligent self-correction capabilities based on Opik evaluation results.
It can automatically detect when responses need improvement and generate corrected versions.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re

from .enhanced_opik_evaluator import enhanced_opik_evaluator, EvaluationResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SelfCorrectionEngine:
    """
    Intelligent self-correction engine that uses Opik evaluation data
    to automatically improve response quality.
    """
    
    def __init__(self):
        """Initialize the self-correction engine."""
        self.evaluator = enhanced_opik_evaluator
        
        # Correction thresholds
        self.bias_threshold = 0.6
        self.hallucination_threshold = 0.6
        self.accuracy_threshold = 0.4
        self.relevance_threshold = 0.4
        
        # Correction templates
        self.correction_templates = self._load_correction_templates()
        
        logger.info("Self-correction engine initialized")
    
    def _load_correction_templates(self) -> Dict[str, str]:
        """Load correction templates for different types of issues."""
        return {
            'bias_reduction': """
                Original response contained biased language. Please revise to:
                - Use inclusive, neutral terminology
                - Avoid stereotypes or assumptions
                - Present multiple perspectives when appropriate
                - Ensure fair representation of all groups
                
                Original: {original_response}
                
                Improved response:
            """,
            
            'hallucination_fix': """
                The response may contain inaccurate information. Please revise to:
                - Only include verified, factual information
                - Clearly distinguish between facts and opinions
                - Acknowledge uncertainty when appropriate
                - Provide reliable sources when possible
                
                Original: {original_response}
                
                Factually accurate response:
            """,
            
            'relevance_improvement': """
                The response didn't fully address the question. Please revise to:
                - Directly answer the specific question asked
                - Stay focused on the main topic
                - Address all parts of the question
                - Provide relevant examples or details
                
                Question: {original_question}
                Original response: {original_response}
                
                More relevant response:
            """,
            
            'usefulness_enhancement': """
                The response could be more helpful. Please revise to:
                - Provide practical, actionable information
                - Include specific steps or recommendations
                - Add relevant examples or use cases
                - Make the information easier to understand and apply
                
                Original: {original_response}
                
                More useful response:
            """,
            
            'general_improvement': """
                Please improve this response by making it more accurate, helpful, and clear:
                
                Original: {original_response}
                
                Improved response:
            """
        }
    
    async def evaluate_and_correct(
        self,
        input_text: str,
        output_text: str,
        conversation_id: str,
        max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Evaluate response and automatically correct if needed.
        
        Returns:
            dict: Correction results with original evaluation, corrections made, and final response
        """
        corrections_made = []
        current_response = output_text
        iteration = 0
        
        while iteration < max_iterations:
            # Evaluate current response
            evaluation = await self.evaluator.evaluate_response_realtime(
                input_text, current_response, conversation_id
            )
            
            # Check if correction is needed
            correction_needed = await self.evaluator.assess_correction_need(evaluation)
            
            if not correction_needed['needs_correction']:
                # Response is good enough
                break
            
            # Generate corrected response
            corrected_response = await self._generate_correction(
                input_text, current_response, correction_needed
            )
            
            if corrected_response and corrected_response != current_response:
                corrections_made.append({
                    'iteration': iteration + 1,
                    'issues_found': correction_needed['suggested_improvements'],
                    'original_scores': {
                        'accuracy': evaluation.accuracy_score,
                        'bias': evaluation.bias_score,
                        'hallucination': evaluation.hallucination_score,
                        'relevance': evaluation.relevance_score,
                        'usefulness': evaluation.usefulness_score,
                        'overall': evaluation.overall_score
                    },
                    'corrected_response': corrected_response
                })
                current_response = corrected_response
            else:
                # Unable to generate better response
                break
            
            iteration += 1
        
        # Final evaluation
        final_evaluation = await self.evaluator.evaluate_response_realtime(
            input_text, current_response, conversation_id
        )
        
        return {
            'original_response': output_text,
            'final_response': current_response,
            'corrections_made': corrections_made,
            'iterations': iteration,
            'improvement_achieved': len(corrections_made) > 0,
            'original_evaluation': {
                'accuracy': corrections_made[0]['original_scores']['accuracy'] if corrections_made else final_evaluation.accuracy_score,
                'bias': corrections_made[0]['original_scores']['bias'] if corrections_made else final_evaluation.bias_score,
                'hallucination': corrections_made[0]['original_scores']['hallucination'] if corrections_made else final_evaluation.hallucination_score,
                'relevance': corrections_made[0]['original_scores']['relevance'] if corrections_made else final_evaluation.relevance_score,
                'usefulness': corrections_made[0]['original_scores']['usefulness'] if corrections_made else final_evaluation.usefulness_score,
                'overall': corrections_made[0]['original_scores']['overall'] if corrections_made else final_evaluation.overall_score
            },
            'final_evaluation': {
                'accuracy': final_evaluation.accuracy_score,
                'bias': final_evaluation.bias_score,
                'hallucination': final_evaluation.hallucination_score,
                'relevance': final_evaluation.relevance_score,
                'usefulness': final_evaluation.usefulness_score,
                'overall': final_evaluation.overall_score,
                'confidence': final_evaluation.confidence_level,
                'requires_correction': final_evaluation.requires_correction
            },
            'correction_summary': self._generate_correction_summary(corrections_made, final_evaluation)
        }
    
    async def _generate_correction(
        self,
        input_text: str,
        output_text: str,
        correction_data: Dict[str, Any]
    ) -> Optional[str]:
        """Generate corrected response based on identified issues."""
        try:
            # Determine primary correction type
            primary_correction = self._identify_primary_correction_type(correction_data)
            
            # Get appropriate template
            template = self.correction_templates.get(primary_correction, self.correction_templates['general_improvement'])
            
            # Format correction prompt
            if primary_correction == 'relevance_improvement':
                correction_prompt = template.format(
                    original_question=input_text,
                    original_response=output_text
                )
            else:
                correction_prompt = template.format(
                    original_response=output_text
                )
            
            # Apply rule-based corrections
            corrected_response = await self._apply_rule_based_corrections(
                output_text, correction_data
            )
            
            # If we have access to an LLM, we could generate a more sophisticated correction here
            # For now, return the rule-based correction
            return corrected_response
            
        except Exception as e:
            logger.error(f"Failed to generate correction: {e}")
            return None
    
    def _identify_primary_correction_type(self, correction_data: Dict[str, Any]) -> str:
        """Identify the most important correction type needed."""
        improvements = correction_data.get('suggested_improvements', [])
        
        # Priority order: bias > hallucination > relevance > usefulness
        for improvement in improvements:
            if improvement['type'] == 'bias_reduction':
                return 'bias_reduction'
        
        for improvement in improvements:
            if improvement['type'] == 'factual_accuracy':
                return 'hallucination_fix'
        
        for improvement in improvements:
            if improvement['type'] == 'relevance':
                return 'relevance_improvement'
        
        for improvement in improvements:
            if improvement['type'] == 'usefulness':
                return 'usefulness_enhancement'
        
        return 'general_improvement'
    
    async def _apply_rule_based_corrections(
        self,
        text: str,
        correction_data: Dict[str, Any]
    ) -> str:
        """Apply rule-based corrections to text."""
        corrected_text = text
        
        # Apply bias reduction rules
        if any(imp['type'] == 'bias_reduction' for imp in correction_data.get('suggested_improvements', [])):
            corrected_text = self._reduce_bias_in_text(corrected_text)
        
        # Apply hallucination reduction rules
        if any(imp['type'] == 'factual_accuracy' for imp in correction_data.get('suggested_improvements', [])):
            corrected_text = self._reduce_hallucination_in_text(corrected_text)
        
        # Apply relevance improvement rules
        if any(imp['type'] == 'relevance' for imp in correction_data.get('suggested_improvements', [])):
            corrected_text = self._improve_relevance_in_text(corrected_text)
        
        # Apply usefulness enhancement rules
        if any(imp['type'] == 'usefulness' for imp in correction_data.get('suggested_improvements', [])):
            corrected_text = self._enhance_usefulness_in_text(corrected_text)
        
        return corrected_text
    
    def _reduce_bias_in_text(self, text: str) -> str:
        """Apply bias reduction rules to text."""
        # Gender-neutral language replacements
        bias_replacements = {
            r'\bhe/she\b': 'they',
            r'\bhis/her\b': 'their',
            r'\bhim/her\b': 'them',
            r'\bguys\b': 'everyone',
            r'\bmankind\b': 'humanity',
            r'\bmanpower\b': 'workforce',
            r'\bchairman\b': 'chairperson',
            r'\bfireman\b': 'firefighter',
            r'\bpoliceman\b': 'police officer',
            r'\bmailman\b': 'mail carrier',
            r'\bstewardess\b': 'flight attendant'
        }
        
        corrected = text
        for pattern, replacement in bias_replacements.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        # Remove potentially biased adjectives
        biased_patterns = [
            r'\b(obviously|clearly|naturally)\s+',  # Remove certainty markers that may hide bias
            r'\b(all|every|no)\s+(men|women|people)\s+(are|do|have)\b',  # Remove generalizations
        ]
        
        for pattern in biased_patterns:
            corrected = re.sub(pattern, '', corrected, flags=re.IGNORECASE)
        
        return corrected.strip()
    
    def _reduce_hallucination_in_text(self, text: str) -> str:
        """Apply hallucination reduction rules to text."""
        # Add uncertainty markers to confident statements
        uncertainty_patterns = [
            (r'\bwill definitely\b', 'will likely'),
            (r'\bwill certainly\b', 'will probably'),
            (r'\bis definitely\b', 'is likely'),
            (r'\bis certainly\b', 'is probably'),
            (r'\bproves that\b', 'suggests that'),
            (r'\bshows that\b', 'indicates that'),
            (r'\balways\b', 'typically'),
            (r'\bnever\b', 'rarely'),
            (r'\beveryone knows\b', 'it is commonly believed'),
            (r'\bobviously\b', 'it appears that'),
        ]
        
        corrected = text
        for pattern, replacement in uncertainty_patterns:
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
        
        # Add disclaimers for factual claims
        if re.search(r'\b(according to|studies show|research indicates)\b', corrected, re.IGNORECASE):
            if not re.search(r'\b(may|might|could|appears|suggests)\b', corrected, re.IGNORECASE):
                corrected += " (Please verify this information with current sources.)"
        
        return corrected.strip()
    
    def _improve_relevance_in_text(self, text: str) -> str:
        """Apply relevance improvement rules to text."""
        # Remove off-topic sentences (simple heuristic)
        sentences = text.split('.')
        if len(sentences) > 3:
            # Keep first 2 and last sentence, remove middle if too long
            corrected = '. '.join(sentences[:2] + sentences[-1:])
            if corrected.endswith('.'):
                return corrected
            else:
                return corrected + '.'
        
        return text
    
    def _enhance_usefulness_in_text(self, text: str) -> str:
        """Apply usefulness enhancement rules to text."""
        # Add action words if missing
        action_indicators = ['should', 'can', 'try', 'consider', 'recommend', 'suggest']
        
        if not any(word in text.lower() for word in action_indicators):
            # Add a suggestion if the text is informational but not actionable
            if len(text) > 50 and not text.endswith('?'):
                corrected = text.rstrip('.')
                corrected += ". Consider applying this information to your specific situation."
                return corrected
        
        return text
    
    def _generate_correction_summary(
        self,
        corrections_made: List[Dict[str, Any]],
        final_evaluation: EvaluationResult
    ) -> Dict[str, Any]:
        """Generate a summary of corrections made."""
        if not corrections_made:
            return {
                'corrections_applied': 0,
                'issues_resolved': [],
                'improvement_percentage': 0,
                'final_quality': final_evaluation.confidence_level
            }
        
        # Calculate improvement
        original_score = corrections_made[0]['original_scores']['overall']
        final_score = final_evaluation.overall_score
        improvement_percentage = ((final_score - original_score) / original_score * 100) if original_score > 0 else 0
        
        # Collect all issues that were addressed
        all_issues = []
        for correction in corrections_made:
            for issue in correction['issues_found']:
                if issue['type'] not in [i['type'] for i in all_issues]:
                    all_issues.append(issue)
        
        return {
            'corrections_applied': len(corrections_made),
            'issues_resolved': [issue['type'] for issue in all_issues],
            'improvement_percentage': round(improvement_percentage, 2),
            'final_quality': final_evaluation.confidence_level,
            'score_improvements': {
                'accuracy': final_evaluation.accuracy_score - corrections_made[0]['original_scores']['accuracy'],
                'bias': corrections_made[0]['original_scores']['bias'] - final_evaluation.bias_score,  # Lower is better
                'hallucination': corrections_made[0]['original_scores']['hallucination'] - final_evaluation.hallucination_score,  # Lower is better
                'relevance': final_evaluation.relevance_score - corrections_made[0]['original_scores']['relevance'],
                'usefulness': final_evaluation.usefulness_score - corrections_made[0]['original_scores']['usefulness'],
                'overall': final_evaluation.overall_score - corrections_made[0]['original_scores']['overall']
            }
        }
    
    async def suggest_proactive_improvements(
        self,
        input_text: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Suggest improvements before generating a response.
        Useful for preventing issues rather than correcting them.
        """
        suggestions = {
            'content_guidelines': [],
            'style_recommendations': [],
            'bias_prevention': [],
            'accuracy_tips': []
        }
        
        # Analyze input for potential bias triggers
        bias_keywords = [
            'gender', 'race', 'religion', 'age', 'disability', 'nationality',
            'sexual orientation', 'ethnicity', 'culture', 'politics'
        ]
        
        if any(keyword in input_text.lower() for keyword in bias_keywords):
            suggestions['bias_prevention'].extend([
                'Use inclusive language and avoid stereotypes',
                'Present multiple perspectives when discussing sensitive topics',
                'Be mindful of cultural differences and avoid generalizations'
            ])
        
        # Check for factual claim indicators
        fact_indicators = [
            'statistic', 'study', 'research', 'data', 'evidence', 'scientific',
            'according to', 'studies show', 'experts say'
        ]
        
        if any(indicator in input_text.lower() for indicator in fact_indicators):
            suggestions['accuracy_tips'].extend([
                'Verify all factual claims before including them',
                'Cite reliable sources when possible',
                'Use qualifying language for uncertain information',
                'Distinguish between facts and opinions clearly'
            ])
        
        # General content guidelines
        suggestions['content_guidelines'].extend([
            'Directly address the question asked',
            'Provide practical, actionable information when appropriate',
            'Use clear, concise language',
            'Structure response logically with clear points'
        ])
        
        # Style recommendations
        suggestions['style_recommendations'].extend([
            'Use active voice when possible',
            'Break up long paragraphs for readability',
            'Include examples or illustrations when helpful',
            'End with a clear conclusion or summary'
        ])
        
        return {
            'input_analysis': {
                'contains_sensitive_topics': any(keyword in input_text.lower() for keyword in bias_keywords),
                'requests_factual_information': any(indicator in input_text.lower() for indicator in fact_indicators),
                'complexity_level': 'high' if len(input_text.split()) > 50 else 'medium' if len(input_text.split()) > 20 else 'low'
            },
            'proactive_suggestions': suggestions,
            'recommended_checks': [
                'Review for inclusive language',
                'Verify factual accuracy',
                'Ensure direct relevance to question',
                'Check for practical value'
            ]
        }

# Global instance
self_correction_engine = SelfCorrectionEngine()
