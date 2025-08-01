"""
User Experience Evaluation Module
Implements user experience and interaction quality assessments
"""

import re
import statistics
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import Counter, defaultdict
import logging

logger = logging.getLogger(__name__)


class UserExperienceEvaluator:
    """
    Evaluates user experience metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.interaction_history = defaultdict(list)
        self.user_feedback_patterns = self._load_feedback_patterns()
    
    async def evaluate_user_experience(self, query: str, response: str, 
                                     context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main user experience evaluation method called by comprehensive evaluator
        """
        evaluation_context = {
            'input_text': query,
            'output_text': response,
            'user_context': context.get('user_context', {}) if context else {}
        }
        
        # Run comprehensive UX evaluation
        results = await self.evaluate_all(evaluation_context)
        
        # Calculate overall UX score
        satisfaction_score = results.get('implicit_satisfaction_score', 0.8)
        helpfulness_score = results.get('actionability', 0.8) 
        engagement_score = results.get('conversational_engagement', 0.8)
        accessibility_score = results.get('language_clarity', 0.8)
        
        overall_score = np.mean([satisfaction_score, helpfulness_score, engagement_score, accessibility_score])
        
        return {
            'overall_score': float(overall_score),
            'meets_enterprise_standards': bool(overall_score >= 0.7),
            'detailed_scores': results
        }

    async def evaluate_ux(self, query: str, response: str, 
                         session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Alternative UX evaluation method called by API
        """
        evaluation_context = {
            'input_text': query,
            'output_text': response,
            'user_context': {'session_id': session_id}
        }
        
        # Run comprehensive UX evaluation
        results = await self.evaluate_all(evaluation_context)
        
        return results
    
    async def evaluate_all(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate all user experience metrics
        """
        results = {}
        
        input_text = context['input_text']
        output_text = context['output_text']
        user_context = context.get('user_context', {})
        
        # User Satisfaction & Feedback
        if self.config.get('enable_satisfaction'):
            satisfaction_scores = await self._evaluate_user_satisfaction(
                input_text, output_text, context
            )
            results.update(satisfaction_scores)
        
        # Helpfulness & Usefulness
        if self.config.get('enable_helpfulness'):
            helpfulness_scores = await self._evaluate_helpfulness(
                input_text, output_text, context
            )
            results.update(helpfulness_scores)
        
        # Engagement & Interactivity
        if self.config.get('enable_engagement'):
            engagement_scores = await self._evaluate_engagement(
                input_text, output_text, context
            )
            results.update(engagement_scores)
        
        # Personalization & Customization
        if self.config.get('enable_personalization'):
            personalization_scores = await self._evaluate_personalization(
                input_text, output_text, user_context
            )
            results.update(personalization_scores)
        
        # Conversation Flow & Coherence
        if self.config.get('enable_conversation_flow'):
            flow_scores = await self._evaluate_conversation_flow(
                input_text, output_text, context
            )
            results.update(flow_scores)
        
        # Emotional Intelligence & Empathy
        if self.config.get('enable_emotional_intelligence'):
            emotion_scores = await self._evaluate_emotional_intelligence(
                input_text, output_text
            )
            results.update(emotion_scores)
        
        # Accessibility & Inclusivity
        if self.config.get('enable_accessibility'):
            accessibility_scores = await self._evaluate_accessibility(
                output_text, context
            )
            results.update(accessibility_scores)
        
        # Interface Usability
        if self.config.get('enable_usability'):
            usability_scores = await self._evaluate_interface_usability(
                context
            )
            results.update(usability_scores)
        
        # Error Recovery & Guidance
        if self.config.get('enable_error_recovery'):
            error_scores = await self._evaluate_error_recovery(
                input_text, output_text, context
            )
            results.update(error_scores)
        
        # Learning & Adaptation
        if self.config.get('enable_learning'):
            learning_scores = await self._evaluate_learning_adaptation(
                context
            )
            results.update(learning_scores)
        
        # Trust & Credibility
        if self.config.get('enable_trust'):
            trust_scores = await self._evaluate_trust_credibility(
                input_text, output_text, context
            )
            results.update(trust_scores)
        
        # Response Appropriateness
        if self.config.get('enable_appropriateness'):
            appropriateness_scores = await self._evaluate_response_appropriateness(
                input_text, output_text, context
            )
            results.update(appropriateness_scores)
        
        return results
    
    async def _evaluate_user_satisfaction(self, input_text: str, output_text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate user satisfaction metrics
        """
        results = {}
        
        # Direct Feedback Analysis
        user_feedback = context.get('user_feedback', '')
        if user_feedback:
            results['direct_satisfaction_score'] = await self._analyze_feedback_sentiment(user_feedback)
        
        # Implicit Satisfaction Indicators
        session_duration = context.get('session_duration', 0)
        follow_up_questions = context.get('follow_up_questions', 0)
        
        if session_duration > 0 or follow_up_questions > 0:
            # Use actual session data if available
            session_score = min(session_duration / 300.0, 1.0)  # Normalize to 5 minutes
            follow_up_score = min(follow_up_questions / 3.0, 1.0)  # Normalize to 3 questions
            results['implicit_satisfaction_score'] = (session_score + follow_up_score) / 2
        else:
            # Analyze output quality indicators for satisfaction prediction
            satisfaction_indicators = [
                'helpful', 'useful', 'thank you', 'exactly what', 'perfect',
                'great explanation', 'clear', 'comprehensive', 'detailed'
            ]
            
            quality_indicators = [
                len(output_text.split()) > 50,  # Substantial response
                output_text.count('.') > 2,     # Well-structured
                any(char in output_text for char in [':', '-', '•']),  # Organized
                len(output_text.split('\n')) > 2  # Multi-paragraph
            ]
            
            # Predict satisfaction based on response quality
            quality_score = sum(quality_indicators) / len(quality_indicators)
            results['implicit_satisfaction_score'] = quality_score
        
        # Response Completeness
        completeness_score = await self._assess_response_completeness(input_text, output_text)
        results['response_completeness'] = completeness_score
        
        # User Intent Fulfillment
        intent_fulfillment = await self._assess_intent_fulfillment(input_text, output_text)
        results['intent_fulfillment'] = intent_fulfillment
        
        return results
    
    async def _evaluate_helpfulness(self, input_text: str, output_text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate helpfulness and usefulness
        """
        results = {}
        
        # Actionability
        actionability_score = await self._assess_actionability(output_text)
        results['actionability'] = actionability_score
        
        # Practical Value
        practical_value = await self._assess_practical_value(input_text, output_text)
        results['practical_value'] = practical_value
        
        # Information Richness
        information_richness = await self._assess_information_richness(output_text)
        results['information_richness'] = information_richness
        
        # Problem Solving Effectiveness
        problem_solving = await self._assess_problem_solving_effectiveness(
            input_text, output_text
        )
        results['problem_solving_effectiveness'] = problem_solving
        
        # Resource Provision
        resource_provision = await self._assess_resource_provision(output_text)
        results['resource_provision'] = resource_provision
        
        return results
    
    async def _evaluate_engagement(self, input_text: str, output_text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate engagement and interactivity
        """
        results = {}
        
        # Conversational Engagement
        conversation_engagement = await self._assess_conversational_engagement(output_text)
        results['conversational_engagement'] = conversation_engagement
        
        # Question Generation
        question_generation = await self._assess_question_generation(output_text)
        results['question_generation'] = question_generation
        
        # Interactive Elements
        interactive_elements = await self._assess_interactive_elements(output_text)
        results['interactive_elements'] = interactive_elements
        
        # Curiosity Stimulation
        curiosity_stimulation = await self._assess_curiosity_stimulation(output_text)
        results['curiosity_stimulation'] = curiosity_stimulation
        
        # Response Variety
        response_variety = await self._assess_response_variety(output_text, context)
        results['response_variety'] = response_variety
        
        return results
    
    async def _evaluate_personalization(self, input_text: str, output_text: str, user_context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate personalization and customization
        """
        results = {}
        
        # Context Awareness
        context_awareness = await self._assess_context_awareness(
            input_text, output_text, user_context
        )
        results['context_awareness'] = context_awareness
        
        # User Preference Alignment
        preference_alignment = await self._assess_preference_alignment(
            output_text, user_context
        )
        results['preference_alignment'] = preference_alignment
        
        # Adaptive Communication Style
        communication_adaptation = await self._assess_communication_adaptation(
            output_text, user_context
        )
        results['communication_adaptation'] = communication_adaptation
        
        # Learning from Interaction
        learning_evidence = await self._assess_learning_evidence(
            output_text, user_context
        )
        results['learning_evidence'] = learning_evidence
        
        return results
    
    async def _evaluate_conversation_flow(self, input_text: str, output_text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate conversation flow and coherence
        """
        results = {}
        
        # Turn-Taking Appropriateness
        turn_taking = await self._assess_turn_taking(input_text, output_text, context)
        results['turn_taking_appropriateness'] = turn_taking
        
        # Topic Continuity
        topic_continuity = await self._assess_topic_continuity(
            input_text, output_text, context
        )
        results['topic_continuity'] = topic_continuity
        
        # Transition Smoothness
        transition_smoothness = await self._assess_transition_smoothness(
            output_text, context
        )
        results['transition_smoothness'] = transition_smoothness
        
        # Contextual Coherence
        contextual_coherence = await self._assess_contextual_coherence(
            input_text, output_text, context
        )
        results['contextual_coherence'] = contextual_coherence
        
        return results
    
    async def _evaluate_emotional_intelligence(self, input_text: str, output_text: str) -> Dict[str, float]:
        """
        Evaluate emotional intelligence and empathy
        """
        results = {}
        
        # Emotion Recognition
        emotion_recognition = await self._assess_emotion_recognition(input_text, output_text)
        results['emotion_recognition'] = emotion_recognition
        
        # Empathetic Response
        empathetic_response = await self._assess_empathetic_response(input_text, output_text)
        results['empathetic_response'] = empathetic_response
        
        # Emotional Appropriateness
        emotional_appropriateness = await self._assess_emotional_appropriateness(
            input_text, output_text
        )
        results['emotional_appropriateness'] = emotional_appropriateness
        
        # Supportive Language
        supportive_language = await self._assess_supportive_language(output_text)
        results['supportive_language'] = supportive_language
        
        return results
    
    async def _evaluate_accessibility(self, output_text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate accessibility and inclusivity
        """
        results = {}
        
        # Language Clarity
        language_clarity = await self._assess_language_clarity(output_text)
        results['language_clarity'] = language_clarity
        
        # Reading Level Appropriateness
        reading_level = await self._assess_reading_level_appropriateness(
            output_text, context
        )
        results['reading_level_appropriateness'] = reading_level
        
        # Inclusive Language
        inclusive_language = await self._assess_inclusive_language(output_text)
        results['inclusive_language'] = inclusive_language
        
        # Accommodation Support
        accommodation_support = await self._assess_accommodation_support(
            output_text, context
        )
        results['accommodation_support'] = accommodation_support
        
        return results
    
    async def _evaluate_interface_usability(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate interface usability
        """
        results = {}
        
        # Interface Intuitiveness
        interface_metrics = context.get('interface_metrics', {})
        
        # Click-through rate
        ctr = interface_metrics.get('click_through_rate', 0.5)
        results['interface_intuitiveness'] = ctr
        
        # Task Completion Rate
        completion_rate = interface_metrics.get('task_completion_rate', 0.5)
        results['task_completion_rate'] = completion_rate
        
        # Navigation Efficiency
        navigation_steps = interface_metrics.get('navigation_steps', 5)
        navigation_efficiency = max(0, 1 - ((navigation_steps - 3) / 10))
        results['navigation_efficiency'] = navigation_efficiency
        
        # Error Rate
        error_rate = interface_metrics.get('error_rate', 0.1)
        results['low_error_rate'] = max(0, 1 - (error_rate / 0.2))
        
        return results
    
    async def _evaluate_error_recovery(self, input_text: str, output_text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate error recovery and guidance
        """
        results = {}
        
        # Error Detection
        error_detection = await self._assess_error_detection(input_text, output_text)
        results['error_detection'] = error_detection
        
        # Recovery Guidance
        recovery_guidance = await self._assess_recovery_guidance(output_text)
        results['recovery_guidance'] = recovery_guidance
        
        # Graceful Degradation
        graceful_degradation = await self._assess_graceful_degradation(
            output_text, context
        )
        results['graceful_degradation'] = graceful_degradation
        
        return results
    
    async def _evaluate_learning_adaptation(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate learning and adaptation capabilities
        """
        results = {}
        
        # Performance Improvement
        performance_history = context.get('performance_history', [0.5])
        if len(performance_history) >= 2:
            improvement = performance_history[-1] - performance_history[0]
            improvement_score = max(0, min(improvement + 0.5, 1.0))
            results['performance_improvement'] = improvement_score
        
        # User Preference Learning
        preference_accuracy = context.get('preference_accuracy', 0.5)
        results['preference_learning'] = preference_accuracy
        
        # Adaptation Speed
        adaptation_time = context.get('adaptation_time', 10)
        adaptation_score = max(0, 1 - (adaptation_time / 20))
        results['adaptation_speed'] = adaptation_score
        
        return results
    
    async def _evaluate_trust_credibility(self, input_text: str, output_text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate trust and credibility factors
        """
        results = {}
        
        # Source Citation
        source_citation = await self._assess_source_citation(output_text)
        results['source_citation'] = source_citation
        
        # Confidence Indicators
        confidence_indicators = await self._assess_confidence_indicators(output_text)
        results['confidence_indicators'] = confidence_indicators
        
        # Transparency
        transparency = await self._assess_transparency(output_text)
        results['transparency'] = transparency
        
        # Consistency
        consistency = await self._assess_response_consistency(
            output_text, context
        )
        results['consistency'] = consistency
        
        return results
    
    async def _evaluate_response_appropriateness(self, input_text: str, output_text: str, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate response appropriateness
        """
        results = {}
        
        # Context Appropriateness
        context_appropriateness = await self._assess_context_appropriateness(
            input_text, output_text, context
        )
        results['context_appropriateness'] = context_appropriateness
        
        # Tone Appropriateness
        tone_appropriateness = await self._assess_tone_appropriateness(
            input_text, output_text
        )
        results['tone_appropriateness'] = tone_appropriateness
        
        # Length Appropriateness
        length_appropriateness = await self._assess_length_appropriateness(
            input_text, output_text
        )
        results['length_appropriateness'] = length_appropriateness
        
        # Cultural Appropriateness
        cultural_appropriateness = await self._assess_cultural_appropriateness(
            output_text, context
        )
        results['cultural_appropriateness'] = cultural_appropriateness
        
        return results
    
    # Helper methods
    def _load_feedback_patterns(self) -> Dict[str, List[str]]:
        """Load user feedback sentiment patterns"""
        return {
            'positive': [
                'great', 'excellent', 'helpful', 'useful', 'perfect',
                'amazing', 'wonderful', 'fantastic', 'love it', 'thank you'
            ],
            'negative': [
                'terrible', 'awful', 'useless', 'unhelpful', 'wrong',
                'bad', 'hate', 'disappointed', 'frustrated', 'confusing'
            ],
            'neutral': [
                'okay', 'fine', 'average', 'decent', 'acceptable'
            ]
        }
    
    async def _analyze_feedback_sentiment(self, feedback: str) -> float:
        """Analyze sentiment of user feedback"""
        feedback_lower = feedback.lower()
        
        positive_count = sum(1 for word in self.user_feedback_patterns['positive']
                           if word in feedback_lower)
        negative_count = sum(1 for word in self.user_feedback_patterns['negative']
                           if word in feedback_lower)
        
        if positive_count + negative_count == 0:
            return 0.5  # Neutral
        
        sentiment_score = positive_count / (positive_count + negative_count)
        return sentiment_score
    
    async def _assess_response_completeness(self, input_text: str, output_text: str) -> float:
        """Assess how complete the response is"""
        # Check if all question components are addressed
        question_indicators = ['what', 'why', 'how', 'when', 'where', 'who']
        
        questions_asked = sum(1 for indicator in question_indicators
                            if indicator in input_text.lower())
        
        if questions_asked == 0:
            return 1.0  # No specific questions to answer
        
        # Simple heuristic: longer responses are more likely to be complete
        response_length = len(output_text.split())
        expected_length = questions_asked * 20  # 20 words per question component
        
        completeness = min(response_length / expected_length, 1.0)
        return completeness
    
    async def _assess_intent_fulfillment(self, input_text: str, output_text: str) -> float:
        """Assess how well the response fulfills user intent"""
        # Extract key terms from input
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'
        }
        
        input_content = input_words - stop_words
        output_content = output_words - stop_words
        
        if not input_content:
            return 1.0
        
        # Calculate overlap
        overlap = len(input_content.intersection(output_content))
        fulfillment_score = overlap / len(input_content)
        
        return min(fulfillment_score, 1.0)
    
    async def _assess_actionability(self, text: str) -> float:
        """Assess how actionable the response is"""
        actionable_indicators = [
            'you can', 'you should', 'try', 'consider', 'follow these steps',
            'first', 'next', 'then', 'finally', 'recommendation', 'suggest'
        ]
        
        actionability_count = sum(1 for indicator in actionable_indicators
                                if indicator in text.lower())
        
        # Also check for numbered lists or bullet points
        if re.search(r'\d+\.|\*|\-', text):
            actionability_count += 1
        
        return min(actionability_count / 3.0, 1.0)
    
    async def _assess_practical_value(self, input_text: str, output_text: str) -> float:
        """Assess practical value of the response"""
        practical_indicators = [
            'example', 'tool', 'resource', 'link', 'website', 'app',
            'method', 'technique', 'approach', 'solution', 'tip'
        ]
        
        practical_count = sum(1 for indicator in practical_indicators
                            if indicator in output_text.lower())
        
        return min(practical_count / 3.0, 1.0)
    
    async def _assess_information_richness(self, text: str) -> float:
        """Assess information richness"""
        # Count different types of information
        info_types = 0
        
        # Numbers/statistics
        if re.search(r'\d+%|\d+\.\d+|\d+,\d+', text):
            info_types += 1
        
        # Dates
        if re.search(r'\d{4}|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', text, re.IGNORECASE):
            info_types += 1
        
        # Names/proper nouns
        if re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text):
            info_types += 1
        
        # Technical terms (words with 8+ characters)
        long_words = len([word for word in text.split() if len(word) >= 8])
        if long_words >= 3:
            info_types += 1
        
        return min(info_types / 4.0, 1.0)
    
    async def _assess_problem_solving_effectiveness(self, input_text: str, output_text: str) -> float:
        """Assess problem-solving effectiveness"""
        problem_indicators = ['problem', 'issue', 'challenge', 'difficulty', 'trouble']
        solution_indicators = ['solution', 'solve', 'fix', 'resolve', 'answer']
        
        has_problem = any(indicator in input_text.lower() for indicator in problem_indicators)
        has_solution = any(indicator in output_text.lower() for indicator in solution_indicators)
        
        if not has_problem:
            return 1.0  # No problem to solve
        
        if has_solution:
            return 1.0
        else:
            # Check for implicit solutions (actionable advice)
            actionability = await self._assess_actionability(output_text)
            return actionability
    
    async def _assess_resource_provision(self, text: str) -> float:
        """Assess provision of resources"""
        resource_indicators = [
            'http', 'www', 'website', 'link', 'url', 'reference',
            'book', 'article', 'study', 'research', 'documentation'
        ]
        
        resource_count = sum(1 for indicator in resource_indicators
                           if indicator in text.lower())
        
        return min(resource_count / 2.0, 1.0)
    
    async def _assess_conversational_engagement(self, text: str) -> float:
        """Assess conversational engagement"""
        engagement_indicators = [
            'what do you think', 'would you like', 'are you interested',
            'have you considered', 'let me know', 'feel free to ask'
        ]
        
        engagement_count = sum(1 for indicator in engagement_indicators
                             if indicator in text.lower())
        
        # Check for questions
        question_count = text.count('?')
        
        total_engagement = engagement_count + min(question_count, 2)
        return min(total_engagement / 3.0, 1.0)
    
    async def _assess_question_generation(self, text: str) -> float:
        """Assess question generation for engagement"""
        questions = text.count('?')
        return min(questions / 3.0, 1.0)
    
    async def _assess_interactive_elements(self, text: str) -> float:
        """Assess interactive elements in response"""
        # Direct interactive elements
        interactive_indicators = [
            'click', 'choose', 'select', 'option', 'menu', 'button',
            'interactive', 'explore', 'try', 'experiment', 'test', 'check'
        ]
        
        # Question-based interactivity
        question_patterns = [
            r'\?', r'would you like', r'do you want', r'have you tried',
            r'what do you think', r'how about', r'consider'
        ]
        
        # Engagement elements
        engagement_indicators = [
            'let me know', 'tell me', 'share', 'feedback', 'thoughts',
            'experience', 'opinion', 'prefer'
        ]
        
        score = 0.0
        
        # Check for direct interactive elements
        interactive_count = sum(1 for indicator in interactive_indicators
                              if indicator in text.lower())
        score += min(interactive_count * 0.2, 0.4)
        
        # Check for questions and prompts
        question_count = sum(1 for pattern in question_patterns
                           if re.search(pattern, text, re.IGNORECASE))
        score += min(question_count * 0.15, 0.3)
        
        # Check for engagement elements
        engagement_count = sum(1 for indicator in engagement_indicators
                             if indicator in text.lower())
        score += min(engagement_count * 0.1, 0.3)
        
        return min(score, 1.0)
    
    async def _assess_curiosity_stimulation(self, text: str) -> float:
        """Assess how well the response stimulates curiosity"""
        # Direct curiosity indicators
        curiosity_indicators = [
            'interesting', 'fascinating', 'surprising', 'did you know',
            'fun fact', 'remarkably', 'amazingly', 'curiously', 'wonder',
            'mysterious', 'intriguing', 'unexpected', 'remarkable'
        ]
        
        # Question-generating phrases
        question_generators = [
            'this leads to', 'you might wonder', 'this raises the question',
            'consider this', 'think about', 'imagine if', 'what if',
            'have you ever wondered', 'makes you think'
        ]
        
        # Learning motivators
        learning_motivators = [
            'learn more', 'explore further', 'dive deeper', 'discover',
            'uncover', 'reveals', 'shows us', 'demonstrates'
        ]
        
        # Educational engagement indicators
        educational_indicators = [
            'why is', 'how does', 'what happens', 'the reason',
            'because', 'therefore', 'as a result', 'this means'
        ]
        
        score = 0.0
        
        # Check for curiosity words
        curiosity_count = sum(1 for indicator in curiosity_indicators
                            if indicator in text.lower())
        score += min(curiosity_count * 0.15, 0.3)
        
        # Check for question-generating phrases
        question_count = sum(1 for generator in question_generators
                           if generator in text.lower())
        score += min(question_count * 0.2, 0.3)
        
        # Check for learning motivators
        learning_count = sum(1 for motivator in learning_motivators
                           if motivator in text.lower())
        score += min(learning_count * 0.1, 0.2)
        
        # Check for educational engagement (explanatory content)
        educational_count = sum(1 for indicator in educational_indicators
                              if indicator in text.lower())
        score += min(educational_count * 0.05, 0.2)
        
        # Bonus for explanation patterns that can stimulate curiosity
        explanation_patterns = [
            r'the process (by which|where|of)',
            r'this (happens|occurs|works) (by|when|because)',
            r'(several|many|multiple) reasons?',
            r'not only.*but also',
            r'for example',
            r'such as'
        ]
        
        pattern_count = sum(1 for pattern in explanation_patterns
                          if re.search(pattern, text, re.IGNORECASE))
        score += min(pattern_count * 0.1, 0.2)
        
        return min(score, 1.0)
    
    async def _assess_response_variety(self, text: str, context: Dict[str, Any]) -> float:
        """Assess variety in responses"""
        response_history = context.get('response_history', [])
        
        if len(response_history) < 2:
            return 1.0  # Can't assess variety with less than 2 responses
        
        # Simple similarity check with recent responses
        current_words = set(text.lower().split())
        similarities = []
        
        for past_response in response_history[-3:]:  # Check last 3 responses
            past_words = set(past_response.lower().split())
            if current_words and past_words:
                overlap = len(current_words.intersection(past_words))
                total = len(current_words.union(past_words))
                similarity = overlap / total if total > 0 else 0
                similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        variety_score = 1.0 - avg_similarity
        
        return variety_score
    
    async def _assess_context_awareness(self, input_text: str, output_text: str, user_context: Dict[str, Any]) -> float:
        """Assess context awareness"""
        context_score = 0.0
        
        # Check if user preferences are reflected (if available)
        user_preferences = user_context.get('preferences', {})
        if user_preferences:
            preference_alignment = 0
            for key, value in user_preferences.items():
                if str(value).lower() in output_text.lower():
                    preference_alignment += 1
            
            if user_preferences:
                context_score += (preference_alignment / len(user_preferences)) * 0.3
        
        # Check for contextual references in the response
        contextual_indicators = [
            'as you mentioned', 'based on your', 'given that you', 'since you',
            'your previous', 'earlier you said', 'you asked about', 'following up',
            'in your case', 'for your situation', 'considering your'
        ]
        
        contextual_count = sum(1 for indicator in contextual_indicators
                             if indicator in output_text.lower())
        context_score += min(contextual_count * 0.2, 0.4)
        
        # Check if response references input context
        input_words = set(input_text.lower().split())
        output_words = set(output_text.lower().split())
        
        # Look for meaningful word overlap (excluding common words)
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        meaningful_input = input_words - common_words
        meaningful_output = output_words - common_words
        
        if meaningful_input:
            overlap = len(meaningful_input.intersection(meaningful_output))
            overlap_ratio = overlap / len(meaningful_input)
            context_score += min(overlap_ratio * 0.3, 0.3)
        
        # Check if conversation history is referenced (if available)
        conversation_history = user_context.get('conversation_history', [])
        if len(conversation_history) > 1:
            # Check for references to previous topics
            previous_topics = set()
            for msg in conversation_history[:-1]:  # Exclude current message
                previous_topics.update(msg.get('content', '').lower().split())
            
            current_words = set(output_text.lower().split())
            if previous_topics and current_words:
                topic_overlap = len(previous_topics.intersection(current_words))
                context_score += min(topic_overlap / 20, 0.2)
        else:
            # If no conversation history, give base score for standalone context awareness
            context_score += 0.2
        
        return min(context_score, 1.0)
    
    async def _assess_preference_alignment(self, text: str, user_context: Dict[str, Any]) -> float:
        """Assess alignment with user preferences"""
        preferences = user_context.get('preferences', {})
        
        if not preferences:
            return 0.5  # No preferences to align with
        
        alignment_score = 0.0
        
        # Communication style preference
        comm_style = preferences.get('communication_style', '')
        if comm_style == 'formal' and not re.search(r'\b(hey|hi|gonna|wanna)\b', text, re.IGNORECASE):
            alignment_score += 0.3
        elif comm_style == 'casual' and re.search(r'\b(hey|hi|you know|like)\b', text, re.IGNORECASE):
            alignment_score += 0.3
        
        # Detail preference
        detail_level = preferences.get('detail_level', '')
        word_count = len(text.split())
        if detail_level == 'brief' and word_count <= 50:
            alignment_score += 0.3
        elif detail_level == 'detailed' and word_count >= 100:
            alignment_score += 0.3
        
        # Topic interests
        interests = preferences.get('interests', [])
        if interests:
            interest_mentions = sum(1 for interest in interests
                                  if interest.lower() in text.lower())
            alignment_score += min(interest_mentions / len(interests), 0.4)
        
        return min(alignment_score, 1.0)
    
    async def _assess_communication_adaptation(self, text: str, user_context: Dict[str, Any]) -> float:
        """Assess communication style adaptation"""
        adaptation_score = 0.0
        
        # Direct adaptation indicators
        adaptation_indicators = [
            'as you prefer', 'in your style', 'to match your needs',
            'based on your feedback', 'adjusting my response', 'tailored to you'
        ]
        
        adaptation_count = sum(1 for indicator in adaptation_indicators
                             if indicator in text.lower())
        adaptation_score += min(adaptation_count * 0.3, 0.4)
        
        # Communication style indicators
        style_indicators = [
            'simple terms', 'technical detail', 'step by step', 'briefly',
            'in detail', 'concisely', 'thoroughly', 'quickly'
        ]
        
        style_count = sum(1 for indicator in style_indicators
                        if indicator in text.lower())
        adaptation_score += min(style_count * 0.2, 0.3)
        
        # Audience awareness
        audience_indicators = [
            'for beginners', 'advanced users', 'professionals', 'students',
            'depending on your level', 'appropriate for your needs'
        ]
        
        audience_count = sum(1 for indicator in audience_indicators
                           if indicator in text.lower())
        adaptation_score += min(audience_count * 0.25, 0.3)
        
        return min(adaptation_score, 1.0)
    
    async def _assess_learning_evidence(self, text: str, user_context: Dict[str, Any]) -> float:
        """Assess evidence of learning from interactions"""
        learning_score = 0.0
        
        # Direct learning indicators
        learning_indicators = [
            'i remember', 'as we discussed', 'from our previous conversation',
            'you mentioned', 'building on what', 'following up on',
            'earlier you said', 'continuing from', 'based on our chat'
        ]
        
        learning_count = sum(1 for indicator in learning_indicators
                           if indicator in text.lower())
        learning_score += min(learning_count * 0.3, 0.3)
        
        # Knowledge building indicators
        building_indicators = [
            'building on', 'expanding on', 'connecting to', 'relates to',
            'similar to what', 'like we talked about', 'as mentioned before'
        ]
        
        building_count = sum(1 for indicator in building_indicators
                           if indicator in text.lower())
        learning_score += min(building_count * 0.25, 0.25)
        
        # Progressive understanding indicators
        progress_indicators = [
            'now that you know', 'having learned', 'since you understand',
            'with this knowledge', 'next step', 'moving forward'
        ]
        
        progress_count = sum(1 for indicator in progress_indicators
                           if indicator in text.lower())
        learning_score += min(progress_count * 0.2, 0.2)
        
        # Educational structure indicators (evidence of systematic learning approach)
        structure_indicators = [
            r'\d+\.\s+\*\*.*\*\*:',  # Numbered bold points
            r'first.*second.*third',  # Sequential structure
            r'step \d+',  # Step-by-step approach
            r'(let\'s|we can) (start|begin)',  # Learning initiation
            r'(next|then|finally)',  # Learning progression
        ]
        
        structure_count = sum(1 for pattern in structure_indicators
                            if re.search(pattern, text, re.IGNORECASE))
        learning_score += min(structure_count * 0.1, 0.25)
        
        return min(learning_score, 1.0)
    
    # Placeholder implementations for remaining assessment methods
    async def _assess_turn_taking(self, input_text: str, output_text: str, context: Dict[str, Any]) -> float:
        return 0.8  # Placeholder
    
    async def _assess_topic_continuity(self, input_text: str, output_text: str, context: Dict[str, Any]) -> float:
        return 0.8  # Placeholder
    
    async def _assess_transition_smoothness(self, output_text: str, context: Dict[str, Any]) -> float:
        return 0.8  # Placeholder
    
    async def _assess_contextual_coherence(self, input_text: str, output_text: str, context: Dict[str, Any]) -> float:
        return 0.8  # Placeholder
    
    async def _assess_emotion_recognition(self, input_text: str, output_text: str) -> float:
        # Check if emotional cues in input are acknowledged in output
        emotion_words = ['sad', 'happy', 'angry', 'frustrated', 'excited', 'worried', 'confused']
        
        input_emotions = [word for word in emotion_words if word in input_text.lower()]
        if not input_emotions:
            return 1.0  # No emotions to recognize
        
        acknowledgment_phrases = ['understand', 'feel', 'seem', 'sounds like', 'i can see']
        acknowledges_emotion = any(phrase in output_text.lower() for phrase in acknowledgment_phrases)
        
        return 1.0 if acknowledges_emotion else 0.5
    
    async def _assess_empathetic_response(self, input_text: str, output_text: str) -> float:
        empathy_indicators = [
            'i understand', 'that must be', 'i can imagine', 'sounds difficult',
            'i\'m sorry', 'that\'s tough', 'i feel for you', 'understandable'
        ]
        
        empathy_count = sum(1 for indicator in empathy_indicators
                          if indicator in output_text.lower())
        
        return min(empathy_count / 2.0, 1.0)
    
    async def _assess_emotional_appropriateness(self, input_text: str, output_text: str) -> float:
        # Simple check for emotional tone matching
        negative_input = any(word in input_text.lower() 
                           for word in ['sad', 'angry', 'frustrated', 'upset', 'disappointed'])
        positive_output = any(word in output_text.lower()
                            for word in ['great', 'awesome', 'fantastic', 'excellent'])
        
        if negative_input and positive_output:
            return 0.3  # Inappropriate mismatch
        else:
            return 0.8  # Generally appropriate
    
    async def _assess_supportive_language(self, text: str) -> float:
        supportive_indicators = [
            'you can do it', 'don\'t worry', 'it\'s okay', 'you\'re not alone',
            'here to help', 'support you', 'believe in you', 'you\'ve got this'
        ]
        
        supportive_count = sum(1 for indicator in supportive_indicators
                             if indicator in text.lower())
        
        return min(supportive_count / 2.0, 1.0)
    
    async def _assess_language_clarity(self, text: str) -> float:
        # Simple readability assessment
        words = text.split()
        if not words:
            return 0.0
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Sentence length
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Clarity score (shorter words and sentences are clearer)
        word_clarity = max(0, 1 - ((avg_word_length - 5) / 10))
        sentence_clarity = max(0, 1 - ((avg_sentence_length - 15) / 25))
        
        return (word_clarity + sentence_clarity) / 2
    
    async def _assess_reading_level_appropriateness(self, text: str, context: Dict[str, Any]) -> float:
        target_level = context.get('target_reading_level', 'adult')
        
        # Simple assessment based on word complexity
        words = text.split()
        complex_words = sum(1 for word in words if len(word) > 7)
        complexity_ratio = complex_words / len(words) if words else 0
        
        if target_level == 'child' and complexity_ratio <= 0.1:
            return 1.0
        elif target_level == 'adult' and 0.1 <= complexity_ratio <= 0.3:
            return 1.0
        elif target_level == 'expert' and complexity_ratio >= 0.2:
            return 1.0
        else:
            return 0.6  # Moderate appropriateness
    
    async def _assess_inclusive_language(self, text: str) -> float:
        # Check for inclusive language patterns
        inclusive_indicators = [
            'everyone', 'all people', 'regardless of', 'inclusive',
            'diverse', 'accessible', 'for all', 'people of all'
        ]
        
        # Check for potentially exclusive language
        exclusive_patterns = [
            'normal people', 'regular people', 'you guys', 'mankind'
        ]
        
        inclusive_count = sum(1 for indicator in inclusive_indicators
                            if indicator in text.lower())
        exclusive_count = sum(1 for pattern in exclusive_patterns
                            if pattern in text.lower())
        
        score = min(inclusive_count / 2.0, 1.0) - (exclusive_count * 0.3)
        return max(0, score)
    
    async def _assess_accommodation_support(self, text: str, context: Dict[str, Any]) -> float:
        return 0.8  # Placeholder
    
    async def _assess_error_detection(self, input_text: str, output_text: str) -> float:
        return 0.8  # Placeholder
    
    async def _assess_recovery_guidance(self, text: str) -> float:
        return 0.8  # Placeholder
    
    async def _assess_graceful_degradation(self, text: str, context: Dict[str, Any]) -> float:
        return 0.8  # Placeholder
    
    async def _assess_source_citation(self, text: str) -> float:
        citation_indicators = ['source:', 'according to', 'reference:', 'cited from', 'based on']
        citation_count = sum(1 for indicator in citation_indicators if indicator in text.lower())
        return min(citation_count / 2.0, 1.0)
    
    async def _assess_confidence_indicators(self, text: str) -> float:
        confidence_indicators = ['certain', 'confident', 'sure', 'definitely', 'likely', 'probably']
        confidence_count = sum(1 for indicator in confidence_indicators if indicator in text.lower())
        return min(confidence_count / 3.0, 1.0)
    
    async def _assess_transparency(self, text: str) -> float:
        transparency_indicators = ['i don\'t know', 'uncertain', 'unclear', 'limitations', 'may not be accurate']
        transparency_count = sum(1 for indicator in transparency_indicators if indicator in text.lower())
        return min(transparency_count / 2.0, 1.0)
    
    async def _assess_response_consistency(self, text: str, context: Dict[str, Any]) -> float:
        return 0.8  # Placeholder
    
    async def _assess_context_appropriateness(self, input_text: str, output_text: str, context: Dict[str, Any]) -> float:
        return 0.8  # Placeholder
    
    async def _assess_tone_appropriateness(self, input_text: str, output_text: str) -> float:
        return 0.8  # Placeholder
    
    async def _assess_length_appropriateness(self, input_text: str, output_text: str) -> float:
        input_length = len(input_text.split())
        output_length = len(output_text.split())
        
        # Reasonable ratio is 2-5x input length
        if input_length > 0:
            ratio = output_length / input_length
            if 2 <= ratio <= 5:
                return 1.0
            elif ratio < 2:
                return ratio / 2.0
            else:
                return max(0, 1 - ((ratio - 5) / 10))
        
        return 0.8
    
    async def _assess_cultural_appropriateness(self, text: str, context: Dict[str, Any]) -> float:
        return 0.8  # Placeholder
