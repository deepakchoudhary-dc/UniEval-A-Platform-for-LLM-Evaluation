"""
Quality & Performance Evaluation Module
Implements foundational quality metrics and performanc        return {
            'overall_score': float(overall_quality_score),
            'meets_enterprise_standards': bool(overall_quality_score >= 0.8),
            'detailed_scores': {
                'accuracy_score': float(accuracy_score),
                'relevance_score': float(relevance_score),
                'coherence_score': float(coherence_score),
                'completeness_score': float(completeness_score),
                'fluency_score': float(fluency_score),
                'raw_evaluation': results
            }
        }essments
"""

import re
import math
import statistics
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import Counter
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import textstat
import logging

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    bert_score = None
    BERT_SCORE_AVAILABLE = False

try:
    from moverscore_v2 import get_idf_dict, word_mover_score
    MOVERSCORE_AVAILABLE = True
except ImportError:
    word_mover_score = None
    MOVERSCORE_AVAILABLE = False

try:
    from comet import download_model, load_from_checkpoint
    COMET_AVAILABLE = True
except ImportError:
    load_from_checkpoint = None
    COMET_AVAILABLE = False

logger = logging.getLogger(__name__)


class QualityPerformanceEvaluator:
    """
    Evaluates foundational quality and performance metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sentence_model = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing_function = SmoothingFunction().method1
        
        # Initialize models if needed
        if self.config.get('enable_bertscore') or self.config.get('enable_moverscore'):
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                except Exception as e:
                    logger.warning(f"Could not load sentence transformer: {e}")
            else:
                logger.warning("SentenceTransformers not available, skipping model initialization")
    
    async def evaluate_quality_performance(self, query: str, response: str, 
                                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main quality and performance evaluation method called by comprehensive evaluator
        """
        evaluation_context = {
            'input_text': query,
            'output_text': response,
            'reference_text': context.get('reference_text') if context else None,
            'context': context.get('context') if context else None
        }
        
        # Run comprehensive quality evaluation
        results = await self.evaluate_all(evaluation_context)
        
        # Calculate component scores
        accuracy_score = results.get('factual_correctness', 0.8)
        relevance_score = results.get('relevancy_score', 0.8)
        coherence_score = results.get('coherence_score', 0.8)
        completeness_score = results.get('completeness_score', 0.8)
        fluency_score = results.get('fluency_score', 0.8)
        
        # Calculate overall quality score
        overall_quality_score = np.mean([
            accuracy_score, relevance_score, coherence_score, 
            completeness_score, fluency_score
        ])
        
        return {
            'overall_score': overall_quality_score,
            'meets_enterprise_standards': overall_quality_score >= 0.7,
            'detailed_scores': results
        }

    async def evaluate_quality(self, query: str, response: str, 
                             context: Optional[str] = None, 
                             expected_output: Optional[str] = None) -> Dict[str, Any]:
        """
        Main quality evaluation method called by API
        """
        evaluation_context = {
            'input_text': query,
            'output_text': response,
            'reference_text': expected_output,
            'context': context
        }
        
        # Run comprehensive quality evaluation
        results = await self.evaluate_all(evaluation_context)
        
        # Calculate component scores
        accuracy_score = results.get('factual_correctness', 0.8)
        relevance_score = results.get('relevancy_score', 0.8)
        coherence_score = results.get('coherence_score', 0.8)
        completeness_score = results.get('completeness_score', 0.8)
        fluency_score = results.get('fluency_score', 0.8)
        
        # Calculate overall quality score
        overall_quality_score = np.mean([
            accuracy_score, relevance_score, coherence_score, 
            completeness_score, fluency_score
        ])
        
        return {
            'overall_quality_score': overall_quality_score,
            'accuracy_score': accuracy_score,
            'relevance_score': relevance_score,
            'coherence_score': coherence_score,
            'completeness_score': completeness_score,
            'fluency_score': fluency_score,
            'detailed_metrics': results
        }
    
    async def evaluate_all(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate all quality and performance metrics
        """
        results = {}
        
        input_text = context['input_text']
        output_text = context['output_text']
        reference_text = context.get('reference_text')
        
        # Factual Correctness
        if self.config.get('enable_factual_correctness'):
            results['factual_correctness'] = await self._evaluate_factual_correctness(
                input_text, output_text, reference_text
            )
        
        # Task-Specific Accuracy
        if self.config.get('enable_task_accuracy'):
            em_score, f1_score = await self._evaluate_task_accuracy(output_text, reference_text)
            results['exact_match'] = em_score
            results['f1_score'] = f1_score
        
        # Faithfulness / Groundedness
        if self.config.get('enable_faithfulness'):
            results['faithfulness'] = await self._evaluate_faithfulness(
                input_text, output_text, context.get('context', {})
            )
        
        # Answer Relevancy
        if self.config.get('enable_relevancy'):
            results['answer_relevancy'] = await self._evaluate_answer_relevancy(input_text, output_text)
        
        # Context Relevance
        if self.config.get('enable_relevancy'):
            results['context_relevance'] = await self._evaluate_context_relevance(
                input_text, context.get('context', {})
            )
        
        # Prompt Alignment / Instruction Following
        if self.config.get('enable_relevancy'):
            results['prompt_alignment'] = await self._evaluate_prompt_alignment(input_text, output_text)
        
        # Fluency / Grammaticality
        if self.config.get('enable_fluency'):
            results['fluency'] = await self._evaluate_fluency(output_text)
        
        # Coherence
        if self.config.get('enable_coherence'):
            results['coherence'] = await self._evaluate_coherence(output_text)
        
        # Readability Scores
        if self.config.get('enable_readability'):
            readability_scores = await self._evaluate_readability(output_text)
            results.update(readability_scores)
        
        # BLEU Score
        if self.config.get('enable_bleu') and reference_text:
            results['bleu_score'] = await self._evaluate_bleu(output_text, reference_text)
        
        # ROUGE Scores
        if self.config.get('enable_rouge') and reference_text:
            rouge_scores = await self._evaluate_rouge(output_text, reference_text)
            results.update(rouge_scores)
        
        # METEOR Score
        if self.config.get('enable_meteor') and reference_text:
            results['meteor_score'] = await self._evaluate_meteor(output_text, reference_text)
        
        # BERTScore
        if self.config.get('enable_bertscore') and reference_text and bert_score:
            results['bert_score'] = await self._evaluate_bert_score(output_text, reference_text)
        
        # MoverScore
        if self.config.get('enable_moverscore') and reference_text and word_mover_score:
            results['mover_score'] = await self._evaluate_mover_score(output_text, reference_text)
        
        # COMET Score
        if self.config.get('enable_comet') and reference_text and load_from_checkpoint:
            results['comet_score'] = await self._evaluate_comet(input_text, output_text, reference_text)
        
        # Diversity and Creativity
        if self.config.get('enable_diversity'):
            diversity_scores = await self._evaluate_diversity(output_text)
            results.update(diversity_scores)
        
        # MAUVE Score
        if self.config.get('enable_mauve') and reference_text:
            results['mauve_score'] = await self._evaluate_mauve(output_text, reference_text)
        
        # Perplexity and Cross-Entropy
        if self.config.get('enable_perplexity'):
            perplexity_scores = await self._evaluate_perplexity(output_text)
            results.update(perplexity_scores)
        
        return results
    
    async def _evaluate_factual_correctness(self, input_text: str, output_text: str, reference_text: Optional[str]) -> float:
        """
        Evaluate factual correctness of the output
        """
        # Basic factual correctness evaluation
        score = 0.0
        
        # Check for factual claims that can be verified
        factual_indicators = [
            r'\b\d{4}\b',  # Years
            r'\b\d+\s*(percent|%)\b',  # Percentages
            r'\b\d+\s*(km|miles|meters|feet)\b',  # Distances
            r'\b\d+\s*(kg|pounds|grams)\b',  # Weights
        ]
        
        claims_found = 0
        for pattern in factual_indicators:
            if re.search(pattern, output_text, re.IGNORECASE):
                claims_found += 1
                # If reference exists, check consistency
                if reference_text and re.search(pattern, reference_text, re.IGNORECASE):
                    score += 0.2
                else:
                    score += 0.1  # Partial credit for having factual claims
        
        # Check for uncertainty markers (good for factual accuracy)
        uncertainty_markers = ['might', 'could', 'possibly', 'perhaps', 'likely', 'approximately']
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in output_text.lower())
        
        # Moderate uncertainty is good for factual correctness
        if 1 <= uncertainty_count <= 3:
            score += 0.2
        
        # Check for contradictions within the text
        contradiction_patterns = [
            (r'\bnot\s+(\w+)', r'\b\1\b'),  # "not X" followed by "X"
            (r'\bno\s+(\w+)', r'\b\1\b'),   # "no X" followed by "X"
        ]
        
        has_contradictions = False
        for neg_pattern, pos_pattern in contradiction_patterns:
            neg_matches = re.findall(neg_pattern, output_text, re.IGNORECASE)
            for match in neg_matches:
                if re.search(pos_pattern.replace('\\1', match), output_text, re.IGNORECASE):
                    has_contradictions = True
                    break
        
        if not has_contradictions:
            score += 0.3
        
        # If reference text exists, check semantic similarity
        if reference_text and self.sentence_model:
            try:
                output_embedding = self.sentence_model.encode([output_text])
                reference_embedding = self.sentence_model.encode([reference_text])
                similarity = np.dot(output_embedding[0], reference_embedding[0]) / (
                    np.linalg.norm(output_embedding[0]) * np.linalg.norm(reference_embedding[0])
                )
                score += similarity * 0.3
            except Exception as e:
                logger.warning(f"Could not compute semantic similarity: {e}")
        
        return min(score, 1.0)
    
    async def _evaluate_task_accuracy(self, output_text: str, reference_text: Optional[str]) -> Tuple[float, float]:
        """
        Evaluate Exact Match (EM) and F1 scores
        """
        if not reference_text:
            return 0.0, 0.0
        
        # Normalize texts
        output_normalized = self._normalize_text(output_text)
        reference_normalized = self._normalize_text(reference_text)
        
        # Exact Match
        em_score = 1.0 if output_normalized == reference_normalized else 0.0
        
        # F1 Score (token-level)
        output_tokens = set(output_normalized.split())
        reference_tokens = set(reference_normalized.split())
        
        if not output_tokens and not reference_tokens:
            f1_score = 1.0
        elif not output_tokens or not reference_tokens:
            f1_score = 0.0
        else:
            intersection = output_tokens.intersection(reference_tokens)
            precision = len(intersection) / len(output_tokens)
            recall = len(intersection) / len(reference_tokens)
            
            if precision + recall == 0:
                f1_score = 0.0
            else:
                f1_score = 2 * (precision * recall) / (precision + recall)
        
        return em_score, f1_score
    
    async def _evaluate_faithfulness(self, input_text: str, output_text: str, context: Dict[str, Any]) -> float:
        """
        Evaluate faithfulness/groundedness to source context
        """
        # Check if output is grounded in the provided context
        context_text = str(context.get('source_text', ''))
        
        if not context_text:
            # No context to check against, evaluate internal consistency
            return await self._evaluate_internal_consistency(output_text)
        
        # Check overlap between output and context
        context_words = set(self._normalize_text(context_text).split())
        output_words = set(self._normalize_text(output_text).split())
        
        if not output_words:
            return 0.0
        
        overlap = len(context_words.intersection(output_words))
        overlap_ratio = overlap / len(output_words)
        
        # Check for hallucinated information (claims not in context)
        hallucination_score = await self._detect_hallucinations(output_text, context_text)
        
        # Combine metrics
        faithfulness_score = (overlap_ratio * 0.6) + ((1 - hallucination_score) * 0.4)
        
        return min(faithfulness_score, 1.0)
    
    async def _evaluate_answer_relevancy(self, input_text: str, output_text: str) -> float:
        """
        Evaluate how relevant the answer is to the input question
        """
        if not self.sentence_model:
            return self._simple_relevancy_check(input_text, output_text)
        
        try:
            # Compute semantic similarity
            input_embedding = self.sentence_model.encode([input_text])
            output_embedding = self.sentence_model.encode([output_text])
            
            similarity = np.dot(input_embedding[0], output_embedding[0]) / (
                np.linalg.norm(input_embedding[0]) * np.linalg.norm(output_embedding[0])
            )
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.warning(f"Could not compute semantic similarity: {e}")
            return self._simple_relevancy_check(input_text, output_text)
    
    async def _evaluate_context_relevance(self, input_text: str, context: Dict[str, Any]) -> float:
        """
        Evaluate relevance of provided context to the input
        """
        context_text = str(context.get('source_text', ''))
        
        if not context_text:
            return 1.0  # No context means perfect relevance by default
        
        # Simple keyword overlap
        input_words = set(self._normalize_text(input_text).split())
        context_words = set(self._normalize_text(context_text).split())
        
        if not input_words:
            return 0.0
        
        overlap = len(input_words.intersection(context_words))
        relevance_score = overlap / len(input_words)
        
        return min(relevance_score, 1.0)
    
    async def _evaluate_prompt_alignment(self, input_text: str, output_text: str) -> float:
        """
        Evaluate how well the output follows the input instructions
        """
        score = 0.0
        
        # Check for instruction keywords in input
        instruction_keywords = [
            'explain', 'describe', 'list', 'summarize', 'compare', 'analyze',
            'define', 'provide', 'give', 'show', 'tell', 'write'
        ]
        
        found_instructions = []
        for keyword in instruction_keywords:
            if keyword in input_text.lower():
                found_instructions.append(keyword)
        
        if not found_instructions:
            return 0.8  # No clear instructions, assume general response is acceptable
        
        # Check if output follows the instruction type
        for instruction in found_instructions:
            if instruction == 'list' and self._has_list_format(output_text):
                score += 0.3
            elif instruction == 'explain' and len(output_text.split()) > 20:
                score += 0.3
            elif instruction == 'summarize' and len(output_text.split()) < len(input_text.split()):
                score += 0.3
            elif instruction in ['compare', 'analyze'] and ('and' in output_text or 'but' in output_text):
                score += 0.3
            else:
                score += 0.1  # Partial credit for attempting to follow
        
        return min(score / len(found_instructions), 1.0)
    
    async def _evaluate_fluency(self, text: str) -> float:
        """
        Evaluate grammatical fluency and naturalness
        """
        score = 0.0
        
        # Check for basic grammar patterns
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check for proper capitalization
            if sentence[0].isupper():
                score += 0.1
            
            # Check for reasonable sentence length
            words = sentence.split()
            if 5 <= len(words) <= 30:
                score += 0.1
            
            # Check for proper word spacing
            if not re.search(r'\s{2,}', sentence):
                score += 0.1
        
        # Check for repetitive patterns
        words = text.lower().split()
        word_counts = Counter(words)
        repeated_words = sum(1 for count in word_counts.values() if count > 3)
        
        if repeated_words < len(set(words)) * 0.1:  # Less than 10% repetition
            score += 0.3
        
        # Normalize by sentence count
        if sentences:
            score = score / len([s for s in sentences if s.strip()])
        
        return min(score, 1.0)
    
    async def _evaluate_coherence(self, text: str) -> float:
        """
        Evaluate logical coherence and flow
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0  # Single sentence is coherent by default
        
        score = 0.0
        
        # Check for transition words
        transition_words = [
            'however', 'therefore', 'furthermore', 'moreover', 'additionally',
            'consequently', 'meanwhile', 'nevertheless', 'thus', 'hence'
        ]
        
        transition_count = sum(1 for word in transition_words if word in text.lower())
        score += min(transition_count / len(sentences), 0.3)
        
        # Check for pronoun references (indicates coherence)
        pronouns = ['it', 'they', 'this', 'that', 'these', 'those']
        pronoun_count = sum(1 for word in pronouns if word in text.lower())
        score += min(pronoun_count / len(sentences), 0.2)
        
        # Check for topical consistency using word overlap
        if self.sentence_model:
            try:
                sentence_embeddings = self.sentence_model.encode(sentences)
                similarities = []
                
                for i in range(len(sentences) - 1):
                    sim = np.dot(sentence_embeddings[i], sentence_embeddings[i + 1]) / (
                        np.linalg.norm(sentence_embeddings[i]) * np.linalg.norm(sentence_embeddings[i + 1])
                    )
                    similarities.append(sim)
                
                avg_similarity = np.mean(similarities)
                score += avg_similarity * 0.5
                
            except Exception as e:
                logger.warning(f"Could not compute sentence similarities: {e}")
        
        return min(score, 1.0)
    
    async def _evaluate_readability(self, text: str) -> Dict[str, float]:
        """
        Evaluate various readability metrics
        """
        try:
            return {
                'flesch_reading_ease': textstat.flesch_reading_ease(text) / 100.0,
                'flesch_kincaid_grade': max(0, 1 - (textstat.flesch_kincaid_grade(text) / 20.0)),
                'automated_readability_index': max(0, 1 - (textstat.automated_readability_index(text) / 20.0)),
                'coleman_liau_index': max(0, 1 - (textstat.coleman_liau_index(text) / 20.0)),
                'reading_time': max(0, 1 - (textstat.reading_time(text) / 600.0))  # Normalize to 10 minutes
            }
        except Exception as e:
            logger.warning(f"Could not compute readability scores: {e}")
            return {
                'flesch_reading_ease': 0.5,
                'flesch_kincaid_grade': 0.5,
                'automated_readability_index': 0.5,
                'coleman_liau_index': 0.5,
                'reading_time': 0.5
            }
    
    async def _evaluate_bleu(self, output_text: str, reference_text: str) -> float:
        """
        Evaluate BLEU score
        """
        try:
            output_tokens = output_text.lower().split()
            reference_tokens = reference_text.lower().split()
            
            score = sentence_bleu(
                [reference_tokens],
                output_tokens,
                smoothing_function=self.smoothing_function
            )
            
            return score
            
        except Exception as e:
            logger.warning(f"Could not compute BLEU score: {e}")
            return 0.0
    
    async def _evaluate_rouge(self, output_text: str, reference_text: str) -> Dict[str, float]:
        """
        Evaluate ROUGE scores
        """
        try:
            scores = self.rouge_scorer.score(reference_text, output_text)
            
            return {
                'rouge1_f1': scores['rouge1'].fmeasure,
                'rouge2_f1': scores['rouge2'].fmeasure,
                'rougeL_f1': scores['rougeL'].fmeasure,
                'rouge1_precision': scores['rouge1'].precision,
                'rouge1_recall': scores['rouge1'].recall
            }
            
        except Exception as e:
            logger.warning(f"Could not compute ROUGE scores: {e}")
            return {
                'rouge1_f1': 0.0,
                'rouge2_f1': 0.0,
                'rougeL_f1': 0.0,
                'rouge1_precision': 0.0,
                'rouge1_recall': 0.0
            }
    
    async def _evaluate_meteor(self, output_text: str, reference_text: str) -> float:
        """
        Evaluate METEOR score
        """
        try:
            output_tokens = nltk.word_tokenize(output_text.lower())
            reference_tokens = nltk.word_tokenize(reference_text.lower())
            
            score = meteor_score([reference_tokens], output_tokens)
            return score
            
        except Exception as e:
            logger.warning(f"Could not compute METEOR score: {e}")
            return 0.0
    
    async def _evaluate_bert_score(self, output_text: str, reference_text: str) -> float:
        """
        Evaluate BERTScore
        """
        if not bert_score:
            return 0.0
        
        try:
            P, R, F1 = bert_score([output_text], [reference_text], lang='en', verbose=False)
            return F1.item()
            
        except Exception as e:
            logger.warning(f"Could not compute BERTScore: {e}")
            return 0.0
    
    async def _evaluate_mover_score(self, output_text: str, reference_text: str) -> float:
        """
        Evaluate MoverScore
        """
        if not word_mover_score:
            return 0.0
        
        try:
            references = [reference_text]
            translations = [output_text]
            idf_dict_ref = get_idf_dict(references)
            idf_dict_hyp = get_idf_dict(translations)
            scores = word_mover_score(references, translations, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
            return scores[0]
            
        except Exception as e:
            logger.warning(f"Could not compute MoverScore: {e}")
            return 0.0
    
    async def _evaluate_comet(self, input_text: str, output_text: str, reference_text: str) -> float:
        """
        Evaluate COMET score
        """
        if not load_from_checkpoint:
            return 0.0
        
        try:
            model_path = download_model("Unbabel/wmt22-comet-da")
            model = load_from_checkpoint(model_path)
            data = [{"src": input_text, "mt": output_text, "ref": reference_text}]
            model_output = model.predict(data, batch_size=1, gpus=0)
            return model_output.scores[0]
            
        except Exception as e:
            logger.warning(f"Could not compute COMET score: {e}")
            return 0.0
    
    async def _evaluate_diversity(self, text: str) -> Dict[str, float]:
        """
        Evaluate diversity and creativity metrics
        """
        words = text.lower().split()
        
        if not words:
            return {'lexical_diversity': 0.0, 'semantic_diversity': 0.0, 'syntactic_diversity': 0.0}
        
        # Lexical diversity (Type-Token Ratio)
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words)
        
        # Semantic diversity (placeholder - would need more sophisticated analysis)
        sentences = re.split(r'[.!?]+', text)
        semantic_diversity = min(len(sentences) / 10.0, 1.0)  # Normalize to max 10 sentences
        
        # Syntactic diversity (sentence length variation)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if len(sentence_lengths) > 1:
            syntactic_diversity = 1.0 - (statistics.stdev(sentence_lengths) / max(sentence_lengths, 1))
        else:
            syntactic_diversity = 0.5
        
        return {
            'lexical_diversity': lexical_diversity,
            'semantic_diversity': semantic_diversity,
            'syntactic_diversity': max(0.0, min(1.0, syntactic_diversity))
        }
    
    async def _evaluate_mauve(self, output_text: str, reference_text: str) -> float:
        """
        Evaluate MAUVE score (placeholder)
        """
        # This would require the MAUVE library implementation
        # For now, return a placeholder based on text similarity
        
        if not self.sentence_model:
            return 0.5
        
        try:
            output_embedding = self.sentence_model.encode([output_text])
            reference_embedding = self.sentence_model.encode([reference_text])
            
            similarity = np.dot(output_embedding[0], reference_embedding[0]) / (
                np.linalg.norm(output_embedding[0]) * np.linalg.norm(reference_embedding[0])
            )
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as e:
            logger.warning(f"Could not compute MAUVE approximation: {e}")
            return 0.5
    
    async def _evaluate_perplexity(self, text: str) -> Dict[str, float]:
        """
        Evaluate perplexity-related metrics
        """
        words = text.split()
        
        if not words:
            return {'perplexity': 0.0, 'cross_entropy': 0.0, 'bits_per_byte': 0.0}
        
        # Simple approximation based on word frequency
        word_counts = Counter(words)
        total_words = len(words)
        
        # Estimate cross-entropy
        cross_entropy = 0.0
        for word, count in word_counts.items():
            prob = count / total_words
            cross_entropy -= prob * math.log2(prob)
        
        # Estimate perplexity
        perplexity = 2 ** cross_entropy
        
        # Normalize perplexity (lower is better, so invert)
        normalized_perplexity = 1.0 / (1.0 + math.log(perplexity))
        
        # Bits per byte approximation
        text_bytes = len(text.encode('utf-8'))
        bits_per_byte = (cross_entropy * len(words)) / text_bytes if text_bytes > 0 else 0.0
        normalized_bpb = 1.0 / (1.0 + bits_per_byte)
        
        return {
            'perplexity': normalized_perplexity,
            'cross_entropy': 1.0 / (1.0 + cross_entropy),
            'bits_per_byte': normalized_bpb
        }
    
    # Helper methods
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    def _simple_relevancy_check(self, input_text: str, output_text: str) -> float:
        """Simple keyword-based relevancy check"""
        input_words = set(self._normalize_text(input_text).split())
        output_words = set(self._normalize_text(output_text).split())
        
        if not input_words or not output_words:
            return 0.0
        
        overlap = len(input_words.intersection(output_words))
        return overlap / len(input_words)
    
    def _has_list_format(self, text: str) -> bool:
        """Check if text has list formatting"""
        list_patterns = [
            r'^\s*\d+\.',  # Numbered lists
            r'^\s*[-*•]',  # Bullet points
            r'^\s*[a-z]\)',  # Lettered lists
        ]
        
        lines = text.split('\n')
        list_lines = 0
        
        for line in lines:
            for pattern in list_patterns:
                if re.search(pattern, line, re.MULTILINE):
                    list_lines += 1
                    break
        
        return list_lines >= 2
    
    async def _evaluate_internal_consistency(self, text: str) -> float:
        """Evaluate internal consistency of the text"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) < 2:
            return 1.0
        
        # Check for contradictory statements
        contradictions = 0
        for i, sentence in enumerate(sentences):
            for j, other_sentence in enumerate(sentences[i+1:], i+1):
                if self._are_contradictory(sentence, other_sentence):
                    contradictions += 1
        
        # Normalize contradiction score
        max_comparisons = len(sentences) * (len(sentences) - 1) / 2
        consistency_score = 1.0 - (contradictions / max_comparisons) if max_comparisons > 0 else 1.0
        
        return consistency_score
    
    def _are_contradictory(self, sentence1: str, sentence2: str) -> bool:
        """Simple contradiction detection"""
        # This is a basic implementation
        s1_lower = sentence1.lower()
        s2_lower = sentence2.lower()
        
        # Check for opposite words
        opposites = [
            ('yes', 'no'), ('true', 'false'), ('good', 'bad'),
            ('hot', 'cold'), ('big', 'small'), ('fast', 'slow')
        ]
        
        for word1, word2 in opposites:
            if (word1 in s1_lower and word2 in s2_lower) or (word2 in s1_lower and word1 in s2_lower):
                return True
        
        return False
    
    async def _detect_hallucinations(self, output_text: str, context_text: str) -> float:
        """Detect potential hallucinations"""
        if not context_text:
            return 0.0
        
        # Extract factual claims from output
        factual_patterns = [
            r'\b\d{4}\b',  # Years
            r'\b\d+\s*%\b',  # Percentages
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b',  # Proper names
        ]
        
        hallucinations = 0
        total_claims = 0
        
        for pattern in factual_patterns:
            output_matches = re.findall(pattern, output_text)
            total_claims += len(output_matches)
            
            for match in output_matches:
                if match not in context_text:
                    hallucinations += 1
        
        if total_claims == 0:
            return 0.0
        
        return hallucinations / total_claims
    
    # Essential evaluation methods that were missing
    async def _evaluate_factual_correctness(self, input_text: str, output_text: str, reference_text: str = None) -> float:
        """Evaluate factual correctness of the output"""
        if not reference_text:
            # Without reference, use simple heuristics
            factual_indicators = [
                'according to', 'research shows', 'studies indicate',
                'evidence suggests', 'data shows', 'statistics show'
            ]
            
            score = 0.0
            output_lower = output_text.lower()
            
            # Check for factual language
            for indicator in factual_indicators:
                if indicator in output_lower:
                    score += 0.2
            
            # Check for specific claims vs vague statements
            specific_patterns = [
                r'\b\d+\s*%\b',  # Percentages
                r'\b\d{4}\b',    # Years
                r'\b\d+\s+(million|billion|thousand)\b',  # Large numbers
                r'\b\d+\.\d+\b'  # Decimal numbers
            ]
            
            for pattern in specific_patterns:
                if re.search(pattern, output_text):
                    score += 0.15
            
            # Penalize uncertain language when making factual claims
            uncertain_patterns = ['maybe', 'might be', 'could be', 'possibly']
            for pattern in uncertain_patterns:
                if pattern in output_lower:
                    score -= 0.1
            
            return max(0.0, min(score, 1.0))
        
        else:
            # With reference, check alignment
            # Simple keyword overlap approach
            ref_words = set(reference_text.lower().split())
            out_words = set(output_text.lower().split())
            
            if len(ref_words) == 0:
                return 0.5
            
            overlap = len(ref_words.intersection(out_words))
            return min(overlap / len(ref_words), 1.0)
    
    async def _evaluate_faithfulness(self, input_text: str, output_text: str, context: Dict[str, Any]) -> float:
        """Evaluate faithfulness to input context"""
        if not context:
            return 0.8  # Default if no context
        
        context_text = str(context)
        
        # Check if output contradicts context
        contradictions = await self._detect_contradictions(output_text, context_text)
        faithfulness_score = 1.0 - contradictions
        
        return max(0.0, faithfulness_score)
    
    async def _detect_contradictions(self, output_text: str, context_text: str) -> float:
        """Detect contradictions between output and context"""
        # Simple approach: look for negation patterns
        negation_patterns = [
            r'not\s+', r'no\s+', r'never\s+', r'neither\s+',
            r'however,', r'but\s+', r'although\s+'
        ]
        
        contradiction_score = 0.0
        for pattern in negation_patterns:
            if re.search(pattern, output_text, re.IGNORECASE):
                contradiction_score += 0.1
        
        return min(contradiction_score, 1.0)
    
    async def _evaluate_answer_relevancy(self, input_text: str, output_text: str) -> float:
        """Evaluate how relevant the answer is to the question"""
        # Extract keywords from input
        input_words = set(re.findall(r'\b\w+\b', input_text.lower()))
        output_words = set(re.findall(r'\b\w+\b', output_text.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        input_words -= stop_words
        output_words -= stop_words
        
        if len(input_words) == 0:
            return 0.5
        
        # Calculate overlap
        overlap = len(input_words.intersection(output_words))
        relevancy_score = overlap / len(input_words)
        
        return min(relevancy_score, 1.0)
    
    async def _evaluate_context_relevance(self, input_text: str, context: Dict[str, Any]) -> float:
        """Evaluate relevance of context to input"""
        if not context:
            return 0.5
        
        context_text = str(context)
        return await self._evaluate_answer_relevancy(input_text, context_text)
    
    async def _evaluate_prompt_alignment(self, input_text: str, output_text: str) -> float:
        """Evaluate how well output follows input instructions"""
        # Check for instruction-following patterns
        instruction_patterns = [
            r'explain', r'describe', r'list', r'compare', r'analyze',
            r'summarize', r'define', r'calculate', r'solve', r'find'
        ]
        
        alignment_score = 0.5  # Base score
        
        for pattern in instruction_patterns:
            if re.search(pattern, input_text, re.IGNORECASE):
                # Check if output attempts to follow this instruction
                if pattern == 'list' and self._has_list_structure(output_text):
                    alignment_score += 0.2
                elif pattern == 'explain' and len(output_text.split()) > 20:
                    alignment_score += 0.2
                elif pattern in ['compare', 'analyze'] and ('however' in output_text.lower() or 'while' in output_text.lower()):
                    alignment_score += 0.2
                elif pattern == 'summarize' and len(output_text.split()) < len(input_text.split()):
                    alignment_score += 0.2
                else:
                    alignment_score += 0.1
        
        return min(alignment_score, 1.0)
