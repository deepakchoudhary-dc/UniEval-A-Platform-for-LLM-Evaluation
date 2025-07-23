"""
Main AI Chatbot with Memory, Search, and Explainability
"""
import time
import uuid
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import openai
from openai import OpenAI

from config.settings import settings
from src.core.memory import memory_manager
from src.core.search import search_engine
from src.data.database import db_manager, ConversationResponse, ExplanationResult
from src.explainability.lime_explainer import lime_explainer
from src.explainability.shap_explainer import shap_explainer
from src.fairness.advanced_bias_detector import advanced_bias_detector
from src.evaluation.opik_evaluator import OpikEvaluator, evaluate_chatbot_response
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TransparentChatbot:
    """
    AI Chatbot with transparent decision-making, memory, and explainability
    """    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.default_model
        self.session_id = str(uuid.uuid4())

        self.model_provider = getattr(settings, "model_provider", "openai").lower()
        self.ollama_model = getattr(settings, "ollama_model", "qwen3:1.7b")

        # Provider logic
        if self.model_provider == "ollama":
            self.client = "ollama"
            print(f"✅ Ollama client selected. Model: {self.ollama_model}")
            self.use_openai = False
        else:
            # Initialize OpenAI client with fallback
            api_key = settings.openai_api_key
            self.use_openai = (
                api_key and 
                api_key.strip() and 
                api_key.strip() != "your_openai_api_key_here" and
                api_key.startswith(("sk-", "sk-proj-", "sk-or-"))
            )
            if self.use_openai:
                if api_key.startswith("sk-or-"):
                    # OpenRouter setup (OpenAI-compatible)
                    self.client = OpenAI(
                        api_key=api_key,
                        base_url="https://openrouter.ai/api/v1"
                    )
                    self.model_name = "qwen/qwen-2.5-72b-instruct"  # Default Qwen model on OpenRouter
                    print(f"✅ OpenRouter client initialized with key: {api_key[:15]}...")
                    print(f"   Using model: {self.model_name}")
                else:
                    # OpenAI setup  
                    self.client = OpenAI(api_key=api_key)
                    print(f"✅ OpenAI client initialized with key: {api_key[:15]}...")
            else:
                self.client = None
                print(f"⚠️ No valid API key found. Using fallback mode.")

        # Conversation context
        self.conversation_history = []
        self.current_context = {}

        # Explainability settings
        self.enable_lime = settings.enable_lime
        self.enable_shap = settings.enable_shap
        self.explanation_detail = settings.explanation_detail_level

        # Initialize Opik evaluator for LLM evaluation
        self.enable_evaluation = getattr(settings, "enable_opik_evaluation", True)
        self.opik_evaluator = OpikEvaluator(project_name="transparent-ai-chatbot") if self.enable_evaluation else None
        if self.opik_evaluator and self.opik_evaluator.is_available():
            logger.info("Opik evaluation enabled for LLM monitoring")
        else:
            logger.info("Opik evaluation not available or disabled")

        # Initialize session in database
        db_manager.create_or_update_session(self.session_id)
    
    def chat(
        self,
        user_query: str,
        enable_memory_search: bool = True,
        enable_explanation: bool = True,
        enable_bias_check: bool = True
    ) -> ConversationResponse:
        """
        Main chat function with full transparency and explainability
        
        Args:
            user_query: User's input message
            enable_memory_search: Whether to search memory for context
            enable_explanation: Whether to generate explanations
            enable_bias_check: Whether to check for bias
        """
        
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant context from memory
            context_data = {}
            if enable_memory_search:
                context_data = self._get_conversation_context(user_query)
            
            # Step 2: Generate AI response
            response_data = self._generate_response(user_query, context_data)
            
            # Step 3: Generate explanations
            explanation_data = {}
            if enable_explanation and (self.enable_lime or self.enable_shap):
                explanation_data = self._generate_explanations(
                    user_query, context_data, response_data
                )
            
            # Step 4: Advanced Bias Detection and Educational Response
            bias_results = {}
            fairness_score = 1.0
            
            if enable_bias_check and settings.fairness_check_enabled:
                # First, analyze user input for bias
                input_bias_analysis = advanced_bias_detector.analyze_input_bias(user_query, context_data)
                
                # If significant bias detected in input, provide educational response
                if (input_bias_analysis["is_biased"] and 
                    input_bias_analysis["severity_level"] in ["high", "medium"]):
                    
                    logger.warning(f"Significant bias detected in user input: {input_bias_analysis['bias_types']}")
                    
                    # Generate educational response
                    bias_education_response = self._generate_bias_education_response(input_bias_analysis)
                    
                    if bias_education_response:
                        # Replace the AI response with educational content
                        response_data["answer"] = bias_education_response
                        response_data["confidence"] = 0.95
                        response_data["educational_intervention"] = True
                        
                        # Set bias results to the input analysis
                        bias_results = {
                            "input_bias": input_bias_analysis,
                            "response_bias": {"is_biased": False, "educational_response": True},
                            "bias_detected": True,
                            "bias_types": input_bias_analysis["bias_types"],
                            "educational_intervention": True,
                            "severity_level": input_bias_analysis["severity_level"]
                        }
                        fairness_score = 0.9  # High fairness for educational intervention
                    else:
                        # Fallback to normal bias checking
                        bias_results, fairness_score = self._check_bias_and_fairness(
                            user_query, response_data["answer"], context_data
                        )
                else:
                    # Normal bias checking on both input and response
                    bias_results, fairness_score = self._check_bias_and_fairness(
                        user_query, response_data["answer"], context_data
                    )
              # Step 5: Store conversation in memory
            response_time_ms = int((time.time() - start_time) * 1000)
            
            conversation_entry = memory_manager.store_conversation(
                user_query=user_query,
                ai_response=response_data["answer"],
                model_used=self.model_name,
                confidence_score=response_data["confidence"],
                response_time_ms=response_time_ms,
                session_id=self.session_id,
                explanation_data=explanation_data,
                data_sources=context_data.get("sources", []),
                decision_factors=response_data.get("factors", {}),
                bias_check_results=bias_results,
                fairness_score=fairness_score
            )
            
            # Step 6: Index for search
            search_engine.index_conversation(conversation_entry)
              # Step 7: Create response object
            response = ConversationResponse(
                answer=response_data["answer"],
                explanation=explanation_data if explanation_data else None,
                confidence=response_data["confidence"],
                sources=context_data.get("source_descriptions", []),
                timestamp=datetime.utcnow(),
                session_id=self.session_id,
                bias_check_results=bias_results if bias_results else None
            )
            
            # Step 8: Evaluate response with Opik (if enabled)
            if self.enable_evaluation and self.opik_evaluator:
                try:
                    # Run evaluation asynchronously in background
                    evaluation_task = self._evaluate_response_async(
                        user_query=user_query,
                        response=response_data["answer"],
                        context=context_data.get("memory_context", ""),
                        conversation_id=conversation_entry.id
                    )
                    # Start evaluation in background (don't wait for completion)
                    asyncio.create_task(evaluation_task)
                except Exception as eval_error:
                    logger.warning(f"Failed to start response evaluation: {eval_error}")
            
            # Update conversation history
            self.conversation_history.append({
                "user_query": user_query,
                "ai_response": response_data["answer"],
                "timestamp": datetime.utcnow(),
                "conversation_id": conversation_entry.id
            })
            
            return response
            
        except Exception as e:
            # Error handling with transparency
            error_response = ConversationResponse(
                answer=f"I apologize, but I encountered an error while processing your request: {str(e)}",
                explanation={"error": str(e), "method": "error_handling"},
                confidence=0.0,
                sources=["error_handler"],
                timestamp=datetime.utcnow(),
                session_id=self.session_id
            )
            
            return error_response
    
    def _get_conversation_context(self, user_query: str) -> Dict[str, Any]:
        """Retrieve and organize relevant context for the conversation"""
        
        context_data = {
            "sources": [],
            "source_descriptions": [],
            "recent_context": [],
            "similar_memories": [],
            "user_query": user_query,
            "session_id": self.session_id,
            "timestamp": datetime.utcnow()
        }
          # Get relevant context from memory
        relevant_context = memory_manager.get_relevant_context(user_query, max_context_items=5)
        
        for item in relevant_context:
            if item["source"] == "recent_conversation":
                context_data["recent_context"].append({
                    "query": item["user_query"],
                    "response": item["ai_response"],
                    "timestamp": item["timestamp"],
                    "relevance": item["relevance_score"]
                })
                context_data["sources"].append(f"recent_conversation_{item['id']}")
                context_data["source_descriptions"].append(
                    f"Recent conversation from {item['timestamp'].strftime('%Y-%m-%d %H:%M')}"
                )
            
            elif item["source"] == "similar_memory":
                context_data["similar_memories"].append({
                    "snippet": item["snippet"],
                    "timestamp": item["timestamp"],
                    "relevance": item["relevance_score"],                    "context": item["context"]
                })
                context_data["sources"].append(f"memory_{item['id']}")
                context_data["source_descriptions"].append(
                    f"Similar conversation from {item['timestamp'].strftime('%Y-%m-%d %H:%M')} (relevance: {item['relevance_score']:.2f})"
                )
        
        return context_data
    
    def _generate_response(self, user_query: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AI response using OpenAI/OpenRouter/Ollama with context or fallback"""

        # Ollama support
        if self.model_provider == "ollama":
            return self._generate_ollama_response(user_query, context_data)

            decision_factors = {
                "model_used": self.ollama_model,
                "context_items_used": len(context_data.get("sources", [])),
                "recent_context_count": len(context_data.get("recent_context", [])),
                "similar_memories_count": len(context_data.get("similar_memories", [])),
                "response_length": len(ai_response),
                "temperature": settings.temperature
            }
            return {
                "answer": ai_response,
                "confidence": confidence,
                "factors": decision_factors,
                "raw_response": response.text
            }

        # If OpenAI is not available, use fallback response
        if not self.use_openai or not self.client:
            return self._generate_fallback_response(user_query, context_data)

        # Build system prompt with context
        system_prompt = self._build_system_prompt(context_data)

        # Build conversation messages
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Add recent conversation history
        for item in context_data.get("recent_context", [])[-3:]:  # Last 3 exchanges
            messages.append({"role": "user", "content": item["query"]})
            messages.append({"role": "assistant", "content": item["response"]})
        # Add current query
        messages.append({"role": "user", "content": user_query})

        try:
            # Use OpenAI-compatible client (for both OpenAI and Qwen)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                presence_penalty=0.1,
                frequency_penalty=0.1
            )

            ai_response = response.choices[0].message.content

            # Calculate confidence based on response characteristics
            confidence = self._calculate_confidence(
                user_query, ai_response, context_data, response
            )

            # Extract decision factors
            decision_factors = {
                "model_used": self.model_name,
                "context_items_used": len(context_data.get("sources", [])),
                "recent_context_count": len(context_data.get("recent_context", [])),
                "similar_memories_count": len(context_data.get("similar_memories", [])),
                "response_length": len(ai_response),
                "temperature": settings.temperature
            }

            return {
                "answer": ai_response,
                "confidence": confidence,
                "factors": decision_factors,
                "raw_response": response
            }

        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            if hasattr(e, 'response'):
                print(f"Response status: {e.response.status_code if hasattr(e.response, 'status_code') else 'Unknown'}")
            return self._generate_fallback_response(user_query, context_data)
    
    def _build_system_prompt(self, context_data: Dict[str, Any]) -> str:
        """Build system prompt with context information"""
        
        base_prompt = """You are a helpful AI assistant with access to conversation memory. 
        You should provide accurate, helpful responses while being transparent about your reasoning.
        When referencing previous conversations, be clear about what information you're using."""
        
        context_parts = []
        
        # Add recent context
        recent_context = context_data.get("recent_context", [])
        if recent_context:
            context_parts.append("Recent conversation context:")
            for item in recent_context[-2:]:  # Last 2 items
                context_parts.append(f"- User asked: {item['query'][:100]}...")
                context_parts.append(f"- You responded: {item['response'][:100]}...")
        
        # Add similar memories
        similar_memories = context_data.get("similar_memories", [])
        if similar_memories:
            context_parts.append("Relevant information from previous conversations:")
            for memory in similar_memories[:2]:  # Top 2 relevant memories
                context_parts.append(f"- {memory['snippet'][:150]}...")
        
        # Instructions for transparency
        context_parts.append("\nInstructions:")
        context_parts.append("- If you reference previous conversations, mention it clearly")
        context_parts.append("- Be honest about what you know and don't know")
        context_parts.append("- Provide reasoning for your answers when appropriate")
        
        if context_parts:
            return base_prompt + "\n\n" + "\n".join(context_parts)
        else:
            return base_prompt
    
    def _calculate_confidence(
        self,
        user_query: str,
        ai_response: str,
        context_data: Dict[str, Any],
        raw_response: Any
    ) -> float:
        """Calculate confidence score for the response"""
        
        confidence_factors = []
        
        # Response length factor
        response_length = len(ai_response.split())
        if 10 <= response_length <= 200:  # Optimal range
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        # Context availability factor
        context_score = 0.5  # Base score
        if context_data.get("recent_context"):
            context_score += 0.2
        if context_data.get("similar_memories"):
            context_score += 0.2
        confidence_factors.append(min(context_score, 1.0))
        
        # Query clarity factor (simple heuristic)
        query_words = len(user_query.split())
        if 3 <= query_words <= 20:  # Reasonable query length
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        # Model response characteristics
        if hasattr(raw_response, 'usage') and raw_response.usage:
            # Higher token usage might indicate more thorough response
            tokens_used = raw_response.usage.completion_tokens
            token_factor = min(tokens_used / 100, 1.0)  # Normalize to max 1.0
            confidence_factors.append(0.5 + token_factor * 0.3)
        else:
            confidence_factors.append(0.7)
        
        # Calculate weighted average
        return sum(confidence_factors) / len(confidence_factors)
    
    def _generate_explanations(
        self,
        user_query: str,
        context_data: Dict[str, Any],
        response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate explanations using LIME and SHAP"""
        
        explanations = {
            "generation_timestamp": datetime.utcnow().isoformat(),
            "methods_used": [],
            "summary": {},
            "detailed_explanations": {}
        }
        
        # LIME explanation
        if self.enable_lime:
            try:
                lime_explanation = lime_explainer.explain_response_generation(
                    user_query,
                    context_data,
                    lambda query, context: response_data,
                    num_features=8
                )
                
                explanations["methods_used"].append("lime")
                explanations["detailed_explanations"]["lime"] = lime_explanation
                
                # Generate human-readable explanation
                explanations["lime_explanation"] = lime_explainer.generate_human_readable_explanation(
                    lime_explanation, self.explanation_detail
                )
                
            except Exception as e:
                explanations["lime_error"] = str(e)
        
        # SHAP explanation
        if self.enable_shap:
            try:
                shap_explanation = shap_explainer.explain_response_factors(
                    user_query,
                    context_data,
                    lambda query, context: response_data
                )
                
                explanations["methods_used"].append("shap")
                explanations["detailed_explanations"]["shap"] = shap_explanation
                
                # Generate summary
                explanations["shap_summary"] = shap_explainer.generate_explanation_summary(
                    shap_explanation
                )
                
            except Exception as e:
                explanations["shap_error"] = str(e)
        
        # Create unified summary
        explanations["summary"] = self._create_unified_explanation_summary(explanations)
        
        return explanations
    
    def _create_unified_explanation_summary(self, explanations: Dict[str, Any]) -> Dict[str, Any]:
        """Create a unified summary from multiple explanation methods"""
        
        summary = {
            "key_factors": [],
            "data_sources_used": [],
            "confidence_factors": [],
            "reasoning": ""
        }
        
        # Extract key factors from LIME
        if "lime" in explanations.get("detailed_explanations", {}):
            lime_data = explanations["detailed_explanations"]["lime"]
            features = lime_data.get("features", [])[:3]  # Top 3
            for feature in features:
                summary["key_factors"].append({
                    "factor": feature["feature"],
                    "impact": "positive" if feature["weight"] > 0 else "negative",
                    "method": "lime"
                })
        
        # Extract key factors from SHAP
        if "shap" in explanations.get("detailed_explanations", {}):
            shap_data = explanations["detailed_explanations"]["shap"]
            shap_explanations = shap_data.get("explanations", [])[:3]  # Top 3
            for exp in shap_explanations:
                summary["key_factors"].append({
                    "factor": exp["feature"],
                    "impact": exp["contribution"],
                    "method": "shap"
                })
        
        # Data sources
        if "context_analysis" in explanations.get("detailed_explanations", {}).get("lime", {}):
            context_analysis = explanations["detailed_explanations"]["lime"]["context_analysis"]
            important_features = context_analysis.get("important_features", [])
            for feature in important_features[:2]:
                summary["data_sources_used"].append(feature["name"])
          # Generate reasoning text
        if summary["key_factors"]:
            reasoning_parts = [
                "This response was generated considering several key factors:"
            ]
            
            for i, factor in enumerate(summary["key_factors"][:3], 1):
                impact_text = "positively influenced" if factor["impact"] == "positive" else "negatively influenced"
                reasoning_parts.append(
                    f"{i}. '{factor['factor']}' {impact_text} the response (detected by {factor['method'].upper()})"
                )
            
            summary["reasoning"] = " ".join(reasoning_parts)
        else:
            summary["reasoning"] = "This response was generated using the AI model's training knowledge without specific contextual factors being identified."
        
        return summary
    
    def _check_bias_and_fairness(
        self,
        user_query: str,
        ai_response: str,
        context_data: Dict[str, Any]
    ) -> tuple:
        """Check for bias and calculate fairness score"""
        
        try:
            print(f"🔧 DEBUG: Running advanced bias detection on query: '{user_query[:50]}...'")
            
            # Analyze user input for bias first
            input_bias_analysis = advanced_bias_detector.analyze_input_bias(user_query, context_data)
            
            # Analyze AI response for bias
            response_bias_analysis = advanced_bias_detector.analyze_response_bias(
                ai_response, user_query, context_data
            )
            
            # Combine analyses
            bias_results = {
                "input_bias": input_bias_analysis,
                "response_bias": response_bias_analysis,
                "bias_detected": input_bias_analysis["is_biased"] or response_bias_analysis["is_biased"],
                "bias_types": list(set(input_bias_analysis["bias_types"] + response_bias_analysis["bias_types"])),
                "overall_bias_score": max(input_bias_analysis["overall_bias_score"], response_bias_analysis["overall_bias_score"]),
                "severity_level": max(input_bias_analysis["severity_level"], response_bias_analysis["severity_level"], key=lambda x: ["none", "low", "medium", "high"].index(x))
            }
            
            print(f"🔧 DEBUG: Advanced bias detection results - Input biased: {input_bias_analysis['is_biased']}, Response biased: {response_bias_analysis['is_biased']}")
            if bias_results.get('bias_detected', False):
                print(f"🔧 DEBUG: Bias types detected: {bias_results.get('bias_types', [])}")
            
            # Calculate fairness score (1.0 = fair, 0.0 = completely biased)
            fairness_score = 1.0 - bias_results["overall_bias_score"]
            
            return bias_results, fairness_score
            
        except Exception as e:
            print(f"🔧 DEBUG: Advanced bias detection error: {str(e)}")
            logger.error(f"Bias detection error: {e}")
            return {"bias_check_error": str(e)}, 0.5
    
    def search_memory(
        self,
        query: str,
        search_type: str = "comprehensive",
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search through conversation memory"""
        
        return search_engine.search(
            query=query,
            session_id=self.session_id,
            limit=limit,
            search_type=search_type
        )
    
    def explain_last_response(self) -> Optional[ExplanationResult]:
        """Get detailed explanation for the last response"""
        
        if not self.conversation_history:
            return None
        
        last_conversation = self.conversation_history[-1]
        conversation_id = last_conversation["conversation_id"]
        
        # Get conversation details from database
        conversation_source = memory_manager.get_conversation_source(conversation_id)
        
        if not conversation_source:
            return None
        
        explanation_data = conversation_source.get("explanation_data", {})
        
        if not explanation_data:
            return None
        
        # Create explanation result
        result = ExplanationResult(
            method=",".join(explanation_data.get("methods_used", ["unknown"])),
            confidence=conversation_source.get("confidence_score", 0.0),
            key_factors=explanation_data.get("summary", {}).get("key_factors", []),
            data_sources=conversation_source.get("data_sources", []),
            decision_reasoning=explanation_data.get("summary", {}).get("reasoning", "")
        )
        
        return result
    
    def ask_about_source(self, query: str) -> str:
        """Answer questions about data sources and reasoning"""
        
        if not self.conversation_history:
            return "I haven't had any conversations yet to reference."
        
        last_conversation = self.conversation_history[-1]
        conversation_id = last_conversation["conversation_id"]
        
        # Get conversation source details
        source_details = memory_manager.get_conversation_source(conversation_id)
        
        if not source_details:
            return "I couldn't retrieve the source information for the last response."
        
        # Process different types of source questions
        query_lower = query.lower()
        
        if "memory" in query_lower or "conversation" in query_lower:
            data_sources = source_details.get("data_sources", [])
            if data_sources:
                return f"For my last response, I referenced {len(data_sources)} previous conversation(s): {', '.join(data_sources)}"
            else:
                return "My last response was generated without referencing specific previous conversations."
        
        elif "confidence" in query_lower or "sure" in query_lower:
            confidence = source_details.get("confidence_score", 0.0)
            return f"I had a confidence level of {confidence:.2f} (out of 1.0) for my last response."
        
        elif "explain" in query_lower or "why" in query_lower:
            explanation_data = source_details.get("explanation_data", {})
            if explanation_data and "summary" in explanation_data:
                reasoning = explanation_data["summary"].get("reasoning", "")
                if reasoning:
                    return f"Here's my reasoning: {reasoning}"
            
            return "I don't have detailed reasoning information available for that response."
        
        elif "bias" in query_lower or "fair" in query_lower:
            bias_results = source_details.get("bias_check_results", {})
            fairness_score = source_details.get("fairness_score", 1.0)
            
            if bias_results:
                return f"I checked for bias and received a fairness score of {fairness_score:.2f}. No significant bias was detected."
            else:
                return "Bias checking was not performed for that response."
        
        else:
            # General source information
            model_used = source_details.get("model_used", "unknown")
            timestamp = source_details.get("timestamp")
            
            response_parts = [
                f"My last response was generated using {model_used}",
                f"at {timestamp.strftime('%Y-%m-%d %H:%M:%S') if timestamp else 'unknown time'}"
            ]
            
            data_sources = source_details.get("data_sources", [])
            if data_sources:
                response_parts.append(f"referencing {len(data_sources)} previous conversation(s)")
            
            return ", ".join(response_parts) + "."
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about the memory system"""
        return memory_manager.get_memory_stats()
    
    def clear_session_memory(self):
        """Clear memory for current session"""
        memory_manager.clear_session_memory(self.session_id)
        self.conversation_history.clear()
    
    def set_session_id(self, session_id: str):
        """Set a specific session ID"""
        self.session_id = session_id
        db_manager.create_or_update_session(session_id)
    
    def _generate_fallback_response(self, user_query: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback response when OpenAI is not available"""
        
        # Simple rule-based responses
        query_lower = user_query.lower()
        
        # Greeting responses
        if any(word in query_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            response = "Hello! I'm an AI chatbot running in fallback mode (no OpenAI API key configured). How can I help you today?"
        
        # Question about capabilities
        elif any(word in query_lower for word in ['what can you do', 'help', 'capabilities']):
            response = "I'm running in fallback mode. I can still:\n• Remember our conversations\n• Search through past conversations\n• Provide explanations of my decisions\n• Check for bias in responses\n• Show you how I make decisions\n\nTo get full AI responses, please configure an OpenAI API key."
        
        # How are you
        elif any(word in query_lower for word in ['how are you', 'how do you feel']):
            response = "I'm functioning well in fallback mode! While I can't access advanced AI models right now, all my transparency features are still working."
        
        # Memory-related
        elif any(word in query_lower for word in ['remember', 'memory', 'conversation']):
            memory_count = len(context_data.get('similar_memories', []))
            response = f"I can access my memory! I found {memory_count} similar conversations. My memory and search capabilities work even without OpenAI."
        
        # Default response
        else:
            memory_info = ""
            if context_data.get('similar_memories'):
                memory_info = f"\n\nI found {len(context_data['similar_memories'])} similar conversations in my memory that might be relevant."
            
            response = f"I'm running in fallback mode without OpenAI API access. I received your message: '{user_query}'{memory_info}\n\nWhile I can't provide full AI responses, I can still remember our conversation, search my memory, and explain my decision-making process. To get full AI capabilities, please configure an OpenAI API key."
        
        # Calculate simple confidence based on pattern matching
        confidence = 0.7 if any(word in query_lower for word in ['hello', 'hi', 'help', 'how are you']) else 0.5
        
        decision_factors = {
            "model_used": "fallback_mode",
            "context_items_used": len(context_data.get("sources", [])),
            "similar_memories_count": len(context_data.get("similar_memories", [])),
            "response_type": "rule_based",
            "pattern_matched": query_lower[:50] + "..." if len(query_lower) > 50 else query_lower
        }
        
        return {
            "answer": response,
            "confidence": confidence,
            "factors": decision_factors,            "raw_response": None
        }

    def _generate_ollama_response(self, user_query: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response using Ollama API with robust error handling"""
        import requests
        import re
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        # Build messages
        system_prompt = self._build_system_prompt(context_data)
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history
        for item in context_data.get("recent_context", [])[-3:]:
            messages.append({"role": "user", "content": item["query"]})
            messages.append({"role": "assistant", "content": item["response"]})
        
        messages.append({"role": "user", "content": user_query})
        
        # Configure session with retries
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        
        url = "http://localhost:11434/api/chat"
        data = {
            "model": self.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": settings.temperature,
                "num_predict": settings.max_tokens,
                "top_p": 0.9,
                "top_k": 40
            }
        }
        
        try:
            response = session.post(url, json=data, timeout=120)
            
            if response.status_code == 200:
                response_data = response.json()
                ai_response = response_data.get("message", {}).get("content", "")
                
                # Clean response (remove <think> tags and artifacts)
                ai_response = self._clean_ollama_response(ai_response)
                
                # Calculate confidence based on response quality
                confidence = self._calculate_ollama_confidence(ai_response, response_data)
                
                decision_factors = {
                    "model_used": self.ollama_model,
                    "context_items_used": len(context_data.get("sources", [])),
                    "recent_context_count": len(context_data.get("recent_context", [])),
                    "similar_memories_count": len(context_data.get("similar_memories", [])),
                    "response_length": len(ai_response),
                    "temperature": settings.temperature,
                    "provider": "ollama",
                    "cleaned": True
                }
                
                return {
                    "answer": ai_response,
                    "confidence": confidence,
                    "factors": decision_factors,
                    "raw_response": response_data
                }
            else:
                raise Exception(f"Ollama API HTTP {response.status_code}: {response.text}")
                
        except requests.exceptions.ConnectionError:
            print("❌ Ollama server not available. Please start Ollama with 'ollama serve'")
            return self._generate_fallback_response(user_query, context_data)
        except requests.exceptions.Timeout:
            print("⏱️ Ollama request timed out. Model may be too large or busy.")
            return self._generate_fallback_response(user_query, context_data)
        except Exception as e:
            print(f"❌ Ollama API error: {str(e)}")
            return self._generate_fallback_response(user_query, context_data)
        finally:
            session.close()

    def _clean_ollama_response(self, response: str) -> str:
        """Clean Ollama response from artifacts and thinking tags"""
        import re
        
        # Remove <think> blocks
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # Remove leading/trailing whitespace
        response = response.strip()
        
        # Remove excessive newlines
        response = re.sub(r'\n{3,}', '\n\n', response)
        
        # Ensure we have content
        if not response or len(response.strip()) < 10:
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        
        return response

    def _calculate_ollama_confidence(self, response: str, response_data: Dict) -> float:
        """Calculate confidence score for Ollama response"""
        confidence_factors = []
        
        # Response length factor
        response_length = len(response.split())
        if 20 <= response_length <= 300:
            confidence_factors.append(0.9)
        elif 10 <= response_length <= 500:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Response quality indicators
        if any(phrase in response.lower() for phrase in ['i apologize', 'i cannot', 'i don\'t know']):
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.8)
        
        # Check for completeness (ends with punctuation)
        if response.strip() and response.strip()[-1] in '.!?':
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.7)
        
        # Model-specific factors
        if 'qwen' in self.ollama_model.lower():
            confidence_factors.append(0.85)  # Qwen is generally reliable
        else:
            confidence_factors.append(0.8)
        
        return min(sum(confidence_factors) / len(confidence_factors), 1.0)

    async def _evaluate_response_async(
        self,
        user_query: str,
        response: str,
        context: str,
        conversation_id: str
    ) -> None:
        """
        Asynchronously evaluate the response using Opik.
        
        Args:
            user_query: User's input query
            response: Generated response
            context: Context used for generation
            conversation_id: Unique conversation identifier
        """
        try:
            if self.opik_evaluator:
                evaluation_result = await evaluate_chatbot_response(
                    evaluator=self.opik_evaluator,
                    user_input=user_query,
                    bot_response=response,
                    context=context
                )
                
                # Log evaluation results
                overall_score = evaluation_result.get("overall_score", 0.0)
                recommendations = evaluation_result.get("recommendations", [])
                
                logger.info(f"Response evaluation completed for conversation {conversation_id}")
                logger.info(f"Overall score: {overall_score:.3f}")
                
                if recommendations:
                    logger.info(f"Recommendations: {'; '.join(recommendations[:2])}")
                
                # Store evaluation results in database for future analysis
                try:
                    db_manager.store_evaluation_result(
                        conversation_id=conversation_id,
                        evaluation_data=evaluation_result
                    )
                except Exception as db_error:
                    logger.warning(f"Failed to store evaluation in database: {db_error}")
                    
        except Exception as e:
            logger.error(f"Error during response evaluation: {e}")

    def _generate_bias_education_response(self, bias_analysis: Dict[str, Any]) -> Optional[str]:
        """
        Generate an educational response when bias is detected in user input.
        
        Args:
            bias_analysis: Results from bias analysis
            
        Returns:
            Educational response string or None
        """
        try:
            bias_types = bias_analysis.get("bias_types", [])
            severity = bias_analysis.get("severity_level", "none")
            recommendations = bias_analysis.get("recommendations", [])
            
            if not bias_types or severity == "none":
                return None
            
            # Generate educational response based on bias types
            response_parts = [
                "I notice your question contains language that might reflect unconscious bias. "
                "I'd like to help you explore this topic in a more inclusive way."
            ]
            
            if "gender_bias" in bias_types:
                response_parts.append(
                    "Gender bias often involves assumptions about what people can or should do based on their gender. "
                    "It's more accurate and fair to consider individual capabilities regardless of gender."
                )
            
            if "racial_bias" in bias_types:
                response_parts.append(
                    "Racial bias involves generalizations about people based on their race or ethnicity. "
                    "Each person is unique, and it's important to avoid stereotypes that can be harmful."
                )
            
            if "age_bias" in bias_types:
                response_parts.append(
                    "Age bias involves assumptions about capabilities based on age. "
                    "People of all ages have diverse skills, experiences, and perspectives to offer."
                )
            
            if "religious_bias" in bias_types:
                response_parts.append(
                    "Religious bias involves generalizations about people based on their faith. "
                    "It's important to respect religious diversity and avoid stereotypes."
                )
            
            if "lgbtq_bias" in bias_types:
                response_parts.append(
                    "LGBTQ+ bias involves assumptions about sexual orientation or gender identity. "
                    "Everyone deserves respect regardless of their identity."
                )
            
            # Add recommendations
            if recommendations:
                response_parts.append("Here are some suggestions for more inclusive language:")
                response_parts.extend(f"• {rec}" for rec in recommendations[:3])
            
            response_parts.append(
                "Would you like to rephrase your question, or would you prefer information about this topic "
                "from a more balanced perspective?"
            )
            
            return "\n\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"Error generating bias education response: {e}")
            return None

    def get_evaluation_summary(self, limit: int = 10) -> Dict[str, Any]:
        """
        Get summary of recent evaluations.
        
        Args:
            limit: Number of recent evaluations to include
            
        Returns:
            Summary of evaluation metrics and trends
        """
        if not self.opik_evaluator:
            return {"message": "Evaluation not enabled"}
        
        return self.opik_evaluator.get_evaluation_summary(limit=limit)

    async def evaluate_conversation_quality(self) -> Dict[str, Any]:
        """
        Evaluate the quality of the current conversation.
        
        Returns:
            Conversation-level evaluation results
        """
        if not self.opik_evaluator:
            return {"message": "Evaluation not enabled"}
        
        # Convert conversation history to evaluation format
        conversation_data = []
        for turn in self.conversation_history:
            conversation_data.append({
                "user": turn["user_query"],
                "assistant": turn["ai_response"]
            })
        
        if conversation_data:
            return await self.opik_evaluator.evaluate_conversation_quality(
                conversation_history=conversation_data,
                criteria=["relevance", "coherence", "engagement"]
            )
        else:
            return {"message": "No conversation history available"}

# Example usage and testing functions
if __name__ == "__main__":
    # Example usage
    chatbot = TransparentChatbot()
    
    # Test conversation
    response = chatbot.chat("What is machine learning?")
    print(f"Response: {response.answer}")
    print(f"Confidence: {response.confidence}")
    print(f"Sources: {response.sources}")
    
    if response.explanation:
        print(f"Explanation: {response.explanation.get('summary', {}).get('reasoning', 'No explanation available')}")
    
    # Test memory search
    search_results = chatbot.search_memory("machine learning")
    print(f"Found {len(search_results)} related conversations")
    
    # Test source questioning
    source_info = chatbot.ask_about_source("Why did you give that answer?")
    print(f"Source info: {source_info}")
