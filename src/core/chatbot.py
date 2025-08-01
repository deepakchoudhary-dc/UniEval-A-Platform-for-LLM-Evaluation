"""
Main AI Chatbot with Memory, Search, and Explainability
"""
import time
import uuid
import json
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
from src.fairness.bias_detector import bias_detector


class TransparentChatbot:
    """
    AI Chatbot with transparent decision-making, memory, and explainability
    """    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.default_model
        self.session_id = str(uuid.uuid4())
        
        # Check model provider configuration
        self.model_provider = getattr(settings, 'model_provider', 'openai').lower()
        
        # Initialize based on model provider
        if self.model_provider == 'ollama':
            # Ollama setup
            self.use_ollama = True
            self.use_openai = False
            self.client = None
            self.model_name = getattr(settings, 'ollama_model', 'qwen2.5:7b')
            print(f"✅ Ollama configured with model: {self.model_name}")
            
            # Test Ollama connection
            try:
                import requests
                test_response = requests.get("http://localhost:11434/api/version", timeout=5)
                if test_response.status_code == 200:
                    print(f"✅ Ollama server is running")
                else:
                    print(f"⚠️ Ollama server responded with status: {test_response.status_code}")
            except Exception as e:
                print(f"⚠️ Cannot connect to Ollama server: {e}")
        else:
            # OpenAI/OpenRouter setup with fallback
            self.use_ollama = False
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
            
            # Step 4: Bias and fairness check
            bias_results = {}
            fairness_score = 1.0
            if enable_bias_check and settings.fairness_check_enabled:
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
        """Generate AI response using OpenAI/OpenRouter with context or fallback"""
        
        # If OpenAI is not available and not using Ollama, use fallback response
        if not self.use_ollama and (not self.use_openai or not self.client):
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
            if self.use_ollama:
                # Use Ollama API
                import requests
                
                url = "http://localhost:11434/api/chat"
                data = {
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": settings.temperature,
                        "num_predict": settings.max_tokens
                    }
                }
                
                print(f"🔧 Ollama API Request to: {url}")
                print(f"🔧 Using model: {self.model_name}")
                
                response = requests.post(url, json=data, timeout=60)
                
                print(f"🔧 Ollama Response Status: {response.status_code}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    if "message" in response_data and "content" in response_data["message"]:
                        raw_response = response_data["message"]["content"]
                        
                        # Clean up the response - remove thinking tags
                        ai_response = self._clean_model_response(raw_response)
                        
                        confidence = 0.9  # High confidence for local Ollama
                        print(f"✅ Ollama response received: {ai_response[:100]}...")
                    else:
                        raise Exception(f"Ollama API unexpected response format: {response_data}")
                else:
                    raise Exception(f"Ollama API HTTP {response.status_code}: {response.text}")
                    
            elif self.client == "qwen_http":
                # Use HTTP requests for Qwen API - Correct DashScope format
                import requests
                
                # Qwen DashScope API endpoint
                url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
                headers = {
                    "Authorization": f"Bearer {self.qwen_api_key}",
                    "Content-Type": "application/json",
                    "X-DashScope-SSE": "disable"
                }
                
                # Convert to proper Qwen format - use messages directly
                qwen_messages = []
                for msg in messages:
                    qwen_messages.append({
                        "role": msg["role"] if msg["role"] != "assistant" else "assistant",
                        "content": msg["content"]
                    })
                
                data = {
                    "model": "qwen-turbo",
                    "input": {
                        "messages": qwen_messages
                    },
                    "parameters": {
                        "max_tokens": settings.max_tokens,
                        "temperature": settings.temperature,
                        "top_p": 0.8
                    }
                }
                
                print(f"🔧 Qwen API Request: {url}")
                print(f"🔧 Headers: {headers}")
                print(f"🔧 Data: {data}")
                
                response = requests.post(url, headers=headers, json=data, timeout=30)
                
                print(f"🔧 Qwen Response Status: {response.status_code}")
                print(f"🔧 Qwen Response: {response.text[:500]}...")
                
                if response.status_code == 200:
                    response_data = response.json()
                    if "output" in response_data and "text" in response_data["output"]:
                        ai_response = response_data["output"]["text"]
                        confidence = 0.8  # Default confidence for Qwen
                    else:
                        raise Exception(f"Qwen API unexpected response format: {response_data}")
                else:
                    raise Exception(f"Qwen API HTTP {response.status_code}: {response.text}")
                    
            else:
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
                "model_provider": getattr(self, 'model_provider', 'unknown'),
                "context_items_used": len(context_data.get("sources", [])),
                "recent_context_count": len(context_data.get("recent_context", [])),
                "similar_memories_count": len(context_data.get("similar_memories", [])),
                "response_length": len(ai_response),
                "temperature": settings.temperature
            }
            
            return {
                "answer": ai_response,
                "confidence": confidence,
                "factors": decision_factors,                "raw_response": response
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
        When referencing previous conversations, be clear about what information you're using.
        
        You can use <think>...</think> tags to show your reasoning process before providing your answer.
        This helps users understand how you arrived at your response."""
        
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
        
        # LIME explanation - Text-based explanation for the response
        if self.enable_lime:
            try:
                # Use the enhanced text prediction explainer
                lime_explanation = lime_explainer.explain_text_prediction(
                    text=response_data.get("answer", ""),
                    predict_fn=None,  # Will use default predictor
                    num_features=8
                )
                
                explanations["methods_used"].append("lime")
                explanations["detailed_explanations"]["lime"] = lime_explanation
                
                # Generate human-readable explanation
                explanations["lime_explanation"] = self._create_lime_summary(lime_explanation)
                
            except Exception as e:
                explanations["lime_error"] = str(e)
                print(f"LIME Error: {e}")
        
        # SHAP explanation - Text classification for response quality
        if self.enable_shap:
            try:
                # Use the enhanced text classification explainer
                shap_explanation = shap_explainer.explain_text_classification(
                    text=response_data.get("answer", ""),
                    classifier_fn=None,  # Will use default classifier
                    max_evals=100  # Reduced for performance
                )
                
                explanations["methods_used"].append("shap")
                explanations["detailed_explanations"]["shap"] = shap_explanation
                
                # Generate summary
                explanations["shap_summary"] = self._create_shap_summary(shap_explanation)
                
            except Exception as e:
                explanations["shap_error"] = str(e)
                print(f"SHAP Error: {e}")
        
        # Create unified summary
        explanations["summary"] = self._create_unified_explanation_summary(explanations)
        
        return explanations
    
    def _create_lime_summary(self, lime_explanation: Dict[str, Any]) -> str:
        """Create human-readable summary from LIME explanation"""
        if lime_explanation.get("error"):
            return f"LIME analysis encountered an error: {lime_explanation['error']}"
        
        features = lime_explanation.get("features", [])
        if not features:
            return "LIME analysis could not identify significant features."
        
        # Get top positive and negative features
        positive_features = [f for f in features if f["weight"] > 0][:3]
        negative_features = [f for f in features if f["weight"] < 0][:3]
        
        summary_parts = []
        
        if positive_features:
            pos_words = [f["feature"] for f in positive_features]
            summary_parts.append(f"Words that contributed positively: {', '.join(pos_words)}")
        
        if negative_features:
            neg_words = [f["feature"] for f in negative_features]
            summary_parts.append(f"Words that contributed negatively: {', '.join(neg_words)}")
        
        prediction_score = lime_explanation.get("prediction_score")
        if prediction_score is not None:
            summary_parts.append(f"Overall response quality score: {prediction_score:.3f}")
        
        return ". ".join(summary_parts)
    
    def _create_shap_summary(self, shap_explanation: Dict[str, Any]) -> str:
        """Create human-readable summary from SHAP explanation"""
        if shap_explanation.get("error"):
            return f"SHAP analysis encountered an error: {shap_explanation['error']}"
        
        explanations = shap_explanation.get("explanations", [])
        if not explanations:
            return "SHAP analysis could not identify significant features."
        
        # Get top contributors
        top_contributors = explanations[:5]
        
        summary_parts = []
        prediction = shap_explanation.get("prediction")
        base_value = shap_explanation.get("base_value", 0.5)
        
        if prediction is not None:
            summary_parts.append(f"SHAP prediction score: {prediction:.3f} (baseline: {base_value:.3f})")
        
        if top_contributors:
            contrib_desc = []
            for contrib in top_contributors:
                impact = "+" if contrib["contribution"] == "positive" else "-"
                contrib_desc.append(f"{contrib['token']} ({impact}{contrib['magnitude']:.3f})")
            summary_parts.append(f"Top word contributions: {', '.join(contrib_desc)}")
        
        return ". ".join(summary_parts)
    
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
                    "strength": abs(feature["weight"]),
                    "method": "lime"
                })
        
        # Extract key factors from SHAP
        if "shap" in explanations.get("detailed_explanations", {}):
            shap_data = explanations["detailed_explanations"]["shap"]
            shap_explanations = shap_data.get("explanations", [])[:3]  # Top 3
            for exp in shap_explanations:
                summary["key_factors"].append({
                    "factor": exp["token"],
                    "impact": exp["contribution"],
                    "strength": exp["magnitude"],
                    "method": "shap"
                })
        
        # Sort by strength
        summary["key_factors"].sort(key=lambda x: x["strength"], reverse=True)
        
        # Create reasoning text
        if summary["key_factors"]:
            reasoning_parts = []
            for factor in summary["key_factors"][:3]:
                impact_text = "positively influenced" if factor["impact"] == "positive" else "negatively influenced"
                reasoning_parts.append(f"'{factor['factor']}' {impact_text} the response")
            
            summary["reasoning"] = "Key findings: " + "; ".join(reasoning_parts)
        else:
            summary["reasoning"] = "No significant factors identified in the explanation analysis."
        
        # Add data sources
        if "methods_used" in explanations:
            summary["data_sources_used"] = explanations["methods_used"]
        
        return summary
    
    def _check_bias_and_fairness(
        self,
        user_query: str,
        ai_response: str,
        context_data: Dict[str, Any]
    ) -> tuple:
        """Check for bias and calculate fairness score"""
        
        try:
            print(f"🔧 DEBUG: Running bias detection on query: '{user_query[:50]}...'")
            bias_results = bias_detector.detect_bias(
                user_query, ai_response, context_data
            )
            
            print(f"🔧 DEBUG: Bias detection results: {bias_results.get('bias_detected', False)}")
            if bias_results.get('bias_detected', False):
                print(f"🔧 DEBUG: Bias types detected: {bias_results.get('bias_types', [])}")
            
            fairness_score = bias_detector.calculate_fairness_score(bias_results)
            
            return bias_results, fairness_score
            
        except Exception as e:
            print(f"🔧 DEBUG: Bias detection error: {str(e)}")
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
    
    async def evaluate_memory(self, query: str, response: str) -> Dict[str, Any]:
        """
        Evaluate memory performance and retrieval accuracy
        """
        # Get current memory stats
        memory_stats = memory_manager.get_memory_stats()
        
        # Calculate memory utilization
        total_memories = memory_stats.get('total_memories', 0)
        memory_utilization = min(total_memories / 1000, 1.0)  # Normalize to 1000 max
        
        # Get retrieval context for this query
        context_data = self._get_conversation_context(query)
        retrieved_items = len(context_data.get('sources', []))
        similar_memories = len(context_data.get('similar_memories', []))
        
        # Calculate retrieval accuracy (simplified)
        retrieval_accuracy = 0.8 if retrieved_items > 0 else 0.3
        
        # Calculate relevance score based on context usage
        relevance_score = min((retrieved_items + similar_memories) / 10, 1.0)
        
        recommendations = []
        if memory_utilization < 0.2:
            recommendations.append("Consider adding more training conversations to improve memory")
        if retrieval_accuracy < 0.5:
            recommendations.append("Review search algorithms to improve retrieval accuracy")
        if relevance_score < 0.3:
            recommendations.append("Enhance context relevance matching")
        
        return {
            'memory_utilization': memory_utilization,
            'retrieval_accuracy': retrieval_accuracy,
            'relevance_score': relevance_score,
            'memory_stats': memory_stats,
            'recommendations': recommendations
        }
    
    def _clean_model_response(self, raw_response: str) -> str:
        """Extract thinking and answer parts from model response"""
        import re
        
        print(f"🔧 DEBUG: Raw response input: {raw_response[:200]}...")
        
        # Extract thinking part
        thinking_match = re.search(r'<think>(.*?)</think>', raw_response, flags=re.DOTALL)
        thinking_part = ""
        if thinking_match:
            thinking_part = thinking_match.group(1).strip()
            print(f"🔧 DEBUG: Found thinking part: {len(thinking_part)} characters")
        
        # Remove <think>...</think> blocks to get the answer
        answer_part = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL)
        answer_part = re.sub(r'\n\s*\n', '\n\n', answer_part).strip()
        print(f"🔧 DEBUG: Answer part: {len(answer_part)} characters")
        
        # Format the response with thinking and answer separated
        if thinking_part and answer_part:
            formatted_response = f"**Thinking Process:**\n{thinking_part}\n\n**Answer:**\n{answer_part}"
        elif thinking_part:
            # Only thinking, no separate answer
            formatted_response = f"**Thinking Process:**\n{thinking_part}"
        elif answer_part:
            # Only answer, no thinking
            formatted_response = answer_part
        else:
            # Neither part found, return original or default
            formatted_response = raw_response.strip() if raw_response.strip() else "I apologize, but I need to provide a clearer response to your question."
        
        print(f"🔧 DEBUG: Final formatted response: {len(formatted_response)} characters")
        return formatted_response
    
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
