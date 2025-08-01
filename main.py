"""
Main entry point for the Transparent AI Chatbot
"""
import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import settings, validate_settings
from src.core.chatbot import TransparentChatbot
from src.explainability.model_card import model_card_generator
from src.fairness.bias_detector import bias_detector


def main():
    """Main function to run the chatbot"""
    
    parser = argparse.ArgumentParser(description="Transparent AI Chatbot with Memory and Explainability")
    parser.add_argument("--mode", choices=["interactive", "api", "demo", "model-card"], 
                       default="interactive", help="Run mode")
    parser.add_argument("--port", type=int, default=8000, help="Port for API mode")
    parser.add_argument("--host", default="localhost", help="Host for API mode")
    parser.add_argument("--export-model-card", help="Export model card to file")
    parser.add_argument("--format", choices=["json", "markdown"], default="json", 
                       help="Format for model card export")
    
    args = parser.parse_args()
    
    try:
        # Validate settings
        validate_settings()
        print("✓ Settings validated successfully")
        
        if args.mode == "model-card":
            generate_model_card(args.export_model_card, args.format)
        elif args.mode == "interactive":
            run_interactive_mode()
        elif args.mode == "api":
            # Use environment PORT for deployment platforms like Render
            port = int(os.getenv("PORT", args.port))
            host = os.getenv("HOST", args.host)
            run_api_mode(host, port)
        elif args.mode == "demo":
            run_demo_mode()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


def generate_model_card(export_path: str = None, format: str = "json"):
    """Generate and optionally export model card"""
    
    print("🔄 Generating model card...")
    
    # Generate model card
    model_card = model_card_generator.generate_model_card()
    
    # Save to database
    if model_card_generator.save_model_card(model_card):
        print("✓ Model card saved to database")
    else:
        print("⚠️ Warning: Could not save model card to database")
    
    # Export to file if requested
    if export_path:
        if model_card_generator.export_model_card(export_path, format):
            print(f"✓ Model card exported to {export_path}")
        else:
            print(f"❌ Error: Could not export model card to {export_path}")
    else:
        # Print summary
        print("\n📋 Model Card Summary:")
        print(f"Name: {model_card['model_details']['name']}")
        print(f"Version: {model_card['model_details']['version']}")
        print(f"Type: {model_card['model_details']['type']}")
        print(f"Primary Uses: {len(model_card['intended_use']['primary_intended_uses'])} defined")
        print(f"Known Limitations: {len(model_card['limitations']['known_limitations'])} identified")
        print(f"Ethical Considerations: {len(model_card['ethical_considerations']['considerations'])} addressed")


def run_interactive_mode():
    """Run the chatbot in interactive command-line mode"""
    
    print("🤖 Transparent AI Chatbot")
    print("=" * 50)
    print("Features: Memory, Search, Explainability, Bias Detection")
    print("Type 'help' for commands, 'quit' to exit")
    print("=" * 50)
    
    # Initialize chatbot
    chatbot = TransparentChatbot()
    
    print(f"✓ Chatbot initialized (Session: {chatbot.session_id[:8]}...)")
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif user_input.lower() == 'help':
                show_help()
                continue
            elif user_input.lower().startswith('search '):
                handle_search_command(chatbot, user_input[7:])
                continue
            elif user_input.lower() == 'stats':
                show_stats(chatbot)
                continue
            elif user_input.lower() == 'explain':
                handle_explain_command(chatbot)
                continue
            elif user_input.lower().startswith('source '):
                handle_source_command(chatbot, user_input[7:])
                continue
            elif user_input.lower() == 'bias-report':
                show_bias_report()
                continue
            elif user_input.lower() == 'clear':
                chatbot.clear_session_memory()
                print("🗑️ Session memory cleared")
                continue
            
            # Regular chat
            print("🤖 AI: ", end="", flush=True)
            
            response = chatbot.chat(user_input)
            print(response.answer)
            
            # Show additional information if available
            if response.confidence < 0.7:
                print(f"ℹ️ Confidence: {response.confidence:.2f} (relatively low)")
            
            if response.sources:
                print(f"📚 Sources: {', '.join(response.sources[:2])}{'...' if len(response.sources) > 2 else ''}")
            
            if response.explanation and settings.explanation_detail_level != "low":
                explanation_summary = response.explanation.get("summary", {}).get("reasoning", "")
                if explanation_summary:
                    print(f"🔍 Reasoning: {explanation_summary[:100]}{'...' if len(explanation_summary) > 100 else ''}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def show_help():
    """Show help information"""
    
    help_text = """
📖 Available Commands:
- help: Show this help message
- search <query>: Search conversation memory
- stats: Show memory and performance statistics
- explain: Get detailed explanation of last response
- source <question>: Ask about data sources and reasoning
- bias-report: Show bias detection report
- clear: Clear session memory
- quit: Exit the chatbot

💡 Tips:
- The AI remembers your conversation and can reference previous topics
- All responses include transparency information when available
- Bias detection runs automatically on all responses
- You can ask "Why did you say that?" or "What sources did you use?"
    """
    print(help_text)


def handle_search_command(chatbot: TransparentChatbot, query: str):
    """Handle search command"""
    
    if not query:
        print("Please provide a search query. Example: search weather")
        return
    
    print(f"🔍 Searching for: {query}")
    
    results = chatbot.search_memory(query, limit=5)
    
    if not results:
        print("No results found.")
        return
    
    print(f"Found {len(results)} results:")
    
    for i, result in enumerate(results, 1):
        timestamp = result.get("timestamp", "Unknown time")
        if hasattr(timestamp, 'strftime'):
            time_str = timestamp.strftime("%Y-%m-%d %H:%M")
        else:
            time_str = str(timestamp)
        
        print(f"{i}. {time_str} - Score: {result.get('relevance_score', 0):.2f}")
        print(f"   Query: {result.get('user_query', '')[:60]}...")
        print(f"   Response: {result.get('ai_response', '')[:60]}...")
        print()


def show_stats(chatbot: TransparentChatbot):
    """Show statistics"""
    
    stats = chatbot.get_memory_stats()
    bias_stats = bias_detector.get_bias_statistics()
    
    print("📊 Chatbot Statistics:")
    print(f"Total Conversations: {stats['total_conversations']}")
    print(f"Current Session: {stats['current_session_conversations']}")
    print(f"Memory Limit: {stats['memory_limit']}")
    print(f"Retention Days: {stats['retention_days']}")
    
    if stats['topic_distribution']:
        print("\n📈 Topic Distribution:")
        for topic, count in list(stats['topic_distribution'].items())[:5]:
            print(f"  {topic}: {count}")
    
    if bias_stats['total_responses'] > 0:
        print(f"\n🛡️ Bias Detection:")
        print(f"Bias Detection Rate: {bias_stats['bias_detection_rate']:.1%}")
        print(f"Recent Trend: {bias_stats['recent_trend']}")


def handle_explain_command(chatbot: TransparentChatbot):
    """Handle explain command"""
    
    explanation = chatbot.explain_last_response()
    
    if not explanation:
        print("No previous response to explain.")
        return
    
    print("🔍 Detailed Explanation:")
    print(f"Method: {explanation.method}")
    print(f"Confidence: {explanation.confidence:.2f}")
    print(f"Reasoning: {explanation.decision_reasoning}")
    
    if explanation.key_factors:
        print("\nKey Factors:")
        for factor in explanation.key_factors[:3]:
            print(f"  - {factor}")
    
    if explanation.data_sources:
        print(f"\nData Sources: {', '.join(explanation.data_sources)}")


def handle_source_command(chatbot: TransparentChatbot, question: str):
    """Handle source question command"""
    
    if not question:
        question = "What sources did you use?"
    
    answer = chatbot.ask_about_source(question)
    print(f"📚 {answer}")


def show_bias_report():
    """Show bias detection report"""
    
    report = bias_detector.generate_bias_report()
    print("🛡️ Bias Detection Report:")
    print(report)


def run_api_mode(host: str, port: int):
    """Run the chatbot as an API server"""
    
    print(f"🚀 Starting API server on {host}:{port}")
    
    try:
        import uvicorn
        from src.api.routes import app
        
        print(f"✅ API server starting at http://{host}:{port}")
        print(f"📝 API documentation available at http://{host}:{port}/docs")
        uvicorn.run(app, host=host, port=port)
        
    except ImportError as e:
        print(f"❌ Error: Missing dependency - {e}")
        print("Please install missing packages with: pip install fastapi")
    except Exception as e:
        print(f"❌ Error starting API server: {e}")


def run_demo_mode():
    """Run a demo conversation"""
    
    print("🎯 Demo Mode - Showcasing AI Chatbot Capabilities")
    print("=" * 60)
    
    chatbot = TransparentChatbot()
    
    demo_queries = [
        "What is machine learning?",
        "How does machine learning relate to artificial intelligence?",
        "Can you search our previous conversation about machine learning?",
        "Why did you give that answer about machine learning and AI?"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n{i}. Demo Query: {query}")
        print("-" * 40)
        
        response = chatbot.chat(query)
        
        print(f"Response: {response.answer}")
        print(f"Confidence: {response.confidence:.2f}")
        
        if response.sources:
            print(f"Sources: {', '.join(response.sources)}")
        
        if response.explanation:
            reasoning = response.explanation.get("summary", {}).get("reasoning", "")
            if reasoning:
                print(f"Reasoning: {reasoning}")
        
        print()
    
    print("✅ Demo completed!")
    print("\nThe chatbot demonstrated:")
    print("- 🧠 Conversational memory and context")
    print("- 🔍 Search capabilities")
    print("- 📊 Transparency and explainability")
    print("- 🛡️ Bias detection (running in background)")


if __name__ == "__main__":
    main()
