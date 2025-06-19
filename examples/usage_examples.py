"""
Example usage of the Transparent AI Chatbot
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.chatbot import TransparentChatbot
from src.explainability.model_card import model_card_generator
from src.fairness.bias_detector import bias_detector


def basic_chat_example():
    """Example of basic chat functionality"""
    print("🤖 Basic Chat Example")
    print("=" * 30)
    
    # Initialize chatbot
    chatbot = TransparentChatbot()
    
    # Simple conversation
    response = chatbot.chat("What is artificial intelligence?")
    
    print(f"User: What is artificial intelligence?")
    print(f"AI: {response.answer}")
    print(f"Confidence: {response.confidence:.2f}")
    
    if response.sources:
        print(f"Sources: {', '.join(response.sources)}")
    
    return chatbot


def memory_and_search_example(chatbot):
    """Example of memory and search functionality"""
    print("\n🧠 Memory and Search Example")
    print("=" * 35)
    
    # Add more conversations to memory
    chatbot.chat("I'm particularly interested in machine learning")
    chatbot.chat("Can you explain neural networks?")
    chatbot.chat("What about deep learning algorithms?")
    
    # Search through conversation memory
    print("\nSearching for 'machine learning':")
    search_results = chatbot.search_memory("machine learning", limit=3)
    
    for i, result in enumerate(search_results, 1):
        print(f"{i}. {result.get('user_query', '')[:50]}...")
        print(f"   Relevance: {result.get('relevance_score', 0):.2f}")
    
    # Show memory statistics
    stats = chatbot.get_memory_stats()
    print(f"\nMemory Stats:")
    print(f"- Total conversations: {stats['total_conversations']}")
    print(f"- Session conversations: {stats['current_session_conversations']}")


def explainability_example(chatbot):
    """Example of explainability features"""
    print("\n🔍 Explainability Example")
    print("=" * 30)
    
    # Ask a question and get explanation
    response = chatbot.chat("Why is Python popular for data science?")
    
    print(f"Question: Why is Python popular for data science?")
    print(f"Answer: {response.answer[:100]}...")
    
    # Get detailed explanation
    explanation = chatbot.explain_last_response()
    
    if explanation:
        print(f"\nExplanation Method: {explanation.method}")
        print(f"Confidence: {explanation.confidence:.2f}")
        print(f"Reasoning: {explanation.decision_reasoning[:150]}...")
        
        if explanation.key_factors:
            print(f"Key Factors: {len(explanation.key_factors)} identified")
    
    # Ask about sources
    source_info = chatbot.ask_about_source("What sources did you use for that answer?")
    print(f"\nSource Information: {source_info}")


def bias_detection_example():
    """Example of bias detection"""
    print("\n🛡️ Bias Detection Example")
    print("=" * 30)
    
    # Test with potentially biased queries
    test_queries = [
        ("What can you tell me about engineers?", "Engineers are typically very logical people who enjoy problem-solving."),
        ("Are men better at math?", "Mathematical ability varies greatly among individuals regardless of gender."),
        ("Tell me about cultural differences", "Different cultures have unique traditions and perspectives that enrich our world.")
    ]
    
    for query, response in test_queries:
        print(f"\nTesting: {query}")
        
        # Run bias detection
        bias_result = bias_detector.detect_bias(query, response, {})
        
        print(f"Bias Detected: {bias_result['bias_detected']}")
        print(f"Bias Score: {bias_result['bias_score']:.2f}")
        print(f"Severity: {bias_result['severity']}")
        
        if bias_result['bias_types']:
            print(f"Bias Types: {', '.join(bias_result['bias_types'])}")
    
    # Show bias statistics
    print(f"\nBias Statistics:")
    stats = bias_detector.get_bias_statistics()
    print(f"- Total responses analyzed: {stats['total_responses']}")
    print(f"- Bias detection rate: {stats['bias_detection_rate']:.1%}")


def transparency_example(chatbot):
    """Example of transparency features"""
    print("\n📊 Transparency Example")
    print("=" * 25)
    
    # Ask about the AI's capabilities
    response = chatbot.chat("What are your capabilities and limitations?")
    
    print(f"Question: What are your capabilities and limitations?")
    print(f"Answer: {response.answer}")
    
    # Show transparency information
    print(f"\nTransparency Information:")
    print(f"- Session ID: {response.session_id}")
    print(f"- Timestamp: {response.timestamp}")
    print(f"- Confidence: {response.confidence:.2f}")
    
    if response.explanation:
        methods = response.explanation.get('methods_used', [])
        if methods:
            print(f"- Explanation methods: {', '.join(methods)}")


def model_card_example():
    """Example of model card generation"""
    print("\n📋 Model Card Example")
    print("=" * 22)
    
    # Generate model card
    model_card = model_card_generator.generate_model_card(include_metrics=False)
    
    # Show key information
    print(f"Model: {model_card['model_details']['name']}")
    print(f"Version: {model_card['model_details']['version']}")
    print(f"Type: {model_card['model_details']['type']}")
    
    print(f"\nIntended Uses ({len(model_card['intended_use']['primary_intended_uses'])}):")
    for use in model_card['intended_use']['primary_intended_uses'][:3]:
        print(f"- {use}")
    
    print(f"\nLimitations ({len(model_card['limitations']['known_limitations'])}):")
    for limitation in model_card['limitations']['known_limitations'][:3]:
        print(f"- {limitation}")
    
    print(f"\nEthical Considerations ({len(model_card['ethical_considerations']['considerations'])}):")
    for consideration in model_card['ethical_considerations']['considerations'][:2]:
        print(f"- {consideration['category']}: {consideration['description'][:60]}...")


def advanced_features_example(chatbot):
    """Example of advanced features"""
    print("\n🚀 Advanced Features Example")
    print("=" * 32)
    
    # Multi-turn conversation with context
    queries = [
        "I'm working on a machine learning project",
        "What algorithms would you recommend for classification?",
        "How do I evaluate model performance?",
        "What did we discuss about machine learning earlier?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. User: {query}")
        response = chatbot.chat(query)
        print(f"   AI: {response.answer[:100]}...")
        
        # Show context usage
        if response.sources:
            print(f"   Sources used: {len(response.sources)}")
    
    # Final search to show memory integration
    print(f"\nFinal search for 'machine learning project':")
    results = chatbot.search_memory("machine learning project", limit=2)
    
    for result in results:
        timestamp = result.get('timestamp', 'Unknown')
        if hasattr(timestamp, 'strftime'):
            time_str = timestamp.strftime('%H:%M:%S')
        else:
            time_str = str(timestamp)
        print(f"- Found at {time_str}: {result.get('user_query', '')[:40]}...")


def main():
    """Run all examples"""
    print("🎯 Transparent AI Chatbot Examples")
    print("=" * 40)
    print("This script demonstrates the key features of the chatbot")
    
    try:
        # Basic chat
        chatbot = basic_chat_example()
        
        # Memory and search
        memory_and_search_example(chatbot)
        
        # Explainability
        explainability_example(chatbot)
        
        # Bias detection
        bias_detection_example()
        
        # Transparency
        transparency_example(chatbot)
        
        # Model card
        model_card_example()
        
        # Advanced features
        advanced_features_example(chatbot)
        
        print("\n✅ All examples completed successfully!")
        print("\n💡 Try running the interactive mode: python main.py --mode interactive")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have:")
        print("1. Set up your .env file with API keys")
        print("2. Installed all dependencies: pip install -r requirements.txt")
        print("3. Run the setup script: python setup.py")


if __name__ == "__main__":
    main()
