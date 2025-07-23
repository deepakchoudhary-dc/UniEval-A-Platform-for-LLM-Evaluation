"""
Opik Evaluation Demo

This script demonstrates the LLM evaluation capabilities using Opik
integrated into the Transparent AI Chatbot.
"""

import asyncio
import time
from src.core.chatbot import TransparentChatbot
from src.evaluation.opik_evaluator import OpikEvaluator, create_evaluation_report
from src.utils.logger import get_logger

logger = get_logger(__name__)


async def demo_single_evaluation():
    """Demonstrate single response evaluation."""
    print("\n🔍 Single Response Evaluation Demo")
    print("=" * 50)
    
    # Create evaluator
    evaluator = OpikEvaluator(project_name="demo-evaluation")
    
    if not evaluator.is_available():
        print("⚠️ Opik not available, using fallback evaluation")
    
    # Test input/output
    user_input = "What are the benefits of renewable energy?"
    bot_response = """
    Renewable energy offers several key benefits:
    
    1. Environmental Impact: Significantly reduces greenhouse gas emissions and air pollution
    2. Sustainability: Uses naturally replenishing resources like solar, wind, and hydro
    3. Economic Benefits: Creates jobs and reduces long-term energy costs
    4. Energy Security: Reduces dependence on fossil fuel imports
    5. Health Benefits: Cleaner air leads to better public health outcomes
    
    These advantages make renewable energy crucial for combating climate change.
    """
    
    # Evaluate the response
    start_time = time.time()
    evaluation_result = await evaluator.evaluate_response(
        input_text=user_input,
        output_text=bot_response,
        context="Educational context about environmental topics",
        evaluation_criteria=["relevance", "hallucination", "moderation", "faithfulness"]
    )
    evaluation_time = time.time() - start_time
    
    # Display results
    print(f"📊 Evaluation completed in {evaluation_time:.2f} seconds")
    print(f"Overall Score: {evaluation_result['overall_score']:.3f}")
    print(f"Timestamp: {evaluation_result['timestamp']}")
    
    print("\n📈 Detailed Metrics:")
    for metric_name, metric_data in evaluation_result['metrics'].items():
        if isinstance(metric_data, dict):
            score = metric_data.get('score', 'N/A')
            print(f"  {metric_name}: {score}")
    
    print("\n💡 Recommendations:")
    for i, recommendation in enumerate(evaluation_result['recommendations'], 1):
        print(f"  {i}. {recommendation}")


async def demo_chatbot_integration():
    """Demonstrate evaluation integrated with the chatbot."""
    print("\n🤖 Chatbot Integration Demo")
    print("=" * 50)
    
    # Create chatbot with evaluation enabled
    chatbot = TransparentChatbot()
    
    test_queries = [
        "Explain quantum computing in simple terms.",
        "What are the ethical implications of AI?",
        "How does photosynthesis work?",
        "What causes climate change?"
    ]
    
    print("Testing chatbot with evaluation...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔤 Query {i}: {query}")
        
        # Get response (evaluation happens in background)
        response = chatbot.chat(query)
        
        print(f"📝 Response: {response.answer[:150]}...")
        print(f"🎯 Confidence: {response.confidence:.3f}")
        
        # Small delay to allow background evaluation
        await asyncio.sleep(1)
    
    # Wait a bit more for evaluations to complete
    print("\n⏳ Waiting for evaluations to complete...")
    await asyncio.sleep(3)
    
    # Get evaluation summary
    print("\n📊 Evaluation Summary:")
    summary = chatbot.get_evaluation_summary(limit=len(test_queries))
    
    if "message" not in summary:
        print(f"Total Evaluations: {summary.get('total_evaluations', 0)}")
        print(f"Average Score: {summary.get('average_score', 0):.3f}")
        print(f"Score Range: {summary.get('min_score', 0):.3f} - {summary.get('max_score', 0):.3f}")
    else:
        print(summary["message"])


async def demo_conversation_evaluation():
    """Demonstrate conversation-level evaluation."""
    print("\n💬 Conversation Evaluation Demo")
    print("=" * 50)
    
    chatbot = TransparentChatbot()
    
    # Simulate a conversation
    conversation = [
        "Hello, I'm interested in learning about machine learning.",
        "Can you explain what neural networks are?",
        "How do neural networks learn from data?",
        "What are some real-world applications of neural networks?",
        "Are there any limitations to neural networks?"
    ]
    
    print("Building conversation...")
    for query in conversation:
        response = chatbot.chat(query)
        print(f"User: {query}")
        print(f"Bot: {response.answer[:100]}...")
        print()
    
    # Evaluate conversation quality
    print("Evaluating conversation quality...")
    conversation_eval = await chatbot.evaluate_conversation_quality()
    
    if "error" not in conversation_eval:
        print(f"📊 Conversation Metrics:")
        metrics = conversation_eval.get("conversation_metrics", {})
        print(f"  Turn Count: {metrics.get('turn_count', 0)}")
        print(f"  Coherence Score: {metrics.get('coherence_score', 0):.3f}")
        print(f"  Engagement Score: {metrics.get('engagement_score', 0):.3f}")
        print(f"  Quality Trend: {metrics.get('quality_trend', 'unknown')}")
        print(f"  Overall Score: {conversation_eval.get('overall_conversation_score', 0):.3f}")
        
        print(f"\n💡 Conversation Recommendations:")
        for rec in conversation_eval.get("recommendations", []):
            print(f"  • {rec}")
    else:
        print(f"❌ Error: {conversation_eval['error']}")


async def demo_batch_evaluation():
    """Demonstrate batch evaluation capabilities."""
    print("\n📦 Batch Evaluation Demo")
    print("=" * 50)
    
    evaluator = OpikEvaluator(project_name="batch-demo")
    
    # Sample batch data
    batch_data = [
        {
            "input": "What is Python?",
            "output": "Python is a high-level programming language known for its simplicity and readability.",
            "context": "Programming education"
        },
        {
            "input": "Explain machine learning",
            "output": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            "context": "AI education"
        },
        {
            "input": "What causes rain?",
            "output": "Rain is caused by water vapor in clouds condensing and falling as precipitation.",
            "context": "Weather science"
        }
    ]
    
    print(f"Evaluating batch of {len(batch_data)} responses...")
    
    start_time = time.time()
    batch_results = await evaluator.batch_evaluate(
        evaluation_batch=batch_data,
        criteria=["relevance", "accuracy", "clarity"]
    )
    batch_time = time.time() - start_time
    
    print(f"⏱️ Batch evaluation completed in {batch_time:.2f} seconds")
    
    for i, result in enumerate(batch_results, 1):
        print(f"\nItem {i}:")
        print(f"  Score: {result['overall_score']:.3f}")
        print(f"  Input: {batch_data[i-1]['input']}")
        print(f"  Recommendations: {'; '.join(result['recommendations'][:2])}")


def demo_evaluation_report():
    """Demonstrate evaluation report generation."""
    print("\n📋 Evaluation Report Demo")
    print("=" * 50)
    
    # Mock evaluation results for demonstration
    mock_results = [
        {
            "timestamp": "2025-01-23T12:00:00",
            "overall_score": 0.85,
            "recommendations": ["Good relevance", "Consider conciseness"]
        },
        {
            "timestamp": "2025-01-23T12:05:00",
            "overall_score": 0.92,
            "recommendations": ["Excellent accuracy", "Well structured"]
        },
        {
            "timestamp": "2025-01-23T12:10:00",
            "overall_score": 0.78,
            "recommendations": ["Improve clarity", "Good content"]
        }
    ]
    
    report = create_evaluation_report(mock_results)
    print(report)


async def main():
    """Run all evaluation demos."""
    print("🚀 Opik LLM Evaluation Demo")
    print("==========================")
    
    try:
        # Run all demos
        await demo_single_evaluation()
        await demo_chatbot_integration()
        await demo_conversation_evaluation()
        await demo_batch_evaluation()
        demo_evaluation_report()
        
        print("\n✅ All demos completed successfully!")
        print("\n📈 Next Steps:")
        print("1. Check the Opik dashboard for detailed evaluation insights")
        print("2. Integrate evaluation endpoints into your applications")
        print("3. Set up automated evaluation pipelines")
        print("4. Use evaluation data to improve your LLM applications")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        logger.error(f"Demo error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
