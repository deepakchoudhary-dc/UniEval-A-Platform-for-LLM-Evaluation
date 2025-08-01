"""
Example usage of the Comprehensive LLM Evaluation Framework
This demonstrates how to use all 7 evaluation modules to assess an LLM
"""

import asyncio
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluation.comprehensive_evaluator import ComprehensiveLLMEvaluator


async def main():
    """
    Example comprehensive evaluation of an LLM response
    """
    
    # Configuration for the evaluation framework
    config = {
        'quality_performance': {
            'enable_factual_correctness': True,
            'enable_task_accuracy': True,
            'enable_faithfulness': True,
            'enable_relevancy': True,
            'enable_fluency': True,
            'enable_coherence': True,
            'enable_readability': True,
            'enable_bleu': True,
            'enable_rouge': True,
            'enable_meteor': True,
            'enable_bertscore': True,
            'enable_diversity': True,
            'enable_mauve': True,
            'enable_perplexity': True,
            'enable_token_efficiency': True
        },
        'reliability_robustness': {
            'enable_robustness': True,
            'enable_consistency': True,
            'enable_adversarial': True,
            'enable_generalization': True,
            'enable_memorization': True,
            'enable_hallucination': True,
            'enable_calibration': True,
            'enable_uncertainty': True,
            'enable_knowledge_cutoff': True
        },
        'safety_ethics': {
            'enable_harmfulness': True,
            'enable_toxicity': True,
            'enable_bias': True,
            'enable_misinformation': True,
            'enable_privacy': True,
            'enable_legal_compliance': True,
            'enable_content_moderation': True,
            'enable_fairness': True,
            'enable_transparency': True
        },
        'operational_efficiency': {
            'enable_latency': True,
            'enable_throughput': True,
            'enable_token_efficiency': True,
            'enable_resource_utilization': True,
            'enable_cost_efficiency': True,
            'enable_energy_efficiency': True,
            'enable_caching': True
        },
        'user_experience': {
            'enable_satisfaction': True,
            'enable_helpfulness': True,
            'enable_engagement': True,
            'enable_personalization': True,
            'enable_conversation_flow': True,
            'enable_emotional_intelligence': True,
            'enable_accessibility': True,
            'enable_usability': True
        },
        'agentic_capabilities': {
            'enable_task_planning': True,
            'enable_goal_achievement': True,
            'enable_tool_use': True,
            'enable_reasoning': True,
            'enable_decision_making': True,
            'enable_learning': True,
            'enable_multi_step': True,
            'enable_collaboration': True,
            'enable_autonomy': True
        },
        'evaluation_methodologies': {
            'enable_human_evaluation': True,
            'enable_benchmark': True,
            'enable_adversarial': True,
            'enable_crossvalidation': True,
            'enable_ab_testing': True,
            'enable_meta_evaluation': True,
            'enable_longitudinal': True,
            'enable_multidimensional': True,
            'enable_comparative': True,
            'enable_statistical': True,
            'enable_error_analysis': True,
            'enable_robustness': True
        }
    }
    
    # Initialize the comprehensive evaluator
    evaluator = ComprehensiveLLMEvaluator(config)
    
    # Example evaluation scenario
    model_name = "GPT-4"
    input_text = "Explain quantum computing and its potential applications in cryptography."
    output_text = """Quantum computing is a revolutionary computing paradigm that leverages the principles of quantum mechanics to process information in fundamentally different ways than classical computers.

Key Concepts:
1. Qubits: Unlike classical bits that exist in states of 0 or 1, qubits can exist in superposition, being both 0 and 1 simultaneously.
2. Entanglement: Qubits can be entangled, meaning their states are correlated regardless of distance.
3. Quantum Interference: Quantum algorithms use interference to amplify correct answers and cancel out wrong ones.

Applications in Cryptography:

1. Breaking Current Encryption:
   - Shor's algorithm can efficiently factor large integers, threatening RSA encryption
   - Grover's algorithm can search unsorted databases quadratically faster, affecting symmetric encryption

2. Quantum Cryptography:
   - Quantum Key Distribution (QKD) provides theoretically unbreakable communication
   - Uses quantum properties to detect eavesdropping attempts
   - Ensures perfect secrecy through quantum mechanics principles

3. Post-Quantum Cryptography:
   - Development of encryption methods resistant to quantum attacks
   - Lattice-based, hash-based, and multivariate cryptographic systems
   - NIST standardization process for quantum-resistant algorithms

Timeline and Challenges:
- Current quantum computers are still in early stages (NISQ era)
- Fault-tolerant quantum computers needed for cryptographic attacks are still years away
- Organizations should begin preparing quantum-resistant security measures now

Quantum computing represents both a threat to current cryptographic systems and an opportunity for fundamentally secure communication methods."""
    
    reference_text = """Quantum computing uses quantum mechanical phenomena to process information. It could break current encryption methods but also enable new forms of secure communication."""
    
    # Additional context for evaluation
    context = {
        'task_domain': 'technical_explanation',
        'expected_length': 'detailed',
        'target_audience': 'technical_professionals',
        'evaluation_timestamp': '2024-01-15T10:30:00Z',
        'human_annotations': [
            {'accuracy': 0.9, 'clarity': 0.85, 'completeness': 0.88},
            {'accuracy': 0.92, 'clarity': 0.87, 'completeness': 0.90}
        ],
        'performance_history': [0.82, 0.85, 0.87, 0.89, 0.88],
        'ab_test_results': {
            'statistical_power': 0.85,
            'effect_size': 0.15,
            'sample_size': 150,
            'required_sample_size': 120
        }
    }
    
    print("🚀 Starting Comprehensive LLM Evaluation...")
    print(f"Model: {model_name}")
    print(f"Task: Technical Explanation - Quantum Computing & Cryptography")
    print("=" * 80)
    
    # Run comprehensive evaluation
    try:
        result = await evaluator.evaluate_comprehensive(
            model_name=model_name,
            input_text=input_text,
            output_text=output_text,
            reference_text=reference_text,
            context=context,
            task_type="technical_explanation"
        )
        
        print("\n📊 EVALUATION RESULTS")
        print("=" * 80)
        
        # Display category scores
        print("\n🎯 CATEGORY SCORES:")
        categories = [
            ("Quality & Performance", result.category_scores.get('quality_performance', 0)),
            ("Reliability & Robustness", result.category_scores.get('reliability_robustness', 0)),
            ("Safety & Ethics", result.category_scores.get('safety_ethics', 0)),
            ("Operational Efficiency", result.category_scores.get('operational_efficiency', 0)),
            ("User Experience", result.category_scores.get('user_experience', 0)),
            ("Agentic Capabilities", result.category_scores.get('agentic_capabilities', 0)),
            ("Evaluation Methodologies", result.category_scores.get('evaluation_methodologies', 0))
        ]
        
        for category, score in categories:
            bar_length = int(score * 20)  # Scale to 20 characters
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{category:25} │ {bar} │ {score:.3f}")
        
        print(f"\n🏆 OVERALL SCORE: {result.overall_score:.3f}")
        
        # Display some detailed metrics
        print("\n📈 DETAILED METRICS (Sample):")
        
        if result.quality_performance:
            print("\nQuality & Performance:")
            for metric, value in list(result.quality_performance.items())[:5]:
                print(f"  • {metric}: {value:.3f}")
        
        if result.safety_ethics:
            print("\nSafety & Ethics:")
            for metric, value in list(result.safety_ethics.items())[:5]:
                print(f"  • {metric}: {value:.3f}")
        
        if result.agentic_capabilities:
            print("\nAgentic Capabilities:")
            for metric, value in list(result.agentic_capabilities.items())[:5]:
                print(f"  • {metric}: {value:.3f}")
        
        # Display recommendations
        if result.recommendations:
            print("\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(result.recommendations[:5], 1):
                print(f"  {i}. {rec}")
        
        # Performance analysis
        print(f"\n⏱️  EVALUATION METADATA:")
        print(f"  • Evaluation ID: {result.evaluation_id}")
        print(f"  • Timestamp: {result.timestamp}")
        print(f"  • Task Type: {result.task_type}")
        print(f"  • Total Metrics Evaluated: {sum(len(getattr(result, attr, {})) for attr in ['quality_performance', 'reliability_robustness', 'safety_ethics', 'operational_efficiency', 'user_experience', 'agentic_capabilities', 'evaluation_methodologies'])}")
        
        print("\n✅ Comprehensive evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
