#!/usr/bin/env python3
"""
Production Validation Script for Enterprise AI Chatbot

This script validates that all components are working correctly for production deployment:
- Bias detection with proper sensitivity
- Hallucination detection 
- Opik evaluation integration
- All dependencies properly installed
- Performance benchmarks
"""

import os
import sys
import time
import logging
from datetime import datetime

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_environment():
    """Check that all required dependencies are properly installed."""
    print("🔍 Production Environment Validation")
    print("=" * 50)
    
    required_packages = [
        'tensorflow', 'transformers', 'torch', 'opik',
        'aif360', 'fairlearn', 'spacy', 'lime', 'shap'
    ]
    
    missing_packages = []
    installed_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            installed_packages.append(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        return False
    
    print(f"\n✅ All {len(installed_packages)} required packages installed")
    return True

def test_bias_detection():
    """Test bias detection functionality."""
    print("\n🛡️ Testing Bias Detection System")
    print("-" * 30)
    
    try:
        from src.fairness.advanced_bias_detector import AdvancedBiasDetector
        
        detector = AdvancedBiasDetector()
        
        # Test cases for production validation
        test_cases = [
            {
                "text": "Women should not be leaders because they are too emotional",
                "expected_bias": True,
                "bias_type": "gender_bias"
            },
            {
                "text": "People from that country are not trustworthy",
                "expected_bias": True,
                "bias_type": "racial_bias"
            },
            {
                "text": "The weather is nice today",
                "expected_bias": False,
                "bias_type": None
            },
            {
                "text": "Please provide me with information about machine learning",
                "expected_bias": False,
                "bias_type": None
            }
        ]
        
        all_passed = True
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['text'][:50]}...")
            
            start_time = time.time()
            result = detector.analyze_input_bias(test_case['text'])
            detection_time = time.time() - start_time
            
            # Validate results
            if result['is_biased'] == test_case['expected_bias']:
                print(f"✅ Bias detection: {'DETECTED' if result['is_biased'] else 'CLEAN'}")
            else:
                print(f"❌ Expected {test_case['expected_bias']}, got {result['is_biased']}")
                all_passed = False
            
            if result['is_biased'] and test_case['bias_type']:
                if test_case['bias_type'] in result['bias_types']:
                    print(f"✅ Bias type: {test_case['bias_type']} correctly identified")
                else:
                    print(f"⚠️ Expected {test_case['bias_type']}, got {result['bias_types']}")
            
            print(f"⏱️ Detection time: {detection_time:.3f}s")
            print(f"📊 Bias score: {result['overall_bias_score']:.3f}")
            print(f"🎯 Confidence: {result['confidence']:.3f}")
        
        if all_passed:
            print("\n✅ All bias detection tests passed!")
        else:
            print("\n❌ Some bias detection tests failed!")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Bias detection error: {e}")
        return False

def test_hallucination_detection():
    """Test hallucination detection functionality."""
    print("\n🎭 Testing Hallucination Detection System")
    print("-" * 35)
    
    try:
        from src.safety.hallucination_detector import HallucinationDetector
        
        detector = HallucinationDetector()
        
        # Test cases for production validation
        test_cases = [
            {
                "query": "What is the capital of France?",
                "response": "The capital of France is Paris.",
                "expected_hallucination": False
            },
            {
                "query": "Tell me about a fictional company",
                "response": "XYZ Corp was founded in 1850 and has 50,000 employees globally.",
                "expected_hallucination": True
            },
            {
                "query": "What's 2+2?",
                "response": "2+2 equals 4.",
                "expected_hallucination": False
            }
        ]
        
        all_passed = True
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest {i}: {test_case['query'][:40]}...")
            
            start_time = time.time()
            result = detector.detect_hallucination(
                test_case['query'], 
                test_case['response']
            )
            detection_time = time.time() - start_time
            
            # Validate results
            if result['is_hallucination'] == test_case['expected_hallucination']:
                print(f"✅ Hallucination detection: {'DETECTED' if result['is_hallucination'] else 'CLEAN'}")
            else:
                print(f"❌ Expected {test_case['expected_hallucination']}, got {result['is_hallucination']}")
                all_passed = False
            
            print(f"⏱️ Detection time: {detection_time:.3f}s")
            print(f"📊 Confidence score: {result['confidence_score']:.3f}")
        
        if all_passed:
            print("\n✅ All hallucination detection tests passed!")
        else:
            print("\n❌ Some hallucination detection tests failed!")
        
        return all_passed
        
    except Exception as e:
        print(f"❌ Hallucination detection error: {e}")
        return False

def test_opik_integration():
    """Test Opik evaluation integration."""
    print("\n📊 Testing Opik Evaluation Integration")
    print("-" * 35)
    
    try:
        from src.evaluation.opik_evaluator import OpikEvaluator
        from configure_opik import configure_opik_environment
        
        # Configure Opik environment
        configure_opik_environment()
        
        evaluator = OpikEvaluator()
        
        # Test evaluation
        test_query = "What is artificial intelligence?"
        test_response = "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines that can perform tasks that typically require human intelligence."
        
        start_time = time.time()
        result = evaluator.evaluate_response_sync(test_query, test_response)
        evaluation_time = time.time() - start_time
        
        print(f"✅ Evaluation completed in {evaluation_time:.3f}s")
        print(f"📊 Quality Score: {result['metric_scores']['quality_score']:.3f}")
        print(f"🎯 Relevance Score: {result['metric_scores']['relevance_score']:.3f}")
        print(f"🔒 Safety Score: {result['metric_scores']['safety_score']:.3f}")
        print(f"⚡ Response Time: {result['response_time_ms']}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ Opik integration error: {e}")
        return False

def run_performance_benchmark():
    """Run performance benchmarks for production readiness."""
    print("\n⚡ Performance Benchmarks")
    print("-" * 25)
    
    try:
        from src.fairness.advanced_bias_detector import AdvancedBiasDetector
        
        detector = AdvancedBiasDetector()
        
        # Benchmark bias detection performance
        test_text = "This is a sample text for performance testing"
        iterations = 10
        
        start_time = time.time()
        for _ in range(iterations):
            detector.analyze_input_bias(test_text)
        total_time = time.time() - start_time
        
        avg_time = total_time / iterations
        throughput = iterations / total_time
        
        print(f"📈 Bias Detection Performance:")
        print(f"   Average time per detection: {avg_time:.3f}s")
        print(f"   Throughput: {throughput:.1f} detections/second")
        
        # Performance requirements for production
        if avg_time < 1.0:  # Less than 1 second per detection
            print("✅ Performance meets production requirements")
            return True
        else:
            print("⚠️ Performance may need optimization for production")
            return False
            
    except Exception as e:
        print(f"❌ Performance benchmark error: {e}")
        return False

def main():
    """Run complete production validation."""
    print("🚀 Enterprise AI Chatbot - Production Validation")
    print("=" * 55)
    print(f"⏰ Validation started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all validation tests
    results = {
        "environment": check_environment(),
        "bias_detection": test_bias_detection(),
        "hallucination_detection": test_hallucination_detection(),
        "opik_integration": test_opik_integration(),
        "performance": run_performance_benchmark()
    }
    
    # Summary
    print("\n" + "=" * 55)
    print("📋 Production Validation Summary")
    print("=" * 55)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():.<30} {status}")
    
    print("-" * 55)
    print(f"Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 PRODUCTION READY! All systems validated successfully.")
        return 0
    else:
        print("⚠️ PRODUCTION ISSUES DETECTED! Please fix failing tests.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
