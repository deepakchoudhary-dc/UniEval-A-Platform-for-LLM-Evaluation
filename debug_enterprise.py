#!/usr/bin/env python3
"""
Simple Enterprise Test - Debug Bias Detection

This script tests the bias detection with proper key handling.
"""

import sys
import os

# Add the src directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

def test_bias_detection_debug():
    """Debug bias detection to see actual return structure."""
    print("🔍 DEBUGGING BIAS DETECTION")
    print("="*50)
    
    try:
        sys.path.insert(0, os.path.join(current_dir, 'src', 'fairness'))
        from advanced_bias_detector import AdvancedBiasDetector
        
        detector = AdvancedBiasDetector()
        
        test_input = "Women shouldn't be in leadership because they're too emotional"
        print(f"Testing: '{test_input}'")
        
        result = detector.analyze_input_bias(test_input)
        
        print("\n📊 FULL RESULT STRUCTURE:")
        print(f"Type: {type(result)}")
        print(f"Keys: {list(result.keys())}")
        
        for key, value in result.items():
            print(f"  {key}: {value} (type: {type(value)})")
        
        return result
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_chatbot_response_debug():
    """Debug chatbot response structure."""
    print("\n🤖 DEBUGGING CHATBOT RESPONSE")
    print("="*50)
    
    try:
        sys.path.insert(0, os.path.join(current_dir, 'src', 'core'))
        from chatbot import TransparentChatbot
        
        chatbot = TransparentChatbot()
        
        test_input = "Women are too emotional for leadership roles"
        print(f"Testing: '{test_input}'")
        
        response = chatbot.chat(test_input)
        
        print("\n📊 FULL RESPONSE STRUCTURE:")
        print(f"Type: {type(response)}")
        
        if hasattr(response, '__dict__'):
            print(f"Attributes: {response.__dict__}")
        
        if hasattr(response, 'content'):
            print(f"Content length: {len(response.content)}")
            print(f"Content preview: {response.content[:100]}...")
        elif hasattr(response, 'response'):
            print(f"Response length: {len(response.response)}")
            print(f"Response preview: {response.response[:100]}...")
        else:
            print(f"String representation: {str(response)[:100]}...")
        
        return response
        
    except Exception as e:
        print(f"❌ Chatbot debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run debug tests."""
    print("🚀 ENTERPRISE DEBUG TEST")
    print("="*40)
    
    # Test bias detection
    bias_result = test_bias_detection_debug()
    
    # Test chatbot
    chatbot_result = test_chatbot_response_debug()
    
    print("\n🎯 DEBUG SUMMARY")
    print("="*40)
    print(f"Bias detection working: {'✅' if bias_result else '❌'}")
    print(f"Chatbot working: {'✅' if chatbot_result else '❌'}")

if __name__ == "__main__":
    main()
