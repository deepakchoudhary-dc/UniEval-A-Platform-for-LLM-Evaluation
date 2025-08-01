#!/usr/bin/env python3

import asyncio
import sys
import os
sys.path.append('d:/chatbot')

from src.evaluation.agentic_capabilities import AgenticCapabilitiesEvaluator

async def test_agentic_evaluation():
    # Create the evaluator with default config
    config = {
        'enable_reasoning': True,
        'enable_planning': True,
        'enable_tool_use': True
    }
    
    evaluator = AgenticCapabilitiesEvaluator(config)
    
    # Test the evaluation
    query = "Test query"
    response = "Test response"
    context = {'context': 'Test context'}  # Pass as dict instead of string
    
    try:
        result = await evaluator.evaluate_agentic_capabilities(
            query=query,
            response=response,
            context=context
        )
        print("SUCCESS: Result type:", type(result))
        print("Result:", result)
        print("Result keys:", list(result.keys()) if isinstance(result, dict) else "NOT A DICT")
        
        # Test the .get() method that's causing the error
        if isinstance(result, dict):
            detailed_scores = result.get("detailed_scores", {})
            print("detailed_scores type:", type(detailed_scores))
            print("detailed_scores:", detailed_scores)
        else:
            print("ERROR: Result is not a dictionary, it's:", type(result))
            
    except Exception as e:
        print("ERROR:", str(e))
        print("ERROR TYPE:", type(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_agentic_evaluation())
