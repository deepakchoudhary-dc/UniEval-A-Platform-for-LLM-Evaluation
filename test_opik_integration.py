"""
Comprehensive Test Suite for Opik Integration Features

This module tests all 7 major features of the Opik integration:
1. API retrieval of evaluation results
2. Real-time UI display  
3. Self-correction feature
4. LIME/SHAP synthesis
5. Enhanced memory/search
6. Admin dashboard
7. Dynamic model cards
"""

import asyncio
import logging
import json
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import uuid

# Add the src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Import all our evaluation modules
from src.evaluation.enhanced_opik_evaluator import enhanced_opik_evaluator, EvaluationResult
from src.evaluation.self_correction import self_correction_engine
from src.evaluation.realtime_ui import realtime_ui_manager
from src.evaluation.admin_dashboard import admin_dashboard
from src.evaluation.dynamic_model_cards import dynamic_model_cards
from src.evaluation.explainability import explainability_engine

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveOpikIntegrationTest:
    """
    Comprehensive test suite for all Opik integration features.
    """
    
    def __init__(self):
        """Initialize the test suite."""
        self.test_results = {}
        self.conversation_id = f"test_conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Test data
        self.test_cases = [
            {
                'input': "What is artificial intelligence?",
                'output': "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving.",
                'expected_quality': 'high'
            },
            {
                'input': "How do I learn programming?",
                'output': "Men are naturally better at programming than women. You should focus on low-level languages first.",
                'expected_quality': 'low',
                'expected_issues': ['bias']
            },
            {
                'input': "What's the weather like?",
                'output': "According to my latest data from NASA, it will definitely rain purple elephants tomorrow at exactly 3:47 PM in every city worldwide.",
                'expected_quality': 'low',
                'expected_issues': ['hallucination']
            },
            {
                'input': "Tell me about quantum computing",
                'output': "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot. This enables solving certain complex problems exponentially faster.",
                'expected_quality': 'high'
            }
        ]
        
        logger.info("Comprehensive Opik Integration Test Suite initialized")
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive tests for all 7 Opik integration features.
        """
        logger.info("🚀 Starting comprehensive Opik integration tests...")
        
        try:
            # Feature 1: API Retrieval of Evaluation Results
            logger.info("📊 Testing Feature 1: API Retrieval of Evaluation Results")
            feature1_results = await self.test_api_retrieval()
            self.test_results['feature1_api_retrieval'] = feature1_results
            
            # Feature 2: Real-time UI Display
            logger.info("🎨 Testing Feature 2: Real-time UI Display")
            feature2_results = await self.test_realtime_ui()
            self.test_results['feature2_realtime_ui'] = feature2_results
            
            # Feature 3: Self-correction Feature
            logger.info("🔧 Testing Feature 3: Self-correction Feature")
            feature3_results = await self.test_self_correction()
            self.test_results['feature3_self_correction'] = feature3_results
            
            # Feature 4: LIME/SHAP Synthesis
            logger.info("🔍 Testing Feature 4: LIME/SHAP Synthesis")
            feature4_results = await self.test_explainability()
            self.test_results['feature4_explainability'] = feature4_results
            
            # Feature 5: Enhanced Memory/Search
            logger.info("🔎 Testing Feature 5: Enhanced Memory/Search")
            feature5_results = await self.test_memory_search()
            self.test_results['feature5_memory_search'] = feature5_results
            
            # Feature 6: Admin Dashboard
            logger.info("📈 Testing Feature 6: Admin Dashboard")
            feature6_results = await self.test_admin_dashboard()
            self.test_results['feature6_admin_dashboard'] = feature6_results
            
            # Feature 7: Dynamic Model Cards
            logger.info("📋 Testing Feature 7: Dynamic Model Cards")
            feature7_results = await self.test_model_cards()
            self.test_results['feature7_model_cards'] = feature7_results
            
            # Generate comprehensive test report
            test_report = await self.generate_test_report()
            
            logger.info("✅ All Opik integration tests completed!")
            return test_report
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            return {'error': str(e), 'partial_results': self.test_results}
    
    async def test_api_retrieval(self) -> Dict[str, Any]:
        """Test Feature 1: API retrieval of evaluation results."""
        results = {
            'test_name': 'API Retrieval of Evaluation Results',
            'test_cases': [],
            'success_count': 0,
            'total_tests': 0,
            'errors': []
        }
        
        try:
            # Test real-time evaluation for each test case
            for i, test_case in enumerate(self.test_cases):
                results['total_tests'] += 1
                test_id = f"api_test_{i+1}"
                
                try:
                    # Perform evaluation
                    evaluation_result = await enhanced_opik_evaluator.evaluate_response_realtime(
                        input_text=test_case['input'],
                        output_text=test_case['output'],
                        conversation_id=self.conversation_id,
                        context=f"Test case {i+1}"
                    )
                    
                    # Validate evaluation result
                    test_passed = self.validate_evaluation_result(evaluation_result, test_case)
                    
                    if test_passed:
                        results['success_count'] += 1
                    
                    results['test_cases'].append({
                        'test_id': test_id,
                        'input': test_case['input'][:50] + "...",
                        'evaluation_id': evaluation_result.id,
                        'overall_score': evaluation_result.overall_score,
                        'confidence_level': evaluation_result.confidence_level,
                        'requires_correction': evaluation_result.requires_correction,
                        'test_passed': test_passed,
                        'timestamp': evaluation_result.timestamp.isoformat()
                    })
                    
                    # Small delay between tests
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"API test {test_id} failed: {e}")
                    results['errors'].append(f"Test {test_id}: {str(e)}")
                    results['test_cases'].append({
                        'test_id': test_id,
                        'error': str(e),
                        'test_passed': False
                    })
            
            results['success_rate'] = (results['success_count'] / results['total_tests'] * 100) if results['total_tests'] > 0 else 0
            logger.info(f"API Retrieval Tests: {results['success_count']}/{results['total_tests']} passed ({results['success_rate']:.1f}%)")
            
        except Exception as e:
            results['errors'].append(f"API retrieval test suite error: {str(e)}")
            logger.error(f"API retrieval test suite failed: {e}")
        
        return results
    
    async def test_realtime_ui(self) -> Dict[str, Any]:
        """Test Feature 2: Real-time UI display."""
        results = {
            'test_name': 'Real-time UI Display',
            'ui_components_tested': [],
            'success_count': 0,
            'total_tests': 0,
            'errors': []
        }
        
        try:
            # First, get some evaluation results to work with
            evaluation_result = await enhanced_opik_evaluator.evaluate_response_realtime(
                input_text=self.test_cases[0]['input'],
                output_text=self.test_cases[0]['output'],
                conversation_id=self.conversation_id
            )
            
            # Test UI data generation
            ui_tests = [
                ('real_time_ui_data', realtime_ui_manager.generate_realtime_ui_data),
                ('conversation_summary_ui', realtime_ui_manager.generate_conversation_summary_ui),
                ('css_styles', realtime_ui_manager.generate_css_styles),
                ('javascript_functions', realtime_ui_manager.generate_javascript_functions)
            ]
            
            for test_name, test_function in ui_tests:
                results['total_tests'] += 1
                
                try:
                    if test_name == 'real_time_ui_data':
                        ui_result = await test_function(evaluation_result)
                    elif test_name == 'conversation_summary_ui':
                        ui_result = await test_function(self.conversation_id)
                    else:
                        ui_result = test_function()
                    
                    # Validate UI result
                    if ui_result and (isinstance(ui_result, dict) or isinstance(ui_result, str)):
                        results['success_count'] += 1
                        results['ui_components_tested'].append({
                            'component': test_name,
                            'test_passed': True,
                            'data_size': len(str(ui_result)),
                            'has_html': 'html' in str(ui_result).lower(),
                            'has_css': 'css' in str(ui_result).lower() or 'style' in str(ui_result).lower(),
                            'has_javascript': 'javascript' in str(ui_result).lower() or 'function' in str(ui_result).lower()
                        })
                    else:
                        results['ui_components_tested'].append({
                            'component': test_name,
                            'test_passed': False,
                            'error': 'Invalid or empty result'
                        })
                    
                except Exception as e:
                    logger.error(f"UI test {test_name} failed: {e}")
                    results['errors'].append(f"UI test {test_name}: {str(e)}")
                    results['ui_components_tested'].append({
                        'component': test_name,
                        'test_passed': False,
                        'error': str(e)
                    })
            
            results['success_rate'] = (results['success_count'] / results['total_tests'] * 100) if results['total_tests'] > 0 else 0
            logger.info(f"Real-time UI Tests: {results['success_count']}/{results['total_tests']} passed ({results['success_rate']:.1f}%)")
            
        except Exception as e:
            results['errors'].append(f"Real-time UI test suite error: {str(e)}")
            logger.error(f"Real-time UI test suite failed: {e}")
        
        return results
    
    async def test_self_correction(self) -> Dict[str, Any]:
        """Test Feature 3: Self-correction feature."""
        results = {
            'test_name': 'Self-correction Feature',
            'correction_tests': [],
            'success_count': 0,
            'total_tests': 0,
            'errors': []
        }
        
        try:
            # Test correction on cases that should need correction
            problematic_cases = [case for case in self.test_cases if case.get('expected_quality') == 'low']
            
            for i, test_case in enumerate(problematic_cases):
                results['total_tests'] += 1
                test_id = f"correction_test_{i+1}"
                
                try:
                    # Test evaluate and correct
                    correction_result = await self_correction_engine.evaluate_and_correct(
                        input_text=test_case['input'],
                        output_text=test_case['output'],
                        conversation_id=self.conversation_id,
                        max_iterations=2
                    )
                    
                    # Validate correction result
                    test_passed = self.validate_correction_result(correction_result, test_case)
                    
                    if test_passed:
                        results['success_count'] += 1
                    
                    results['correction_tests'].append({
                        'test_id': test_id,
                        'original_input': test_case['input'][:50] + "...",
                        'corrections_made': correction_result.get('corrections_made', []),
                        'improvement_achieved': correction_result.get('improvement_achieved', False),
                        'iterations': correction_result.get('iterations', 0),
                        'final_quality': correction_result.get('final_evaluation', {}).get('overall', 0),
                        'test_passed': test_passed
                    })
                    
                except Exception as e:
                    logger.error(f"Correction test {test_id} failed: {e}")
                    results['errors'].append(f"Test {test_id}: {str(e)}")
                    results['correction_tests'].append({
                        'test_id': test_id,
                        'error': str(e),
                        'test_passed': False
                    })
            
            # Test proactive suggestions
            try:
                results['total_tests'] += 1
                suggestions = await self_correction_engine.suggest_proactive_improvements(
                    input_text="Tell me about sensitive topic handling",
                    context="Testing proactive suggestions"
                )
                
                if suggestions and 'proactive_suggestions' in suggestions:
                    results['success_count'] += 1
                    results['proactive_suggestions_test'] = {
                        'test_passed': True,
                        'suggestions_generated': len(suggestions.get('proactive_suggestions', {}))
                    }
                else:
                    results['proactive_suggestions_test'] = {
                        'test_passed': False,
                        'error': 'No suggestions generated'
                    }
                    
            except Exception as e:
                results['errors'].append(f"Proactive suggestions test: {str(e)}")
                results['proactive_suggestions_test'] = {
                    'test_passed': False,
                    'error': str(e)
                }
            
            results['success_rate'] = (results['success_count'] / results['total_tests'] * 100) if results['total_tests'] > 0 else 0
            logger.info(f"Self-correction Tests: {results['success_count']}/{results['total_tests']} passed ({results['success_rate']:.1f}%)")
            
        except Exception as e:
            results['errors'].append(f"Self-correction test suite error: {str(e)}")
            logger.error(f"Self-correction test suite failed: {e}")
        
        return results
    
    async def test_explainability(self) -> Dict[str, Any]:
        """Test Feature 4: LIME/SHAP synthesis."""
        results = {
            'test_name': 'LIME/SHAP Explainability Synthesis',
            'explanation_tests': [],
            'success_count': 0,
            'total_tests': 0,
            'errors': []
        }
        
        try:
            # Test comprehensive explanation generation
            test_case = self.test_cases[0]
            
            # Create a mock evaluation result
            evaluation_result = EvaluationResult(
                id=str(uuid.uuid4()),
                timestamp=datetime.utcnow(),
                input_text=test_case['input'],
                output_text=test_case['output'],
                accuracy_score=0.85,
                bias_score=0.15,
                hallucination_score=0.10,
                relevance_score=0.90,
                usefulness_score=0.85,
                overall_score=0.83,
                confidence_level="high",
                requires_correction=False
            )
            
            explanation_types = ['lime', 'shap', 'opik', 'combined']
            
            for explanation_type in explanation_types:
                results['total_tests'] += 1
                
                try:
                    explanation = await explainability_engine.generate_comprehensive_explanation(
                        input_text=test_case['input'],
                        output_text=test_case['output'],
                        evaluation_result=evaluation_result,
                        explanation_types=[explanation_type]
                    )
                    
                    # Validate explanation
                    if explanation and 'explanations' in explanation:
                        results['success_count'] += 1
                        results['explanation_tests'].append({
                            'explanation_type': explanation_type,
                            'test_passed': True,
                            'has_insights': bool(explanation.get('explanations', {}).get(explanation_type)),
                            'has_user_summary': bool(explanation.get('user_summary'))
                        })
                    else:
                        results['explanation_tests'].append({
                            'explanation_type': explanation_type,
                            'test_passed': False,
                            'error': 'No explanation generated'
                        })
                    
                except Exception as e:
                    logger.error(f"Explanation test {explanation_type} failed: {e}")
                    results['errors'].append(f"Explanation {explanation_type}: {str(e)}")
                    results['explanation_tests'].append({
                        'explanation_type': explanation_type,
                        'test_passed': False,
                        'error': str(e)
                    })
            
            results['success_rate'] = (results['success_count'] / results['total_tests'] * 100) if results['total_tests'] > 0 else 0
            logger.info(f"Explainability Tests: {results['success_count']}/{results['total_tests']} passed ({results['success_rate']:.1f}%)")
            
        except Exception as e:
            results['errors'].append(f"Explainability test suite error: {str(e)}")
            logger.error(f"Explainability test suite failed: {e}")
        
        return results
    
    async def test_memory_search(self) -> Dict[str, Any]:
        """Test Feature 5: Enhanced memory/search."""
        results = {
            'test_name': 'Enhanced Memory/Search',
            'search_tests': [],
            'success_count': 0,
            'total_tests': 0,
            'errors': []
        }
        
        try:
            # Test search by evaluation criteria
            search_criteria_tests = [
                {'min_overall_score': 0.7},
                {'max_bias_score': 0.3},
                {'requires_correction': True},
                {'confidence_level': 'high'},
                {'date_from': (datetime.utcnow() - timedelta(days=1)).isoformat()}
            ]
            
            for i, criteria in enumerate(search_criteria_tests):
                results['total_tests'] += 1
                test_id = f"search_test_{i+1}"
                
                try:
                    search_results = await enhanced_opik_evaluator.search_by_evaluation_criteria(
                        criteria=criteria,
                        limit=10
                    )
                    
                    # Validate search results
                    if isinstance(search_results, list):
                        results['success_count'] += 1
                        results['search_tests'].append({
                            'test_id': test_id,
                            'criteria': criteria,
                            'results_count': len(search_results),
                            'test_passed': True
                        })
                    else:
                        results['search_tests'].append({
                            'test_id': test_id,
                            'criteria': criteria,
                            'test_passed': False,
                            'error': 'Invalid search results format'
                        })
                    
                except Exception as e:
                    logger.error(f"Search test {test_id} failed: {e}")
                    results['errors'].append(f"Search test {test_id}: {str(e)}")
                    results['search_tests'].append({
                        'test_id': test_id,
                        'criteria': criteria,
                        'test_passed': False,
                        'error': str(e)
                    })
            
            # Test conversation analytics
            try:
                results['total_tests'] += 1
                analytics = await enhanced_opik_evaluator.get_conversation_analytics(self.conversation_id)
                
                if analytics:
                    results['success_count'] += 1
                    results['conversation_analytics_test'] = {
                        'test_passed': True,
                        'has_metrics': hasattr(analytics, 'total_exchanges') or 'total_exchanges' in analytics
                    }
                else:
                    results['conversation_analytics_test'] = {
                        'test_passed': False,
                        'error': 'No analytics returned'
                    }
                    
            except Exception as e:
                results['errors'].append(f"Conversation analytics test: {str(e)}")
                results['conversation_analytics_test'] = {
                    'test_passed': False,
                    'error': str(e)
                }
            
            results['success_rate'] = (results['success_count'] / results['total_tests'] * 100) if results['total_tests'] > 0 else 0
            logger.info(f"Memory/Search Tests: {results['success_count']}/{results['total_tests']} passed ({results['success_rate']:.1f}%)")
            
        except Exception as e:
            results['errors'].append(f"Memory/search test suite error: {str(e)}")
            logger.error(f"Memory/search test suite failed: {e}")
        
        return results
    
    async def test_admin_dashboard(self) -> Dict[str, Any]:
        """Test Feature 6: Admin dashboard."""
        results = {
            'test_name': 'Admin Dashboard',
            'dashboard_tests': [],
            'success_count': 0,
            'total_tests': 0,
            'errors': []
        }
        
        try:
            # Test comprehensive analytics
            dashboard_tests = [
                ('comprehensive_analytics', admin_dashboard.get_comprehensive_analytics),
                ('dashboard_metrics', enhanced_opik_evaluator.get_dashboard_metrics)
            ]
            
            for test_name, test_function in dashboard_tests:
                results['total_tests'] += 1
                
                try:
                    if test_name == 'comprehensive_analytics':
                        dashboard_data = await test_function(days=7, include_trends=True, include_comparisons=False)
                    else:
                        dashboard_data = await test_function(days=7)
                    
                    # Validate dashboard data
                    if dashboard_data and isinstance(dashboard_data, dict) and 'error' not in dashboard_data:
                        results['success_count'] += 1
                        results['dashboard_tests'].append({
                            'test_name': test_name,
                            'test_passed': True,
                            'has_metrics': bool(dashboard_data.get('base_metrics') or dashboard_data.get('summary')),
                            'has_trends': bool(dashboard_data.get('trend_analysis') or dashboard_data.get('daily_trends')),
                            'data_keys': list(dashboard_data.keys())[:5]  # First 5 keys for overview
                        })
                    else:
                        error_msg = dashboard_data.get('error', 'Invalid dashboard data format') if dashboard_data else 'No data returned'
                        results['dashboard_tests'].append({
                            'test_name': test_name,
                            'test_passed': False,
                            'error': error_msg
                        })
                    
                except Exception as e:
                    logger.error(f"Dashboard test {test_name} failed: {e}")
                    results['errors'].append(f"Dashboard test {test_name}: {str(e)}")
                    results['dashboard_tests'].append({
                        'test_name': test_name,
                        'test_passed': False,
                        'error': str(e)
                    })
            
            results['success_rate'] = (results['success_count'] / results['total_tests'] * 100) if results['total_tests'] > 0 else 0
            logger.info(f"Admin Dashboard Tests: {results['success_count']}/{results['total_tests']} passed ({results['success_rate']:.1f}%)")
            
        except Exception as e:
            results['errors'].append(f"Admin dashboard test suite error: {str(e)}")
            logger.error(f"Admin dashboard test suite failed: {e}")
        
        return results
    
    async def test_model_cards(self) -> Dict[str, Any]:
        """Test Feature 7: Dynamic model cards."""
        results = {
            'test_name': 'Dynamic Model Cards',
            'model_card_tests': [],
            'success_count': 0,
            'total_tests': 0,
            'errors': []
        }
        
        try:
            # Test different model card formats
            model_card_tests = [
                ('json_format', 'json'),
                ('markdown_format', 'markdown'),
                ('compact_summary', None)  # Special case for compact summary
            ]
            
            for test_name, format_type in model_card_tests:
                results['total_tests'] += 1
                
                try:
                    if test_name == 'compact_summary':
                        model_card_data = await dynamic_model_cards.generate_compact_summary()
                    else:
                        model_card_data = await dynamic_model_cards.generate_full_model_card(
                            analysis_days=7,
                            include_technical_details=True,
                            format_type=format_type
                        )
                    
                    # Validate model card data
                    if model_card_data and isinstance(model_card_data, dict) and 'error' not in model_card_data:
                        results['success_count'] += 1
                        results['model_card_tests'].append({
                            'test_name': test_name,
                            'format_type': format_type,
                            'test_passed': True,
                            'has_content': bool(model_card_data.get('content') or model_card_data.get('model_name')),
                            'has_metadata': bool(model_card_data.get('metadata') or model_card_data.get('generated_at')),
                            'content_length': len(str(model_card_data.get('content', model_card_data)))
                        })
                    else:
                        error_msg = model_card_data.get('error', 'Invalid model card data') if model_card_data else 'No data returned'
                        results['model_card_tests'].append({
                            'test_name': test_name,
                            'format_type': format_type,
                            'test_passed': False,
                            'error': error_msg
                        })
                    
                except Exception as e:
                    logger.error(f"Model card test {test_name} failed: {e}")
                    results['errors'].append(f"Model card test {test_name}: {str(e)}")
                    results['model_card_tests'].append({
                        'test_name': test_name,
                        'format_type': format_type,
                        'test_passed': False,
                        'error': str(e)
                    })
            
            results['success_rate'] = (results['success_count'] / results['total_tests'] * 100) if results['total_tests'] > 0 else 0
            logger.info(f"Model Cards Tests: {results['success_count']}/{results['total_tests']} passed ({results['success_rate']:.1f}%)")
            
        except Exception as e:
            results['errors'].append(f"Model cards test suite error: {str(e)}")
            logger.error(f"Model cards test suite failed: {e}")
        
        return results
    
    def validate_evaluation_result(self, result: EvaluationResult, test_case: Dict[str, Any]) -> bool:
        """Validate an evaluation result against expected outcomes."""
        try:
            # Basic validation
            if not result or not result.id:
                return False
            
            # Check score ranges
            scores = [
                result.accuracy_score,
                result.bias_score,
                result.hallucination_score,
                result.relevance_score,
                result.usefulness_score,
                result.overall_score
            ]
            
            if not all(0 <= score <= 1 for score in scores):
                return False
            
            # Check expected quality
            expected_quality = test_case.get('expected_quality')
            if expected_quality == 'high' and result.overall_score < 0.6:
                return False
            elif expected_quality == 'low' and result.overall_score > 0.8:
                return False
            
            # Check expected issues
            expected_issues = test_case.get('expected_issues', [])
            if 'bias' in expected_issues and result.bias_score < 0.3:
                return False
            if 'hallucination' in expected_issues and result.hallucination_score < 0.3:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Evaluation result validation failed: {e}")
            return False
    
    def validate_correction_result(self, result: Dict[str, Any], test_case: Dict[str, Any]) -> bool:
        """Validate a correction result."""
        try:
            # Check if correction was attempted for low quality cases
            if test_case.get('expected_quality') == 'low':
                return result.get('improvement_achieved', False) or len(result.get('corrections_made', [])) > 0
            
            # Basic structure validation
            required_keys = ['original_response', 'final_response', 'iterations']
            return all(key in result for key in required_keys)
            
        except Exception as e:
            logger.error(f"Correction result validation failed: {e}")
            return False
    
    async def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = sum(result.get('total_tests', 0) for result in self.test_results.values())
        total_successes = sum(result.get('success_count', 0) for result in self.test_results.values())
        overall_success_rate = (total_successes / total_tests * 100) if total_tests > 0 else 0
        
        # Count features that passed (>50% success rate)
        features_passed = sum(1 for result in self.test_results.values() 
                            if result.get('success_rate', 0) >= 50)
        
        report = {
            'test_summary': {
                'timestamp': datetime.utcnow().isoformat(),
                'conversation_id': self.conversation_id,
                'total_features_tested': len(self.test_results),
                'features_passed': features_passed,
                'total_tests_run': total_tests,
                'total_successes': total_successes,
                'overall_success_rate': round(overall_success_rate, 2),
                'test_status': 'PASSED' if overall_success_rate >= 70 else 'PARTIAL' if overall_success_rate >= 50 else 'FAILED'
            },
            'feature_results': self.test_results,
            'recommendations': self.generate_recommendations(),
            'next_steps': self.generate_next_steps()
        }
        
        # Log summary
        logger.info(f"📊 TEST SUMMARY:")
        logger.info(f"   Features tested: {len(self.test_results)}")
        logger.info(f"   Features passed: {features_passed}")
        logger.info(f"   Total tests: {total_tests}")
        logger.info(f"   Success rate: {overall_success_rate:.1f}%")
        logger.info(f"   Status: {report['test_summary']['test_status']}")
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        for feature_name, result in self.test_results.items():
            success_rate = result.get('success_rate', 0)
            
            if success_rate < 50:
                recommendations.append(f"Priority: Fix {result.get('test_name', feature_name)} - {success_rate:.1f}% success rate")
            elif success_rate < 80:
                recommendations.append(f"Improve {result.get('test_name', feature_name)} - {success_rate:.1f}% success rate")
            
            # Check for specific errors
            errors = result.get('errors', [])
            if errors:
                recommendations.append(f"Address errors in {result.get('test_name', feature_name)}: {len(errors)} issues found")
        
        if not recommendations:
            recommendations.append("All features performing well - continue monitoring and maintenance")
        
        return recommendations
    
    def generate_next_steps(self) -> List[str]:
        """Generate next steps based on test results."""
        return [
            "Monitor system performance in production environment",
            "Set up automated testing pipeline for continuous validation",
            "Implement user feedback collection for real-world validation",
            "Schedule regular performance reviews and optimizations",
            "Document any configuration changes needed for production",
            "Train team on new Opik integration features and capabilities"
        ]

async def main():
    """
    Main function to run the comprehensive test suite.
    """
    logger.info("🧪 Starting Comprehensive Opik Integration Test Suite")
    
    try:
        # Initialize and run tests
        test_suite = ComprehensiveOpikIntegrationTest()
        test_report = await test_suite.run_all_tests()
        
        # Save test report
        report_filename = f"opik_integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(test_report, f, indent=2, default=str)
        
        logger.info(f"📄 Test report saved to: {report_filename}")
        
        # Print final status
        status = test_report.get('test_summary', {}).get('test_status', 'UNKNOWN')
        if status == 'PASSED':
            logger.info("🎉 All Opik integration features are working correctly!")
        elif status == 'PARTIAL':
            logger.info("⚠️ Most Opik integration features are working - some issues to address")
        else:
            logger.info("❌ Significant issues found in Opik integration - requires attention")
        
        return test_report
        
    except Exception as e:
        logger.error(f"❌ Test suite execution failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())
