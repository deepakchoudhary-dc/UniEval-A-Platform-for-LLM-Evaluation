"""
Real-time UI Integration for Opik Evaluation Display

This module provides real-time UI components and data formatting for displaying
Opik evaluation results directly in the chat interface.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import asdict

from .enhanced_opik_evaluator import enhanced_opik_evaluator, EvaluationResult
from .self_correction import self_correction_engine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeUIManager:
    """
    Manages real-time UI updates for Opik evaluation results.
    """
    
    def __init__(self):
        """Initialize the real-time UI manager."""
        self.evaluator = enhanced_opik_evaluator
        self.correction_engine = self_correction_engine
        
        # UI component templates
        self.ui_templates = self._load_ui_templates()
        
        logger.info("Real-time UI manager initialized")
    
    def _load_ui_templates(self) -> Dict[str, str]:
        """Load UI component templates."""
        return {
            'evaluation_badge': """
                <div class="evaluation-badge {confidence_class}">
                    <span class="badge-label">Quality: {confidence_label}</span>
                    <span class="badge-score">{overall_score}%</span>
                </div>
            """,
            
            'metrics_panel': """
                <div class="metrics-panel">
                    <div class="metric-item">
                        <span class="metric-label">Accuracy</span>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: {accuracy_percent}%; background-color: {accuracy_color};"></div>
                        </div>
                        <span class="metric-value">{accuracy_score}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Bias Level</span>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: {bias_percent}%; background-color: {bias_color};"></div>
                        </div>
                        <span class="metric-value">{bias_score}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Hallucination Risk</span>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: {hallucination_percent}%; background-color: {hallucination_color};"></div>
                        </div>
                        <span class="metric-value">{hallucination_score}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Relevance</span>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: {relevance_percent}%; background-color: {relevance_color};"></div>
                        </div>
                        <span class="metric-value">{relevance_score}</span>
                    </div>
                    <div class="metric-item">
                        <span class="metric-label">Usefulness</span>
                        <div class="metric-bar">
                            <div class="metric-fill" style="width: {usefulness_percent}%; background-color: {usefulness_color};"></div>
                        </div>
                        <span class="metric-value">{usefulness_score}</span>
                    </div>
                </div>
            """,
            
            'correction_notice': """
                <div class="correction-notice {notice_type}">
                    <div class="notice-header">
                        <span class="notice-icon">{icon}</span>
                        <span class="notice-title">{title}</span>
                    </div>
                    <div class="notice-content">
                        <p>{message}</p>
                        {improvements_list}
                        {correction_button}
                    </div>
                </div>
            """,
            
            'improvement_item': """
                <li class="improvement-item {priority_class}">
                    <span class="improvement-type">{improvement_type}</span>
                    <span class="improvement-message">{message}</span>
                </li>
            """,
            
            'correction_button': """
                <button class="correction-button" onclick="requestCorrection('{evaluation_id}')">
                    Improve Response
                </button>
            """,
            
            'live_feedback': """
                <div class="live-feedback-container">
                    <div class="feedback-header">
                        <h4>Response Quality Assessment</h4>
                        <span class="timestamp">{timestamp}</span>
                    </div>
                    {evaluation_badge}
                    {metrics_panel}
                    {correction_notice}
                </div>
            """,
            
            'mini_indicator': """
                <div class="mini-evaluation-indicator {quality_class}" title="Overall Quality: {confidence_label}">
                    <div class="indicator-dot"></div>
                    <span class="indicator-score">{score_display}</span>
                </div>
            """
        }
    
    async def generate_realtime_ui_data(
        self,
        evaluation_result: EvaluationResult,
        include_correction_options: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive UI data for real-time display.
        """
        try:
            # Get UI-formatted data from evaluator
            ui_data = self.evaluator.get_realtime_ui_data(evaluation_result)
            
            # Generate HTML components
            html_components = await self._generate_html_components(evaluation_result, ui_data)
            
            # Generate JavaScript data for interactive features
            js_data = self._generate_javascript_data(evaluation_result, ui_data)
            
            # Generate CSS classes for styling
            css_classes = self._generate_css_classes(evaluation_result, ui_data)
            
            # Generate correction options if needed
            correction_data = {}
            if include_correction_options and evaluation_result.requires_correction:
                correction_data = await self.correction_engine.assess_correction_need(evaluation_result)
            
            return {
                'evaluation_id': evaluation_result.id,
                'timestamp': evaluation_result.timestamp.isoformat(),
                'ui_data': ui_data,
                'html_components': html_components,
                'javascript_data': js_data,
                'css_classes': css_classes,
                'correction_data': correction_data,
                'display_config': {
                    'show_detailed_metrics': True,
                    'show_correction_notice': evaluation_result.requires_correction,
                    'animation_enabled': True,
                    'auto_hide_after': 30000 if evaluation_result.confidence_level == 'high' else None
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate real-time UI data: {e}")
            return self._generate_fallback_ui_data(evaluation_result)
    
    async def _generate_html_components(
        self,
        evaluation_result: EvaluationResult,
        ui_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate HTML components for the evaluation display."""
        
        # Generate evaluation badge
        evaluation_badge = self.ui_templates['evaluation_badge'].format(
            confidence_class=f"confidence-{evaluation_result.confidence_level}",
            confidence_label=evaluation_result.confidence_level.title(),
            overall_score=round(evaluation_result.overall_score * 100)
        )
        
        # Generate metrics panel
        metrics_panel = self.ui_templates['metrics_panel'].format(
            accuracy_percent=round(evaluation_result.accuracy_score * 100),
            accuracy_color=ui_data['scores']['accuracy']['color'],
            accuracy_score=ui_data['scores']['accuracy']['label'],
            
            bias_percent=round(evaluation_result.bias_score * 100),
            bias_color=ui_data['scores']['bias']['color'],
            bias_score=ui_data['scores']['bias']['label'],
            
            hallucination_percent=round(evaluation_result.hallucination_score * 100),
            hallucination_color=ui_data['scores']['hallucination']['color'],
            hallucination_score=ui_data['scores']['hallucination']['label'],
            
            relevance_percent=round(evaluation_result.relevance_score * 100),
            relevance_color=ui_data['scores']['relevance']['color'],
            relevance_score=ui_data['scores']['relevance']['label'],
            
            usefulness_percent=round(evaluation_result.usefulness_score * 100),
            usefulness_color=ui_data['scores']['usefulness']['color'],
            usefulness_score=ui_data['scores']['usefulness']['label']
        )
        
        # Generate correction notice if needed
        correction_notice = ""
        if evaluation_result.requires_correction:
            correction_data = await self.correction_engine.assess_correction_need(evaluation_result)
            
            improvements_list = ""
            if correction_data['suggested_improvements']:
                improvements_html = []
                for improvement in correction_data['suggested_improvements']:
                    improvement_html = self.ui_templates['improvement_item'].format(
                        priority_class=f"priority-{improvement['priority']}",
                        improvement_type=improvement['type'].replace('_', ' ').title(),
                        message=improvement['message']
                    )
                    improvements_html.append(improvement_html)
                improvements_list = f"<ul class='improvements-list'>{''.join(improvements_html)}</ul>"
            
            correction_button = self.ui_templates['correction_button'].format(
                evaluation_id=evaluation_result.id
            )
            
            correction_notice = self.ui_templates['correction_notice'].format(
                notice_type="warning" if evaluation_result.confidence_level == "low" else "info",
                icon="⚠️" if evaluation_result.confidence_level == "low" else "ℹ️",
                title="Response Could Be Improved",
                message=evaluation_result.correction_reason or "Several aspects of this response could be enhanced.",
                improvements_list=improvements_list,
                correction_button=correction_button
            )
        
        # Generate mini indicator for compact view
        mini_indicator = self.ui_templates['mini_indicator'].format(
            quality_class=f"quality-{ui_data['ui_elements']['quality_indicator']}",
            confidence_label=evaluation_result.confidence_level.title(),
            score_display=f"{round(evaluation_result.overall_score * 100)}%"
        )
        
        # Generate complete live feedback component
        live_feedback = self.ui_templates['live_feedback'].format(
            timestamp=evaluation_result.timestamp.strftime("%H:%M:%S"),
            evaluation_badge=evaluation_badge,
            metrics_panel=metrics_panel,
            correction_notice=correction_notice
        )
        
        return {
            'evaluation_badge': evaluation_badge,
            'metrics_panel': metrics_panel,
            'correction_notice': correction_notice,
            'mini_indicator': mini_indicator,
            'live_feedback': live_feedback
        }
    
    def _generate_javascript_data(
        self,
        evaluation_result: EvaluationResult,
        ui_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate JavaScript data for interactive features."""
        return {
            'evaluationId': evaluation_result.id,
            'scores': {
                'accuracy': evaluation_result.accuracy_score,
                'bias': evaluation_result.bias_score,
                'hallucination': evaluation_result.hallucination_score,
                'relevance': evaluation_result.relevance_score,
                'usefulness': evaluation_result.usefulness_score,
                'overall': evaluation_result.overall_score
            },
            'metadata': {
                'confidence': evaluation_result.confidence_level,
                'requiresCorrection': evaluation_result.requires_correction,
                'correctionReason': evaluation_result.correction_reason,
                'timestamp': evaluation_result.timestamp.isoformat()
            },
            'interactions': {
                'correctionEnabled': evaluation_result.requires_correction,
                'detailsExpandable': True,
                'feedbackEnabled': True
            },
            'animations': {
                'scoreCountUp': True,
                'progressBars': True,
                'slideIn': True,
                'fadeOut': evaluation_result.confidence_level == 'high'
            }
        }
    
    def _generate_css_classes(
        self,
        evaluation_result: EvaluationResult,
        ui_data: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate CSS classes for styling."""
        return {
            'container': f"evaluation-container confidence-{evaluation_result.confidence_level}",
            'badge': f"evaluation-badge {ui_data['ui_elements']['confidence_badge']}",
            'quality_indicator': f"quality-indicator {ui_data['ui_elements']['quality_indicator']}",
            'correction_notice': f"correction-notice {'visible' if evaluation_result.requires_correction else 'hidden'}",
            'metrics_panel': "metrics-panel expanded",
            'overall_status': f"status-{evaluation_result.confidence_level}"
        }
    
    def _generate_fallback_ui_data(self, evaluation_result: EvaluationResult) -> Dict[str, Any]:
        """Generate minimal UI data when full generation fails."""
        return {
            'evaluation_id': evaluation_result.id,
            'timestamp': evaluation_result.timestamp.isoformat(),
            'fallback': True,
            'html_components': {
                'mini_indicator': f'<div class="evaluation-fallback">Quality: {evaluation_result.confidence_level}</div>'
            },
            'javascript_data': {
                'evaluationId': evaluation_result.id,
                'fallback': True,
                'scores': {'overall': evaluation_result.overall_score}
            },
            'css_classes': {
                'container': 'evaluation-container fallback'
            }
        }
    
    async def generate_conversation_summary_ui(self, conversation_id: str) -> Dict[str, Any]:
        """Generate UI data for conversation-level summary."""
        try:
            # Get conversation analytics
            analytics = await self.evaluator.get_conversation_analytics(conversation_id)
            
            if not analytics:
                return {'error': 'No data available for this conversation'}
            
            # Generate summary HTML
            summary_html = f"""
                <div class="conversation-summary">
                    <div class="summary-header">
                        <h3>Conversation Quality Summary</h3>
                        <span class="conversation-id">ID: {conversation_id}</span>
                    </div>
                    <div class="summary-stats">
                        <div class="stat-item">
                            <span class="stat-label">Total Exchanges</span>
                            <span class="stat-value">{analytics.total_exchanges}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Overall Quality</span>
                            <span class="stat-value">{round(analytics.overall_quality * 100)}%</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Corrections Made</span>
                            <span class="stat-value">{analytics.correction_count}</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Quality Trend</span>
                            <span class="stat-value trend-{analytics.quality_trend}">{analytics.quality_trend.title()}</span>
                        </div>
                    </div>
                    <div class="summary-metrics">
                        <div class="metric-row">
                            <span class="metric-name">Average Accuracy</span>
                            <div class="metric-bar">
                                <div class="metric-fill" style="width: {round(analytics.avg_accuracy * 100)}%; background-color: #28a745;"></div>
                            </div>
                            <span class="metric-percent">{round(analytics.avg_accuracy * 100)}%</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-name">Average Bias</span>
                            <div class="metric-bar">
                                <div class="metric-fill" style="width: {round(analytics.avg_bias * 100)}%; background-color: #dc3545;"></div>
                            </div>
                            <span class="metric-percent">{round(analytics.avg_bias * 100)}%</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-name">Hallucination Risk</span>
                            <div class="metric-bar">
                                <div class="metric-fill" style="width: {round(analytics.avg_hallucination * 100)}%; background-color: #dc3545;"></div>
                            </div>
                            <span class="metric-percent">{round(analytics.avg_hallucination * 100)}%</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-name">Average Relevance</span>
                            <div class="metric-bar">
                                <div class="metric-fill" style="width: {round(analytics.avg_relevance * 100)}%; background-color: #28a745;"></div>
                            </div>
                            <span class="metric-percent">{round(analytics.avg_relevance * 100)}%</span>
                        </div>
                        <div class="metric-row">
                            <span class="metric-name">Average Usefulness</span>
                            <div class="metric-bar">
                                <div class="metric-fill" style="width: {round(analytics.avg_usefulness * 100)}%; background-color: #28a745;"></div>
                            </div>
                            <span class="metric-percent">{round(analytics.avg_usefulness * 100)}%</span>
                        </div>
                    </div>
                </div>
            """
            
            return {
                'conversation_id': conversation_id,
                'analytics': asdict(analytics),
                'html_summary': summary_html,
                'javascript_data': {
                    'conversationId': conversation_id,
                    'analytics': asdict(analytics),
                    'chartData': {
                        'accuracy': analytics.avg_accuracy,
                        'bias': analytics.avg_bias,
                        'hallucination': analytics.avg_hallucination,
                        'relevance': analytics.avg_relevance,
                        'usefulness': analytics.avg_usefulness
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate conversation summary UI: {e}")
            return {'error': str(e)}
    
    def generate_css_styles(self) -> str:
        """Generate CSS styles for evaluation UI components."""
        return """
            .evaluation-container {
                background: #f8f9fa;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            }
            
            .evaluation-badge {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 4px 12px;
                border-radius: 16px;
                font-size: 14px;
                font-weight: 500;
            }
            
            .confidence-high { background-color: #d4edda; color: #155724; }
            .confidence-medium { background-color: #fff3cd; color: #856404; }
            .confidence-low { background-color: #f8d7da; color: #721c24; }
            
            .metrics-panel {
                margin-top: 15px;
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            .metric-item {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .metric-label {
                min-width: 120px;
                font-size: 13px;
                color: #6c757d;
            }
            
            .metric-bar {
                flex: 1;
                height: 8px;
                background-color: #e9ecef;
                border-radius: 4px;
                overflow: hidden;
            }
            
            .metric-fill {
                height: 100%;
                transition: width 0.3s ease;
            }
            
            .metric-value {
                min-width: 60px;
                text-align: right;
                font-size: 13px;
                font-weight: 500;
            }
            
            .correction-notice {
                margin-top: 15px;
                padding: 12px;
                border-radius: 6px;
                border-left: 4px solid;
            }
            
            .correction-notice.warning {
                background-color: #fff3cd;
                border-color: #ffc107;
                color: #856404;
            }
            
            .correction-notice.info {
                background-color: #d1ecf1;
                border-color: #17a2b8;
                color: #0c5460;
            }
            
            .notice-header {
                display: flex;
                align-items: center;
                gap: 8px;
                font-weight: 600;
                margin-bottom: 8px;
            }
            
            .improvements-list {
                margin: 10px 0;
                padding-left: 20px;
            }
            
            .improvement-item {
                margin-bottom: 5px;
                font-size: 14px;
            }
            
            .priority-high { color: #dc3545; }
            .priority-medium { color: #fd7e14; }
            .priority-low { color: #6c757d; }
            
            .correction-button {
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                margin-top: 10px;
            }
            
            .correction-button:hover {
                background-color: #0056b3;
            }
            
            .mini-evaluation-indicator {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                padding: 2px 8px;
                border-radius: 12px;
                font-size: 12px;
            }
            
            .quality-high_quality { background-color: #d4edda; color: #155724; }
            .quality-medium_quality { background-color: #fff3cd; color: #856404; }
            .quality-low_quality { background-color: #f8d7da; color: #721c24; }
            
            .indicator-dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background-color: currentColor;
            }
            
            .conversation-summary {
                background: white;
                border: 1px solid #dee2e6;
                border-radius: 8px;
                padding: 20px;
                margin: 15px 0;
            }
            
            .summary-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 10px;
                border-bottom: 1px solid #dee2e6;
            }
            
            .summary-stats {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }
            
            .stat-item {
                text-align: center;
                padding: 10px;
                background-color: #f8f9fa;
                border-radius: 6px;
            }
            
            .stat-label {
                display: block;
                font-size: 12px;
                color: #6c757d;
                margin-bottom: 5px;
            }
            
            .stat-value {
                display: block;
                font-size: 18px;
                font-weight: 600;
                color: #212529;
            }
            
            .trend-improving { color: #28a745; }
            .trend-declining { color: #dc3545; }
            .trend-stable { color: #6c757d; }
            
            .summary-metrics {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            
            .metric-row {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .metric-name {
                min-width: 140px;
                font-size: 14px;
            }
            
            .metric-percent {
                min-width: 50px;
                text-align: right;
                font-size: 14px;
                font-weight: 500;
            }
        """
    
    def generate_javascript_functions(self) -> str:
        """Generate JavaScript functions for interactive features."""
        return """
            function requestCorrection(evaluationId) {
                // Send correction request to backend
                fetch('/api/evaluation/request-correction', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ evaluation_id: evaluationId })
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showCorrectionInProgress(evaluationId);
                    } else {
                        showCorrectionError(data.error);
                    }
                })
                .catch(error => {
                    console.error('Correction request failed:', error);
                    showCorrectionError('Failed to request correction');
                });
            }
            
            function showCorrectionInProgress(evaluationId) {
                const button = document.querySelector(`button[onclick="requestCorrection('${evaluationId}')"]`);
                if (button) {
                    button.disabled = true;
                    button.textContent = 'Improving...';
                    button.style.backgroundColor = '#6c757d';
                }
            }
            
            function showCorrectionError(error) {
                alert('Correction failed: ' + error);
            }
            
            function toggleMetricsDetails(evaluationId) {
                const panel = document.querySelector(`.metrics-panel[data-evaluation-id="${evaluationId}"]`);
                if (panel) {
                    panel.classList.toggle('collapsed');
                }
            }
            
            function animateScoreCountUp(element, finalValue, duration = 1000) {
                const startValue = 0;
                const startTime = Date.now();
                
                function updateValue() {
                    const elapsed = Date.now() - startTime;
                    const progress = Math.min(elapsed / duration, 1);
                    const currentValue = Math.round(startValue + (finalValue - startValue) * progress);
                    
                    element.textContent = currentValue + '%';
                    
                    if (progress < 1) {
                        requestAnimationFrame(updateValue);
                    }
                }
                
                requestAnimationFrame(updateValue);
            }
            
            function initializeEvaluationUI(data) {
                // Initialize score animations
                if (data.animations.scoreCountUp) {
                    const scoreElements = document.querySelectorAll('.metric-value');
                    scoreElements.forEach(element => {
                        const score = parseFloat(element.dataset.score);
                        if (!isNaN(score)) {
                            animateScoreCountUp(element, Math.round(score * 100));
                        }
                    });
                }
                
                // Initialize progress bar animations
                if (data.animations.progressBars) {
                    const progressBars = document.querySelectorAll('.metric-fill');
                    progressBars.forEach(bar => {
                        const width = bar.style.width;
                        bar.style.width = '0%';
                        setTimeout(() => {
                            bar.style.width = width;
                        }, 100);
                    });
                }
                
                // Auto-hide after delay if configured
                if (data.display_config.auto_hide_after) {
                    setTimeout(() => {
                        const container = document.querySelector('.evaluation-container');
                        if (container) {
                            container.style.opacity = '0.5';
                        }
                    }, data.display_config.auto_hide_after);
                }
            }
        """

# Global instance
realtime_ui_manager = RealTimeUIManager()
