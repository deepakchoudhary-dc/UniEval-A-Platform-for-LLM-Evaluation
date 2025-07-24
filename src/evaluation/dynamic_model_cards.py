"""
Dynamic Model Cards with Live Opik Performance Statistics

This module generates dynamic model cards that are automatically updated with
live performance statistics from Opik evaluations.
"""

import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from .enhanced_opik_evaluator import enhanced_opik_evaluator
from .admin_dashboard import admin_dashboard

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelCardData:
    """Data structure for model card information."""
    model_name: str
    model_version: str
    last_updated: datetime
    performance_summary: Dict[str, Any]
    bias_analysis: Dict[str, Any]
    safety_metrics: Dict[str, Any]
    usage_statistics: Dict[str, Any]
    limitations: List[str]
    recommendations: List[str]

class DynamicModelCards:
    """
    Dynamic model cards that automatically update with live performance data.
    """
    
    def __init__(self, model_name: str = "Enterprise AI Chatbot", model_version: str = "2.0"):
        """Initialize dynamic model cards."""
        self.model_name = model_name
        self.model_version = model_version
        self.evaluator = enhanced_opik_evaluator
        self.dashboard = admin_dashboard
        
        # Model card templates
        self.templates = self._load_templates()
        
        logger.info(f"Dynamic model cards initialized for {model_name} v{model_version}")
    
    def _load_templates(self) -> Dict[str, str]:
        """Load model card templates."""
        return {
            'full_card': """
                # {model_name} - Model Card
                
                **Version:** {model_version}  
                **Last Updated:** {last_updated}  
                **Generated:** {generated_at}
                
                ## Model Overview
                
                {model_description}
                
                ## Performance Summary
                
                ### Overall Quality: {overall_quality_badge}
                
                | Metric | Score | Status | Trend |
                |--------|-------|--------|-------|
                | Accuracy | {accuracy_score}% | {accuracy_status} | {accuracy_trend} |
                | Bias Level | {bias_score}% | {bias_status} | {bias_trend} |
                | Hallucination Risk | {hallucination_score}% | {hallucination_status} | {hallucination_trend} |
                | Relevance | {relevance_score}% | {relevance_status} | {relevance_trend} |
                | Usefulness | {usefulness_score}% | {usefulness_status} | {usefulness_trend} |
                
                ## Safety & Bias Analysis
                
                ### Bias Detection
                {bias_analysis}
                
                ### Safety Metrics
                {safety_metrics}
                
                ## Usage Statistics
                
                {usage_statistics}
                
                ## Known Limitations
                
                {limitations}
                
                ## Recommendations
                
                {recommendations}
                
                ## Technical Details
                
                {technical_details}
                
                ---
                *This model card is automatically generated and updated with live performance data.*
            """,
            
            'compact_card': """
                ## {model_name} v{model_version}
                
                **Quality:** {overall_quality} | **Bias:** {bias_level} | **Safety:** {safety_level}
                
                {performance_summary}
                
                *Updated: {last_updated}*
            """,
            
            'dashboard_widget': """
                <div class="model-card-widget">
                    <div class="card-header">
                        <h3>{model_name}</h3>
                        <span class="version">v{model_version}</span>
                        <span class="quality-badge {quality_class}">{overall_quality}</span>
                    </div>
                    <div class="card-metrics">
                        {metrics_grid}
                    </div>
                    <div class="card-footer">
                        <span class="last-updated">Updated: {last_updated}</span>
                        <button onclick="refreshModelCard()">Refresh</button>
                    </div>
                </div>
            """,
            
            'api_response': """
                {
                    "model_name": "{model_name}",
                    "model_version": "{model_version}",
                    "last_updated": "{last_updated}",
                    "performance_summary": {performance_summary},
                    "status": "{status}",
                    "recommendations": {recommendations}
                }
            """
        }
    
    async def generate_full_model_card(
        self,
        analysis_days: int = 30,
        include_technical_details: bool = True,
        format_type: str = "markdown"
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive model card with live performance data.
        
        Args:
            analysis_days: Number of days to analyze for performance metrics
            include_technical_details: Whether to include technical implementation details
            format_type: Output format ('markdown', 'html', 'json')
        """
        try:
            # Get comprehensive performance data
            performance_stats = await self.evaluator.get_model_performance_stats()
            dashboard_metrics = await self.dashboard.get_comprehensive_analytics(analysis_days)
            
            # Create model card data
            model_card_data = await self._compile_model_card_data(
                performance_stats, dashboard_metrics, analysis_days
            )
            
            # Generate formatted output
            if format_type == "markdown":
                return await self._generate_markdown_card(model_card_data, include_technical_details)
            elif format_type == "html":
                return await self._generate_html_card(model_card_data, include_technical_details)
            elif format_type == "json":
                return await self._generate_json_card(model_card_data, include_technical_details)
            else:
                raise ValueError(f"Unsupported format type: {format_type}")
                
        except Exception as e:
            logger.error(f"Failed to generate full model card: {e}")
            return {'error': str(e)}
    
    async def _compile_model_card_data(
        self,
        performance_stats: Dict[str, Any],
        dashboard_metrics: Dict[str, Any],
        analysis_days: int
    ) -> ModelCardData:
        """Compile all data needed for model card generation."""
        
        # Extract performance metrics
        overall_metrics = performance_stats.get('performance_metrics', {}).get('overall', {})
        recent_performance = performance_stats.get('recent_performance', {})
        quality_assessment = performance_stats.get('quality_assessment', {})
        
        # Extract dashboard data
        base_metrics = dashboard_metrics.get('base_metrics', {})
        insights = dashboard_metrics.get('insights_and_recommendations', {})
        
        # Compile performance summary
        performance_summary = {
            'overall_score': overall_metrics.get('composite_score', 0) * 100,
            'accuracy_score': overall_metrics.get('accuracy', 0) * 100,
            'bias_score': overall_metrics.get('bias', 0) * 100,
            'hallucination_score': overall_metrics.get('hallucination', 0) * 100,
            'relevance_score': overall_metrics.get('relevance', 0) * 100,
            'usefulness_score': overall_metrics.get('usefulness', 0) * 100,
            'confidence_distribution': performance_stats.get('confidence_distribution', {}),
            'total_evaluations': performance_stats.get('model_info', {}).get('evaluation_period', {}).get('total_evaluations', 0),
            'correction_rate': recent_performance.get('correction_rate', 0)
        }
        
        # Compile bias analysis
        bias_analysis = {
            'overall_bias_score': overall_metrics.get('bias', 0),
            'bias_level': self._categorize_bias_level(overall_metrics.get('bias', 0)),
            'bias_trends': dashboard_metrics.get('trend_analysis', {}).get('daily_trends', []),
            'bias_insights': [finding for finding in insights.get('key_findings', []) if 'bias' in finding.lower()]
        }
        
        # Compile safety metrics
        safety_metrics = {
            'hallucination_score': overall_metrics.get('hallucination', 0),
            'safety_level': self._categorize_safety_level(overall_metrics.get('hallucination', 0)),
            'safety_alerts': [alert for alert in insights.get('alerts', []) if 'hallucination' in alert.lower() or 'safety' in alert.lower()],
            'moderation_effectiveness': 1 - overall_metrics.get('bias', 0)  # Inverse of bias as a proxy
        }
        
        # Compile usage statistics
        usage_statistics = {
            'total_evaluations': performance_summary['total_evaluations'],
            'evaluation_period': analysis_days,
            'average_daily_usage': performance_summary['total_evaluations'] / analysis_days if analysis_days > 0 else 0,
            'peak_quality_period': self._identify_peak_quality_period(dashboard_metrics.get('trend_analysis', {})),
            'quality_trend': dashboard_metrics.get('trend_analysis', {}).get('trend_analysis', {}).get('overall_trend', 'stable')
        }
        
        # Generate limitations based on performance data
        limitations = self._generate_limitations(performance_summary, bias_analysis, safety_metrics)
        
        # Generate recommendations based on insights
        recommendations = insights.get('recommendations', [])
        
        return ModelCardData(
            model_name=self.model_name,
            model_version=self.model_version,
            last_updated=datetime.utcnow(),
            performance_summary=performance_summary,
            bias_analysis=bias_analysis,
            safety_metrics=safety_metrics,
            usage_statistics=usage_statistics,
            limitations=limitations,
            recommendations=recommendations
        )
    
    def _categorize_bias_level(self, bias_score: float) -> str:
        """Categorize bias level based on score."""
        if bias_score <= 0.2:
            return "Low"
        elif bias_score <= 0.5:
            return "Moderate"
        else:
            return "High"
    
    def _categorize_safety_level(self, hallucination_score: float) -> str:
        """Categorize safety level based on hallucination score."""
        if hallucination_score <= 0.2:
            return "High Safety"
        elif hallucination_score <= 0.5:
            return "Moderate Safety"
        else:
            return "Safety Concerns"
    
    def _identify_peak_quality_period(self, trend_analysis: Dict[str, Any]) -> str:
        """Identify the period with peak quality performance."""
        daily_trends = trend_analysis.get('daily_trends', [])
        if not daily_trends:
            return "Insufficient data"
        
        best_day = max(daily_trends, key=lambda x: x.get('avg_overall', 0))
        return best_day.get('date', 'Unknown')
    
    def _generate_limitations(
        self,
        performance_summary: Dict[str, Any],
        bias_analysis: Dict[str, Any],
        safety_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate limitations based on performance data."""
        limitations = []
        
        # Quality-based limitations
        if performance_summary['overall_score'] < 80:
            limitations.append("Overall performance below optimal threshold (80%)")
        
        # Bias-based limitations
        if bias_analysis['bias_level'] in ['Moderate', 'High']:
            limitations.append(f"Detected {bias_analysis['bias_level'].lower()} bias levels in responses")
        
        # Safety-based limitations
        if safety_metrics['safety_level'] == 'Safety Concerns':
            limitations.append("Elevated hallucination risk requiring additional verification")
        
        # Usage-based limitations
        if performance_summary['correction_rate'] > 20:
            limitations.append("High correction rate indicates potential quality consistency issues")
        
        # Add general limitations
        limitations.extend([
            "Performance may vary based on input complexity and domain specificity",
            "Continuous monitoring and evaluation required for optimal performance",
            "Designed for general conversational AI tasks, may not be suitable for specialized domains"
        ])
        
        return limitations
    
    async def _generate_markdown_card(
        self,
        data: ModelCardData,
        include_technical: bool
    ) -> Dict[str, Any]:
        """Generate markdown format model card."""
        
        # Format performance metrics
        metrics_table = self._format_performance_table(data.performance_summary)
        
        # Format bias analysis
        bias_section = self._format_bias_analysis(data.bias_analysis)
        
        # Format safety metrics
        safety_section = self._format_safety_metrics(data.safety_metrics)
        
        # Format usage statistics
        usage_section = self._format_usage_statistics(data.usage_statistics)
        
        # Format limitations and recommendations
        limitations_list = '\n'.join([f"- {limitation}" for limitation in data.limitations])
        recommendations_list = '\n'.join([f"- {rec}" for rec in data.recommendations])
        
        # Generate technical details if requested
        technical_section = ""
        if include_technical:
            technical_section = await self._generate_technical_details()
        
        # Fill template
        markdown_content = self.templates['full_card'].format(
            model_name=data.model_name,
            model_version=data.model_version,
            last_updated=data.last_updated.strftime("%Y-%m-%d %H:%M:%S UTC"),
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            model_description=self._get_model_description(),
            overall_quality_badge=self._generate_quality_badge(data.performance_summary['overall_score']),
            accuracy_score=round(data.performance_summary['accuracy_score'], 1),
            accuracy_status=self._get_metric_status(data.performance_summary['accuracy_score']),
            accuracy_trend="→",  # Would need trend calculation
            bias_score=round(data.performance_summary['bias_score'], 1),
            bias_status=self._get_bias_status(data.performance_summary['bias_score']),
            bias_trend="→",
            hallucination_score=round(data.performance_summary['hallucination_score'], 1),
            hallucination_status=self._get_safety_status(data.performance_summary['hallucination_score']),
            hallucination_trend="→",
            relevance_score=round(data.performance_summary['relevance_score'], 1),
            relevance_status=self._get_metric_status(data.performance_summary['relevance_score']),
            relevance_trend="→",
            usefulness_score=round(data.performance_summary['usefulness_score'], 1),
            usefulness_status=self._get_metric_status(data.performance_summary['usefulness_score']),
            usefulness_trend="→",
            bias_analysis=bias_section,
            safety_metrics=safety_section,
            usage_statistics=usage_section,
            limitations=limitations_list,
            recommendations=recommendations_list,
            technical_details=technical_section
        )
        
        return {
            'format': 'markdown',
            'content': markdown_content,
            'metadata': asdict(data),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def _generate_html_card(
        self,
        data: ModelCardData,
        include_technical: bool
    ) -> Dict[str, Any]:
        """Generate HTML format model card."""
        
        # Generate metrics grid for dashboard widget
        metrics_grid = self._generate_metrics_grid_html(data.performance_summary)
        
        # Generate full HTML card
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{data.model_name} Model Card</title>
            <style>
                {self._get_model_card_css()}
            </style>
        </head>
        <body>
            <div class="model-card">
                <header class="card-header">
                    <h1>{data.model_name}</h1>
                    <div class="version-info">
                        <span class="version">Version {data.model_version}</span>
                        <span class="updated">Updated: {data.last_updated.strftime('%Y-%m-%d %H:%M UTC')}</span>
                    </div>
                    <div class="quality-overview">
                        {self._generate_quality_badge_html(data.performance_summary['overall_score'])}
                    </div>
                </header>
                
                <section class="performance-section">
                    <h2>Performance Metrics</h2>
                    {self._generate_performance_chart_html(data.performance_summary)}
                </section>
                
                <section class="bias-section">
                    <h2>Bias Analysis</h2>
                    {self._generate_bias_analysis_html(data.bias_analysis)}
                </section>
                
                <section class="safety-section">
                    <h2>Safety Metrics</h2>
                    {self._generate_safety_metrics_html(data.safety_metrics)}
                </section>
                
                <section class="usage-section">
                    <h2>Usage Statistics</h2>
                    {self._generate_usage_stats_html(data.usage_statistics)}
                </section>
                
                <section class="limitations-section">
                    <h2>Limitations</h2>
                    <ul>
                        {''.join([f'<li>{limitation}</li>' for limitation in data.limitations])}
                    </ul>
                </section>
                
                <section class="recommendations-section">
                    <h2>Recommendations</h2>
                    <ul>
                        {''.join([f'<li>{rec}</li>' for rec in data.recommendations])}
                    </ul>
                </section>
                
                {self._generate_technical_section_html() if include_technical else ''}
                
                <footer class="card-footer">
                    <p>This model card is automatically generated and updated with live performance data.</p>
                    <p>Generated at: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
                </footer>
            </div>
        </body>
        </html>
        """
        
        return {
            'format': 'html',
            'content': html_content,
            'widget': self.templates['dashboard_widget'].format(
                model_name=data.model_name,
                model_version=data.model_version,
                overall_quality=f"{round(data.performance_summary['overall_score'], 1)}%",
                quality_class=self._get_quality_css_class(data.performance_summary['overall_score']),
                metrics_grid=metrics_grid,
                last_updated=data.last_updated.strftime('%Y-%m-%d %H:%M')
            ),
            'metadata': asdict(data),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    async def _generate_json_card(
        self,
        data: ModelCardData,
        include_technical: bool
    ) -> Dict[str, Any]:
        """Generate JSON format model card."""
        
        json_data = {
            'model_card': {
                'model_name': data.model_name,
                'model_version': data.model_version,
                'last_updated': data.last_updated.isoformat(),
                'generated_at': datetime.utcnow().isoformat(),
                'performance_summary': data.performance_summary,
                'bias_analysis': data.bias_analysis,
                'safety_metrics': data.safety_metrics,
                'usage_statistics': data.usage_statistics,
                'limitations': data.limitations,
                'recommendations': data.recommendations,
                'quality_assessment': {
                    'overall_status': self._get_overall_status(data.performance_summary['overall_score']),
                    'bias_level': data.bias_analysis['bias_level'],
                    'safety_level': data.safety_metrics['safety_level'],
                    'recommendation_priority': self._assess_recommendation_priority(data)
                }
            }
        }
        
        if include_technical:
            json_data['model_card']['technical_details'] = await self._get_technical_details_dict()
        
        return {
            'format': 'json',
            'content': json.dumps(json_data, indent=2),
            'data': json_data,
            'metadata': asdict(data),
            'generated_at': datetime.utcnow().isoformat()
        }
    
    def _format_performance_table(self, performance_data: Dict[str, Any]) -> str:
        """Format performance metrics as a table."""
        return f"""
        | Metric | Score | Status |
        |--------|-------|--------|
        | Overall Quality | {performance_data['overall_score']:.1f}% | {self._get_metric_status(performance_data['overall_score'])} |
        | Accuracy | {performance_data['accuracy_score']:.1f}% | {self._get_metric_status(performance_data['accuracy_score'])} |
        | Bias Level | {performance_data['bias_score']:.1f}% | {self._get_bias_status(performance_data['bias_score'])} |
        | Hallucination Risk | {performance_data['hallucination_score']:.1f}% | {self._get_safety_status(performance_data['hallucination_score'])} |
        | Relevance | {performance_data['relevance_score']:.1f}% | {self._get_metric_status(performance_data['relevance_score'])} |
        | Usefulness | {performance_data['usefulness_score']:.1f}% | {self._get_metric_status(performance_data['usefulness_score'])} |
        """
    
    def _format_bias_analysis(self, bias_data: Dict[str, Any]) -> str:
        """Format bias analysis section."""
        return f"""
        **Bias Level:** {bias_data['bias_level']} ({bias_data['overall_bias_score']:.1%})
        
        **Key Findings:**
        {chr(10).join([f"- {finding}" for finding in bias_data.get('bias_insights', [])])}
        
        **Mitigation Status:** {'Active' if bias_data['overall_bias_score'] < 0.3 else 'Needs Improvement'}
        """
    
    def _format_safety_metrics(self, safety_data: Dict[str, Any]) -> str:
        """Format safety metrics section."""
        return f"""
        **Safety Level:** {safety_data['safety_level']}
        
        **Hallucination Risk:** {safety_data['hallucination_score']:.1%}
        
        **Moderation Effectiveness:** {safety_data['moderation_effectiveness']:.1%}
        
        **Active Alerts:** {len(safety_data.get('safety_alerts', []))}
        """
    
    def _format_usage_statistics(self, usage_data: Dict[str, Any]) -> str:
        """Format usage statistics section."""
        return f"""
        **Total Evaluations:** {usage_data['total_evaluations']:,}
        
        **Analysis Period:** {usage_data['evaluation_period']} days
        
        **Average Daily Usage:** {usage_data['average_daily_usage']:.1f} evaluations
        
        **Quality Trend:** {usage_data['quality_trend'].title()}
        
        **Peak Quality Period:** {usage_data['peak_quality_period']}
        """
    
    def _get_model_description(self) -> str:
        """Get model description."""
        return f"""
        {self.model_name} is an enterprise-grade conversational AI system designed for 
        high-quality, bias-aware, and safe interactions. The model incorporates advanced 
        evaluation capabilities through Opik integration, providing real-time quality 
        assessment and self-correction mechanisms.
        
        **Key Features:**
        - Real-time bias detection and mitigation
        - Hallucination monitoring and prevention
        - Self-correction capabilities
        - Comprehensive quality evaluation
        - Administrative dashboard and analytics
        """
    
    def _generate_quality_badge(self, score: float) -> str:
        """Generate quality badge for markdown."""
        if score >= 80:
            return "🟢 **Excellent**"
        elif score >= 70:
            return "🟡 **Good**"
        elif score >= 60:
            return "🟠 **Fair**"
        else:
            return "🔴 **Needs Improvement**"
    
    def _get_metric_status(self, score: float) -> str:
        """Get status label for metric score."""
        if score >= 80:
            return "✅ Excellent"
        elif score >= 70:
            return "✅ Good"
        elif score >= 60:
            return "⚠️ Fair"
        else:
            return "❌ Poor"
    
    def _get_bias_status(self, score: float) -> str:
        """Get status label for bias score (lower is better)."""
        if score <= 20:
            return "✅ Low"
        elif score <= 40:
            return "⚠️ Moderate"
        else:
            return "❌ High"
    
    def _get_safety_status(self, score: float) -> str:
        """Get status label for safety score (lower is better)."""
        if score <= 20:
            return "✅ Safe"
        elif score <= 40:
            return "⚠️ Monitor"
        else:
            return "❌ Concerning"
    
    def _get_overall_status(self, score: float) -> str:
        """Get overall model status."""
        if score >= 80:
            return "production_ready"
        elif score >= 70:
            return "good_performance"
        elif score >= 60:
            return "needs_monitoring"
        else:
            return "needs_improvement"
    
    def _assess_recommendation_priority(self, data: ModelCardData) -> str:
        """Assess priority level for recommendations."""
        if data.performance_summary['overall_score'] < 60:
            return "high"
        elif data.bias_analysis['bias_level'] == 'High':
            return "high"
        elif data.safety_metrics['safety_level'] == 'Safety Concerns':
            return "high"
        elif data.performance_summary['correction_rate'] > 30:
            return "medium"
        else:
            return "low"
    
    # Additional HTML generation methods would go here...
    def _get_model_card_css(self) -> str:
        """Get CSS styles for HTML model card."""
        return """
            .model-card { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                max-width: 1200px; 
                margin: 0 auto; 
                padding: 20px;
                background: white;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                border-radius: 8px;
            }
            .card-header { 
                border-bottom: 2px solid #e9ecef; 
                padding-bottom: 20px; 
                margin-bottom: 30px; 
            }
            .quality-overview { 
                display: flex; 
                gap: 15px; 
                margin-top: 15px; 
            }
            .performance-section { 
                margin-bottom: 30px; 
            }
            .metric-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 20px; 
            }
            .metric-card { 
                background: #f8f9fa; 
                padding: 15px; 
                border-radius: 8px; 
                text-align: center; 
            }
            .metric-score { 
                font-size: 2em; 
                font-weight: bold; 
                color: #007bff; 
            }
            .quality-badge { 
                padding: 5px 15px; 
                border-radius: 20px; 
                font-weight: bold; 
            }
            .quality-excellent { 
                background: #d4edda; 
                color: #155724; 
            }
            .quality-good { 
                background: #fff3cd; 
                color: #856404; 
            }
            .quality-fair { 
                background: #f8d7da; 
                color: #721c24; 
            }
        """
    
    def _generate_metrics_grid_html(self, performance_data: Dict[str, Any]) -> str:
        """Generate HTML metrics grid."""
        return f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-score">{performance_data['accuracy_score']:.1f}%</div>
                    <div class="metric-label">Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-score">{performance_data['bias_score']:.1f}%</div>
                    <div class="metric-label">Bias</div>
                </div>
                <div class="metric-card">
                    <div class="metric-score">{performance_data['relevance_score']:.1f}%</div>
                    <div class="metric-label">Relevance</div>
                </div>
                <div class="metric-card">
                    <div class="metric-score">{performance_data['usefulness_score']:.1f}%</div>
                    <div class="metric-label">Usefulness</div>
                </div>
            </div>
        """
    
    def _get_quality_css_class(self, score: float) -> str:
        """Get CSS class for quality badge."""
        if score >= 80:
            return "quality-excellent"
        elif score >= 70:
            return "quality-good"
        else:
            return "quality-fair"
    
    async def _generate_technical_details(self) -> str:
        """Generate technical details section."""
        return """
        ### Technical Implementation
        
        - **Evaluation Framework:** Opik 1.8.6+ with real-time integration
        - **Bias Detection:** Multi-dimensional analysis across 8+ categories
        - **Safety Monitoring:** Hallucination detection with ML-based scoring
        - **Self-Correction:** Automated improvement based on quality signals
        - **Data Storage:** SQLite with comprehensive evaluation history
        - **API Integration:** RESTful endpoints for real-time data access
        - **UI Components:** Real-time dashboard with live metric updates
        
        ### Model Architecture
        
        - **Core Model:** Enterprise conversational AI with safety layers
        - **Evaluation Layer:** Opik-based quality assessment
        - **Correction Engine:** Rule-based and ML-guided improvements
        - **Memory System:** Context-aware conversation tracking
        - **Monitoring:** Comprehensive logging and analytics
        """
    
    async def _get_technical_details_dict(self) -> Dict[str, Any]:
        """Get technical details as dictionary."""
        return {
            'evaluation_framework': 'Opik 1.8.6+',
            'bias_detection': 'Multi-dimensional analysis (8+ categories)',
            'safety_monitoring': 'ML-based hallucination detection',
            'self_correction': 'Automated quality-based improvements',
            'data_storage': 'SQLite with evaluation history',
            'api_integration': 'RESTful real-time endpoints',
            'ui_components': 'Live dashboard with real-time updates',
            'architecture': {
                'core_model': 'Enterprise conversational AI',
                'evaluation_layer': 'Opik-based assessment',
                'correction_engine': 'Rule-based + ML-guided',
                'memory_system': 'Context-aware tracking',
                'monitoring': 'Comprehensive analytics'
            }
        }
    
    async def generate_compact_summary(self) -> Dict[str, Any]:
        """Generate a compact model summary for quick reference."""
        try:
            # Get latest performance stats
            performance_stats = await self.evaluator.get_model_performance_stats()
            overall_metrics = performance_stats.get('performance_metrics', {}).get('overall', {})
            
            return {
                'model_name': self.model_name,
                'model_version': self.model_version,
                'overall_quality': f"{overall_metrics.get('composite_score', 0) * 100:.1f}%",
                'bias_level': self._categorize_bias_level(overall_metrics.get('bias', 0)),
                'safety_level': self._categorize_safety_level(overall_metrics.get('hallucination', 0)),
                'status': performance_stats.get('quality_assessment', {}).get('status', 'unknown'),
                'last_updated': datetime.utcnow().isoformat(),
                'quick_stats': {
                    'accuracy': f"{overall_metrics.get('accuracy', 0) * 100:.1f}%",
                    'relevance': f"{overall_metrics.get('relevance', 0) * 100:.1f}%",
                    'usefulness': f"{overall_metrics.get('usefulness', 0) * 100:.1f}%"
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to generate compact summary: {e}")
            return {'error': str(e)}

# Global instance
dynamic_model_cards = DynamicModelCards()
