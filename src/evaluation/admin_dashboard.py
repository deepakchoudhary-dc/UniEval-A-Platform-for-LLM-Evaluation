"""
Administrative Dashboard for Opik Evaluation Analytics

This module provides comprehensive administrative dashboard functionality for
analyzing historical evaluation trends, performance metrics, and system insights.
"""

import logging
import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import statistics
from dataclasses import dataclass, asdict

from .enhanced_opik_evaluator import enhanced_opik_evaluator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DashboardMetrics:
    """Comprehensive dashboard metrics."""
    period_start: datetime
    period_end: datetime
    total_evaluations: int
    total_conversations: int
    average_scores: Dict[str, float]
    quality_distribution: Dict[str, int]
    correction_statistics: Dict[str, int]
    daily_trends: List[Dict[str, Any]]
    top_issues: List[Dict[str, Any]]
    performance_indicators: Dict[str, Any]

class AdminDashboard:
    """
    Administrative dashboard for comprehensive evaluation analytics.
    """
    
    def __init__(self):
        """Initialize the administrative dashboard."""
        self.evaluator = enhanced_opik_evaluator
        self.db_path = self.evaluator.db_path
        
        logger.info("Administrative dashboard initialized")
    
    async def get_comprehensive_analytics(
        self,
        days: int = 30,
        include_trends: bool = True,
        include_comparisons: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive analytics for the dashboard.
        
        Args:
            days: Number of days to analyze
            include_trends: Whether to include trend analysis
            include_comparisons: Whether to include period comparisons
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Get base metrics
            base_metrics = await self.evaluator.get_dashboard_metrics(days)
            
            # Get additional analytics
            conversation_analytics = await self._get_conversation_analytics(start_date, end_date)
            user_patterns = await self._get_user_interaction_patterns(start_date, end_date)
            quality_insights = await self._get_quality_insights(start_date, end_date)
            
            # Get trend analysis if requested
            trend_analysis = {}
            if include_trends:
                trend_analysis = await self._get_trend_analysis(start_date, end_date)
            
            # Get period comparisons if requested
            period_comparison = {}
            if include_comparisons:
                period_comparison = await self._get_period_comparison(days)
            
            # Generate insights and recommendations
            insights = await self._generate_insights(base_metrics, trend_analysis, quality_insights)
            
            return {
                'dashboard_metadata': {
                    'generated_at': datetime.utcnow().isoformat(),
                    'period_days': days,
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'data_completeness': await self._assess_data_completeness(start_date, end_date)
                },
                'base_metrics': base_metrics,
                'conversation_analytics': conversation_analytics,
                'user_patterns': user_patterns,
                'quality_insights': quality_insights,
                'trend_analysis': trend_analysis,
                'period_comparison': period_comparison,
                'insights_and_recommendations': insights,
                'charts_data': await self._generate_charts_data(base_metrics, trend_analysis),
                'alerts': await self._generate_alerts(base_metrics, trend_analysis)
            }
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive analytics: {e}")
            return {'error': str(e)}
    
    async def _get_conversation_analytics(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get detailed conversation-level analytics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Conversation statistics
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT conversation_id) as total_conversations,
                    AVG(exchanges) as avg_exchanges_per_conversation,
                    MAX(exchanges) as max_exchanges,
                    MIN(exchanges) as min_exchanges
                FROM (
                    SELECT 
                        conversation_id,
                        COUNT(*) as exchanges
                    FROM evaluations 
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY conversation_id
                ) as conv_stats
            """, (start_date.isoformat(), end_date.isoformat()))
            
            conv_stats = cursor.fetchone()
            
            # Conversation quality distribution
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN avg_score >= 0.8 THEN 'high_quality'
                        WHEN avg_score >= 0.6 THEN 'medium_quality'
                        ELSE 'low_quality'
                    END as quality_category,
                    COUNT(*) as count
                FROM (
                    SELECT 
                        conversation_id,
                        AVG(overall_score) as avg_score
                    FROM evaluations 
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY conversation_id
                ) as conv_quality
                GROUP BY quality_category
            """, (start_date.isoformat(), end_date.isoformat()))
            
            quality_dist = dict(cursor.fetchall())
            
            # Conversation length impact on quality
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN exchanges <= 5 THEN 'short'
                        WHEN exchanges <= 15 THEN 'medium'
                        ELSE 'long'
                    END as length_category,
                    AVG(avg_score) as avg_quality,
                    COUNT(*) as count
                FROM (
                    SELECT 
                        conversation_id,
                        COUNT(*) as exchanges,
                        AVG(overall_score) as avg_score
                    FROM evaluations 
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY conversation_id
                ) as conv_analysis
                GROUP BY length_category
            """, (start_date.isoformat(), end_date.isoformat()))
            
            length_impact = cursor.fetchall()
            
            conn.close()
            
            return {
                'total_conversations': conv_stats[0] if conv_stats[0] else 0,
                'avg_exchanges_per_conversation': round(conv_stats[1], 2) if conv_stats[1] else 0,
                'max_exchanges': conv_stats[2] if conv_stats[2] else 0,
                'min_exchanges': conv_stats[3] if conv_stats[3] else 0,
                'quality_distribution': quality_dist,
                'length_impact_on_quality': [
                    {
                        'length_category': row[0],
                        'avg_quality': round(row[1], 3) if row[1] else 0,
                        'conversation_count': row[2]
                    }
                    for row in length_impact
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get conversation analytics: {e}")
            return {}
    
    async def _get_user_interaction_patterns(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Analyze user interaction patterns."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Hourly usage patterns
            cursor.execute("""
                SELECT 
                    strftime('%H', timestamp) as hour,
                    COUNT(*) as evaluations,
                    AVG(overall_score) as avg_quality
                FROM evaluations 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY strftime('%H', timestamp)
                ORDER BY hour
            """, (start_date.isoformat(), end_date.isoformat()))
            
            hourly_patterns = [
                {
                    'hour': int(row[0]),
                    'evaluations': row[1],
                    'avg_quality': round(row[2], 3) if row[2] else 0
                }
                for row in cursor.fetchall()
            ]
            
            # Daily usage patterns
            cursor.execute("""
                SELECT 
                    strftime('%w', timestamp) as day_of_week,
                    COUNT(*) as evaluations,
                    AVG(overall_score) as avg_quality
                FROM evaluations 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY strftime('%w', timestamp)
                ORDER BY day_of_week
            """, (start_date.isoformat(), end_date.isoformat()))
            
            daily_patterns = [
                {
                    'day_of_week': int(row[0]),
                    'day_name': ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][int(row[0])],
                    'evaluations': row[1],
                    'avg_quality': round(row[2], 3) if row[2] else 0
                }
                for row in cursor.fetchall()
            ]
            
            # Response length patterns
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN LENGTH(output_text) <= 100 THEN 'short'
                        WHEN LENGTH(output_text) <= 500 THEN 'medium'
                        WHEN LENGTH(output_text) <= 1500 THEN 'long'
                        ELSE 'very_long'
                    END as response_length,
                    COUNT(*) as count,
                    AVG(overall_score) as avg_quality,
                    AVG(relevance_score) as avg_relevance,
                    AVG(usefulness_score) as avg_usefulness
                FROM evaluations 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY response_length
            """, (start_date.isoformat(), end_date.isoformat()))
            
            length_patterns = [
                {
                    'response_length': row[0],
                    'count': row[1],
                    'avg_quality': round(row[2], 3) if row[2] else 0,
                    'avg_relevance': round(row[3], 3) if row[3] else 0,
                    'avg_usefulness': round(row[4], 3) if row[4] else 0
                }
                for row in cursor.fetchall()
            ]
            
            conn.close()
            
            return {
                'hourly_patterns': hourly_patterns,
                'daily_patterns': daily_patterns,
                'response_length_patterns': length_patterns,
                'peak_usage_hour': max(hourly_patterns, key=lambda x: x['evaluations'])['hour'] if hourly_patterns else None,
                'best_quality_hour': max(hourly_patterns, key=lambda x: x['avg_quality'])['hour'] if hourly_patterns else None,
                'optimal_response_length': max(length_patterns, key=lambda x: x['avg_quality'])['response_length'] if length_patterns else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get user interaction patterns: {e}")
            return {}
    
    async def _get_quality_insights(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get detailed quality insights and correlations."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all evaluation data for analysis
            cursor.execute("""
                SELECT 
                    accuracy_score, bias_score, hallucination_score,
                    relevance_score, usefulness_score, overall_score,
                    requires_correction, confidence_level
                FROM evaluations 
                WHERE timestamp >= ? AND timestamp <= ?
            """, (start_date.isoformat(), end_date.isoformat()))
            
            data = cursor.fetchall()
            
            if not data:
                return {}
            
            # Calculate correlations and insights
            accuracy_scores = [row[0] for row in data if row[0] is not None]
            bias_scores = [row[1] for row in data if row[1] is not None]
            hallucination_scores = [row[2] for row in data if row[2] is not None]
            relevance_scores = [row[3] for row in data if row[3] is not None]
            usefulness_scores = [row[4] for row in data if row[4] is not None]
            overall_scores = [row[5] for row in data if row[5] is not None]
            
            # Statistical analysis
            quality_stats = {
                'accuracy': {
                    'mean': statistics.mean(accuracy_scores) if accuracy_scores else 0,
                    'median': statistics.median(accuracy_scores) if accuracy_scores else 0,
                    'stdev': statistics.stdev(accuracy_scores) if len(accuracy_scores) > 1 else 0,
                    'min': min(accuracy_scores) if accuracy_scores else 0,
                    'max': max(accuracy_scores) if accuracy_scores else 0
                },
                'bias': {
                    'mean': statistics.mean(bias_scores) if bias_scores else 0,
                    'median': statistics.median(bias_scores) if bias_scores else 0,
                    'stdev': statistics.stdev(bias_scores) if len(bias_scores) > 1 else 0,
                    'min': min(bias_scores) if bias_scores else 0,
                    'max': max(bias_scores) if bias_scores else 0
                },
                'hallucination': {
                    'mean': statistics.mean(hallucination_scores) if hallucination_scores else 0,
                    'median': statistics.median(hallucination_scores) if hallucination_scores else 0,
                    'stdev': statistics.stdev(hallucination_scores) if len(hallucination_scores) > 1 else 0,
                    'min': min(hallucination_scores) if hallucination_scores else 0,
                    'max': max(hallucination_scores) if hallucination_scores else 0
                },
                'overall': {
                    'mean': statistics.mean(overall_scores) if overall_scores else 0,
                    'median': statistics.median(overall_scores) if overall_scores else 0,
                    'stdev': statistics.stdev(overall_scores) if len(overall_scores) > 1 else 0,
                    'min': min(overall_scores) if overall_scores else 0,
                    'max': max(overall_scores) if overall_scores else 0
                }
            }
            
            # Quality consistency analysis
            consistency_analysis = {
                'overall_consistency': 1 - (quality_stats['overall']['stdev'] / quality_stats['overall']['mean']) if quality_stats['overall']['mean'] > 0 else 0,
                'accuracy_consistency': 1 - (quality_stats['accuracy']['stdev'] / quality_stats['accuracy']['mean']) if quality_stats['accuracy']['mean'] > 0 else 0,
                'most_consistent_metric': min(
                    ['accuracy', 'bias', 'hallucination'],
                    key=lambda x: quality_stats[x]['stdev']
                ) if len(data) > 1 else 'unknown',
                'least_consistent_metric': max(
                    ['accuracy', 'bias', 'hallucination'],
                    key=lambda x: quality_stats[x]['stdev']
                ) if len(data) > 1 else 'unknown'
            }
            
            # Correction pattern analysis
            correction_data = [row for row in data if row[6] is not None]  # requires_correction
            correction_rate = sum(1 for row in correction_data if row[6]) / len(correction_data) if correction_data else 0
            
            # Confidence level analysis
            confidence_data = [row[7] for row in data if row[7] is not None]
            confidence_distribution = {
                'high': sum(1 for conf in confidence_data if conf == 'high') / len(confidence_data) if confidence_data else 0,
                'medium': sum(1 for conf in confidence_data if conf == 'medium') / len(confidence_data) if confidence_data else 0,
                'low': sum(1 for conf in confidence_data if conf == 'low') / len(confidence_data) if confidence_data else 0
            }
            
            conn.close()
            
            return {
                'quality_statistics': quality_stats,
                'consistency_analysis': consistency_analysis,
                'correction_patterns': {
                    'overall_correction_rate': round(correction_rate, 3),
                    'corrections_needed': sum(1 for row in correction_data if row[6]),
                    'total_evaluations': len(correction_data)
                },
                'confidence_distribution': confidence_distribution,
                'quality_insights': {
                    'strongest_metric': max(['accuracy', 'relevance', 'usefulness'], key=lambda x: quality_stats.get(x, {}).get('mean', 0)),
                    'weakest_metric': min(['accuracy', 'relevance', 'usefulness'], key=lambda x: quality_stats.get(x, {}).get('mean', 1)),
                    'bias_level': 'low' if quality_stats['bias']['mean'] < 0.3 else 'medium' if quality_stats['bias']['mean'] < 0.6 else 'high',
                    'hallucination_risk': 'low' if quality_stats['hallucination']['mean'] < 0.3 else 'medium' if quality_stats['hallucination']['mean'] < 0.6 else 'high'
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get quality insights: {e}")
            return {}
    
    async def _get_trend_analysis(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Perform comprehensive trend analysis."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Daily trends with more detail
            cursor.execute("""
                SELECT 
                    DATE(timestamp) as date,
                    COUNT(*) as evaluations,
                    AVG(accuracy_score) as avg_accuracy,
                    AVG(bias_score) as avg_bias,
                    AVG(hallucination_score) as avg_hallucination,
                    AVG(relevance_score) as avg_relevance,
                    AVG(usefulness_score) as avg_usefulness,
                    AVG(overall_score) as avg_overall,
                    SUM(CASE WHEN requires_correction THEN 1 ELSE 0 END) as corrections
                FROM evaluations 
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, (start_date.isoformat(), end_date.isoformat()))
            
            daily_trends = []
            for row in cursor.fetchall():
                daily_trends.append({
                    'date': row[0],
                    'evaluations': row[1],
                    'avg_accuracy': round(row[2], 3) if row[2] else 0,
                    'avg_bias': round(row[3], 3) if row[3] else 0,
                    'avg_hallucination': round(row[4], 3) if row[4] else 0,
                    'avg_relevance': round(row[5], 3) if row[5] else 0,
                    'avg_usefulness': round(row[6], 3) if row[6] else 0,
                    'avg_overall': round(row[7], 3) if row[7] else 0,
                    'corrections': row[8]
                })
            
            # Calculate trend directions
            trend_analysis = {}
            if len(daily_trends) >= 3:
                recent_period = daily_trends[-7:]  # Last week
                early_period = daily_trends[:7]    # First week
                
                if recent_period and early_period:
                    recent_avg = statistics.mean([day['avg_overall'] for day in recent_period])
                    early_avg = statistics.mean([day['avg_overall'] for day in early_period])
                    
                    trend_analysis = {
                        'overall_trend': 'improving' if recent_avg > early_avg else 'declining' if recent_avg < early_avg else 'stable',
                        'trend_magnitude': abs(recent_avg - early_avg),
                        'recent_avg_quality': recent_avg,
                        'early_avg_quality': early_avg,
                        'improvement_rate': (recent_avg - early_avg) / early_avg * 100 if early_avg > 0 else 0
                    }
            
            conn.close()
            
            return {
                'daily_trends': daily_trends,
                'trend_analysis': trend_analysis,
                'data_points': len(daily_trends),
                'analysis_period': (end_date - start_date).days
            }
            
        except Exception as e:
            logger.error(f"Failed to get trend analysis: {e}")
            return {}
    
    async def _get_period_comparison(self, days: int) -> Dict[str, Any]:
        """Compare current period with previous period."""
        try:
            current_end = datetime.utcnow()
            current_start = current_end - timedelta(days=days)
            previous_end = current_start
            previous_start = previous_end - timedelta(days=days)
            
            # Get metrics for both periods
            current_metrics = await self.evaluator.get_dashboard_metrics(days)
            
            # Get previous period metrics manually
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_evaluations,
                    AVG(accuracy_score) as avg_accuracy,
                    AVG(bias_score) as avg_bias,
                    AVG(hallucination_score) as avg_hallucination,
                    AVG(relevance_score) as avg_relevance,
                    AVG(usefulness_score) as avg_usefulness,
                    AVG(overall_score) as avg_overall,
                    SUM(CASE WHEN requires_correction THEN 1 ELSE 0 END) as total_corrections
                FROM evaluations 
                WHERE timestamp >= ? AND timestamp <= ?
            """, (previous_start.isoformat(), previous_end.isoformat()))
            
            previous_data = cursor.fetchone()
            conn.close()
            
            if not previous_data or not previous_data[0]:
                return {'error': 'Insufficient data for period comparison'}
            
            previous_metrics = {
                'total_evaluations': previous_data[0],
                'average_scores': {
                    'accuracy': previous_data[1] or 0,
                    'bias': previous_data[2] or 0,
                    'hallucination': previous_data[3] or 0,
                    'relevance': previous_data[4] or 0,
                    'usefulness': previous_data[5] or 0,
                    'overall': previous_data[6] or 0
                },
                'total_corrections': previous_data[7] or 0
            }
            
            # Calculate comparisons
            comparison = {
                'current_period': {
                    'start': current_start.isoformat(),
                    'end': current_end.isoformat(),
                    'metrics': current_metrics
                },
                'previous_period': {
                    'start': previous_start.isoformat(),
                    'end': previous_end.isoformat(),
                    'metrics': previous_metrics
                },
                'changes': {
                    'evaluations_change': current_metrics.get('summary', {}).get('total_evaluations', 0) - previous_metrics['total_evaluations'],
                    'quality_change': current_metrics.get('average_scores', {}).get('overall', 0) - previous_metrics['average_scores']['overall'],
                    'bias_change': previous_metrics['average_scores']['bias'] - current_metrics.get('average_scores', {}).get('bias', 0),  # Lower is better
                    'corrections_change': current_metrics.get('summary', {}).get('total_corrections', 0) - previous_metrics['total_corrections']
                },
                'percentage_changes': {}
            }
            
            # Calculate percentage changes
            for metric in ['accuracy', 'bias', 'hallucination', 'relevance', 'usefulness', 'overall']:
                current_val = current_metrics.get('average_scores', {}).get(metric, 0)
                previous_val = previous_metrics['average_scores'][metric]
                
                if previous_val > 0:
                    change = (current_val - previous_val) / previous_val * 100
                    comparison['percentage_changes'][metric] = round(change, 2)
                else:
                    comparison['percentage_changes'][metric] = 0
            
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to get period comparison: {e}")
            return {'error': str(e)}
    
    async def _generate_insights(
        self,
        base_metrics: Dict[str, Any],
        trend_analysis: Dict[str, Any],
        quality_insights: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate actionable insights and recommendations."""
        insights = {
            'key_findings': [],
            'recommendations': [],
            'alerts': [],
            'achievements': []
        }
        
        # Analyze base metrics
        if base_metrics.get('average_scores', {}).get('overall', 0) >= 0.8:
            insights['achievements'].append("Excellent overall quality maintained above 80%")
        elif base_metrics.get('average_scores', {}).get('overall', 0) < 0.6:
            insights['alerts'].append("Overall quality below 60% - immediate attention needed")
        
        # Analyze bias
        bias_score = base_metrics.get('average_scores', {}).get('bias', 0)
        if bias_score > 0.6:
            insights['alerts'].append(f"High bias detected (>{bias_score:.1%}) - review content guidelines")
            insights['recommendations'].append("Implement additional bias detection and training")
        elif bias_score < 0.2:
            insights['achievements'].append("Excellent bias control maintained")
        
        # Analyze hallucination
        hallucination_score = base_metrics.get('average_scores', {}).get('hallucination', 0)
        if hallucination_score > 0.5:
            insights['alerts'].append(f"High hallucination risk ({hallucination_score:.1%}) detected")
            insights['recommendations'].append("Enhance fact-checking and source verification")
        
        # Analyze correction rate
        correction_rate = base_metrics.get('summary', {}).get('correction_rate', 0)
        if correction_rate > 30:
            insights['alerts'].append(f"High correction rate ({correction_rate:.1f}%) indicates quality issues")
            insights['recommendations'].append("Review and improve initial response generation")
        elif correction_rate < 10:
            insights['achievements'].append("Low correction rate indicates high initial quality")
        
        # Analyze trends
        if trend_analysis.get('trend_analysis', {}).get('overall_trend') == 'improving':
            insights['achievements'].append("Quality trend is improving over time")
        elif trend_analysis.get('trend_analysis', {}).get('overall_trend') == 'declining':
            insights['alerts'].append("Quality trend is declining - investigate root causes")
            insights['recommendations'].append("Analyze recent changes that may impact quality")
        
        # Quality consistency insights
        consistency = quality_insights.get('consistency_analysis', {}).get('overall_consistency', 0)
        if consistency < 0.7:
            insights['key_findings'].append("Quality consistency is below optimal levels")
            insights['recommendations'].append("Improve consistency through standardized processes")
        
        # Add general recommendations
        insights['recommendations'].extend([
            "Monitor daily trends for early detection of issues",
            "Regular review of correction patterns to identify improvement opportunities",
            "Continue bias and hallucination monitoring",
            "Maintain quality documentation and guidelines"
        ])
        
        return insights
    
    async def _generate_charts_data(
        self,
        base_metrics: Dict[str, Any],
        trend_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate data for dashboard charts and visualizations."""
        return {
            'quality_overview_chart': {
                'type': 'radar',
                'data': {
                    'labels': ['Accuracy', 'Relevance', 'Usefulness', 'Low Bias', 'Low Hallucination'],
                    'values': [
                        base_metrics.get('average_scores', {}).get('accuracy', 0) * 100,
                        base_metrics.get('average_scores', {}).get('relevance', 0) * 100,
                        base_metrics.get('average_scores', {}).get('usefulness', 0) * 100,
                        (1 - base_metrics.get('average_scores', {}).get('bias', 0)) * 100,
                        (1 - base_metrics.get('average_scores', {}).get('hallucination', 0)) * 100
                    ]
                }
            },
            'daily_trends_chart': {
                'type': 'line',
                'data': {
                    'dates': [trend['date'] for trend in trend_analysis.get('daily_trends', [])],
                    'overall_quality': [trend['avg_overall'] * 100 for trend in trend_analysis.get('daily_trends', [])],
                    'evaluations_count': [trend['evaluations'] for trend in trend_analysis.get('daily_trends', [])],
                    'corrections': [trend['corrections'] for trend in trend_analysis.get('daily_trends', [])]
                }
            },
            'quality_distribution_chart': {
                'type': 'pie',
                'data': base_metrics.get('quality_distribution', {})
            },
            'top_issues_chart': {
                'type': 'bar',
                'data': {
                    'issues': [issue['issue'] for issue in base_metrics.get('top_issues', [])[:5]],
                    'frequencies': [issue['frequency'] for issue in base_metrics.get('top_issues', [])[:5]]
                }
            }
        }
    
    async def _generate_alerts(
        self,
        base_metrics: Dict[str, Any],
        trend_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate system alerts based on metrics."""
        alerts = []
        
        # Quality alerts
        overall_quality = base_metrics.get('average_scores', {}).get('overall', 0)
        if overall_quality < 0.5:
            alerts.append({
                'type': 'critical',
                'message': f'Overall quality critically low: {overall_quality:.1%}',
                'action': 'Immediate review and intervention required'
            })
        elif overall_quality < 0.7:
            alerts.append({
                'type': 'warning',
                'message': f'Overall quality below target: {overall_quality:.1%}',
                'action': 'Review recent changes and adjust parameters'
            })
        
        # Bias alerts
        bias_score = base_metrics.get('average_scores', {}).get('bias', 0)
        if bias_score > 0.6:
            alerts.append({
                'type': 'warning',
                'message': f'High bias detected: {bias_score:.1%}',
                'action': 'Review content guidelines and bias detection rules'
            })
        
        # Correction rate alerts
        correction_rate = base_metrics.get('summary', {}).get('correction_rate', 0)
        if correction_rate > 40:
            alerts.append({
                'type': 'warning',
                'message': f'High correction rate: {correction_rate:.1f}%',
                'action': 'Investigate and improve initial response quality'
            })
        
        # Trend alerts
        if trend_analysis.get('trend_analysis', {}).get('overall_trend') == 'declining':
            alerts.append({
                'type': 'info',
                'message': 'Quality trend declining over analysis period',
                'action': 'Monitor closely and identify potential causes'
            })
        
        return alerts
    
    async def _assess_data_completeness(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Assess the completeness and quality of evaluation data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total records in period
            cursor.execute("""
                SELECT COUNT(*) FROM evaluations 
                WHERE timestamp >= ? AND timestamp <= ?
            """, (start_date.isoformat(), end_date.isoformat()))
            total_records = cursor.fetchone()[0]
            
            # Records with complete data
            cursor.execute("""
                SELECT COUNT(*) FROM evaluations 
                WHERE timestamp >= ? AND timestamp <= ?
                AND accuracy_score IS NOT NULL 
                AND bias_score IS NOT NULL 
                AND hallucination_score IS NOT NULL
                AND relevance_score IS NOT NULL
                AND usefulness_score IS NOT NULL
                AND overall_score IS NOT NULL
            """, (start_date.isoformat(), end_date.isoformat()))
            complete_records = cursor.fetchone()[0]
            
            # Days with data
            cursor.execute("""
                SELECT COUNT(DISTINCT DATE(timestamp)) FROM evaluations 
                WHERE timestamp >= ? AND timestamp <= ?
            """, (start_date.isoformat(), end_date.isoformat()))
            days_with_data = cursor.fetchone()[0]
            
            total_days = (end_date - start_date).days + 1
            
            conn.close()
            
            return {
                'total_records': total_records,
                'complete_records': complete_records,
                'completeness_rate': (complete_records / total_records * 100) if total_records > 0 else 0,
                'days_with_data': days_with_data,
                'total_days': total_days,
                'coverage_rate': (days_with_data / total_days * 100) if total_days > 0 else 0,
                'data_quality': 'excellent' if complete_records / total_records > 0.95 else 'good' if complete_records / total_records > 0.8 else 'fair'
            }
            
        except Exception as e:
            logger.error(f"Failed to assess data completeness: {e}")
            return {'error': str(e)}

# Global instance
admin_dashboard = AdminDashboard()
