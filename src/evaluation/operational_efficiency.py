"""
Operational Efficiency Evaluation Module
Implements operational efficiency and performance assessments
"""

import time
import asyncio
import psutil
import statistics
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class OperationalEfficiencyEvaluator:
    """
    Evaluates operational efficiency metrics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_history = defaultdict(list)
        self.resource_monitor = ResourceMonitor()
    
    async def evaluate_operational_efficiency(self, query: str, response: str, 
                                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Main operational efficiency evaluation method called by comprehensive evaluator
        """
        evaluation_context = {
            'input_text': query,
            'output_text': response,
            'session_id': context.get('session_id') if context else None,
            'model': context.get('model') if context else None
        }
        
        # Run comprehensive efficiency evaluation
        results = await self.evaluate_all(evaluation_context)
        
        # Calculate overall efficiency score
        performance_score = results.get('performance_score', 0.8)
        resource_score = results.get('resource_efficiency', 0.8)
        scalability_score = results.get('scalability_score', 0.8)
        
        overall_score = np.mean([performance_score, resource_score, scalability_score])
        
        return {
            'overall_score': float(overall_score),
            'meets_enterprise_standards': bool(overall_score >= 0.7),
            'detailed_scores': results
        }

    async def evaluate_efficiency(self, query: str, response: str, 
                                session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Main efficiency evaluation method called by API
        """
        evaluation_context = {
            'input_text': query,
            'output_text': response,
            'session_id': session_id,
            'model': None
        }
        
        # Run comprehensive efficiency evaluation
        results = await self.evaluate_all(evaluation_context)
        
        return results
    
    async def evaluate_all(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate all operational efficiency metrics
        """
        results = {}
        
        input_text = context['input_text']
        output_text = context['output_text']
        model = context.get('model')
        
        # Latency & Response Time
        if self.config.get('enable_latency'):
            latency_scores = await self._evaluate_latency_metrics(context)
            results.update(latency_scores)
        
        # Throughput & Scalability
        if self.config.get('enable_throughput'):
            throughput_scores = await self._evaluate_throughput_metrics(context)
            results.update(throughput_scores)
        
        # Token Efficiency
        if self.config.get('enable_token_efficiency'):
            token_scores = await self._evaluate_token_efficiency(
                input_text, output_text
            )
            results.update(token_scores)
        
        # Resource Utilization
        if self.config.get('enable_resource_utilization'):
            resource_scores = await self._evaluate_resource_utilization(context)
            results.update(resource_scores)
        
        # Cost Efficiency
        if self.config.get('enable_cost_efficiency'):
            cost_scores = await self._evaluate_cost_efficiency(context)
            results.update(cost_scores)
        
        # Energy Efficiency
        if self.config.get('enable_energy_efficiency'):
            energy_scores = await self._evaluate_energy_efficiency(context)
            results.update(energy_scores)
        
        # Caching & Optimization
        if self.config.get('enable_caching'):
            cache_scores = await self._evaluate_caching_efficiency(context)
            results.update(cache_scores)
        
        # Load Balancing
        if self.config.get('enable_load_balancing'):
            load_scores = await self._evaluate_load_balancing(context)
            results.update(load_scores)
        
        # Memory Management
        if self.config.get('enable_memory_management'):
            memory_scores = await self._evaluate_memory_management(context)
            results.update(memory_scores)
        
        # Concurrency & Parallelization
        if self.config.get('enable_concurrency'):
            concurrency_scores = await self._evaluate_concurrency_efficiency(context)
            results.update(concurrency_scores)
        
        # API Rate Limiting
        if self.config.get('enable_rate_limiting'):
            rate_limit_scores = await self._evaluate_rate_limiting_efficiency(context)
            results.update(rate_limit_scores)
        
        # Error Handling Efficiency
        if self.config.get('enable_error_handling'):
            error_scores = await self._evaluate_error_handling_efficiency(context)
            results.update(error_scores)
        
        return results
    
    async def _evaluate_latency_metrics(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate latency and response time metrics
        """
        results = {}
        
        # Basic latency measurements
        start_time = context.get('start_time', time.time())
        end_time = context.get('end_time', time.time())
        
        response_time = end_time - start_time
        
        # Normalize response time (assuming 0-10 seconds is reasonable range)
        normalized_response_time = max(0, 1 - (response_time / 10.0))
        results['response_time_score'] = normalized_response_time
        
        # First Token Time (if available)
        first_token_time = context.get('first_token_time')
        if first_token_time:
            time_to_first_token = first_token_time - start_time
            normalized_ttft = max(0, 1 - (time_to_first_token / 5.0))
            results['time_to_first_token_score'] = normalized_ttft
        
        # Token Generation Rate
        output_tokens = len(context.get('output_text', '').split())
        if output_tokens > 0 and response_time > 0:
            tokens_per_second = output_tokens / response_time
            # Normalize assuming 100 tokens/sec is excellent
            normalized_tps = min(tokens_per_second / 100.0, 1.0)
            results['token_generation_rate'] = normalized_tps
        
        # Percentile Latencies (if history available)
        latency_history = self.performance_history.get('latency', [])
        latency_history.append(response_time)
        
        if len(latency_history) >= 10:
            p50 = np.percentile(latency_history, 50)
            p95 = np.percentile(latency_history, 95)
            p99 = np.percentile(latency_history, 99)
            
            results['p50_latency_score'] = max(0, 1 - (p50 / 10.0))
            results['p95_latency_score'] = max(0, 1 - (p95 / 15.0))
            results['p99_latency_score'] = max(0, 1 - (p99 / 20.0))
        
        # Consistency Score (low variance is better)
        if len(latency_history) >= 5:
            latency_std = np.std(latency_history[-10:])  # Last 10 measurements
            consistency_score = max(0, 1 - (latency_std / 2.0))
            results['latency_consistency'] = consistency_score
        
        return results
    
    async def _evaluate_throughput_metrics(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate throughput and scalability metrics
        """
        results = {}
        
        # Requests per second (if available)
        concurrent_requests = context.get('concurrent_requests', 1)
        response_time = context.get('end_time', time.time()) - context.get('start_time', time.time())
        
        if response_time > 0:
            rps = concurrent_requests / response_time
            # Normalize assuming 50 RPS is excellent
            normalized_rps = min(rps / 50.0, 1.0)
            results['requests_per_second'] = normalized_rps
        
        # Queue Processing Efficiency
        queue_wait_time = context.get('queue_wait_time', 0)
        if queue_wait_time >= 0:
            # Lower wait time is better
            queue_efficiency = max(0, 1 - (queue_wait_time / 5.0))
            results['queue_efficiency'] = queue_efficiency
        
        # Batch Processing Efficiency
        batch_size = context.get('batch_size', 1)
        if batch_size > 1:
            # Larger batches are generally more efficient
            batch_efficiency = min(batch_size / 32.0, 1.0)  # Normalize to 32 as max
            results['batch_processing_efficiency'] = batch_efficiency
        
        # Scalability Score
        load_factor = context.get('load_factor', 1.0)  # Current load vs capacity
        if load_factor <= 1.0:
            scalability_score = 1.0 - (load_factor * 0.8)  # Linear degradation
        else:
            scalability_score = max(0, 0.2 - ((load_factor - 1.0) * 0.2))
        
        results['scalability_score'] = scalability_score
        
        return results
    
    async def _evaluate_token_efficiency(self, input_text: str, output_text: str) -> Dict[str, float]:
        """
        Evaluate token usage efficiency
        """
        results = {}
        
        input_tokens = len(input_text.split())
        output_tokens = len(output_text.split())
        total_tokens = input_tokens + output_tokens
        
        # Token-to-Information Ratio
        information_content = self._estimate_information_content(output_text)
        if output_tokens > 0:
            token_efficiency = information_content / output_tokens
            # Normalize assuming 0.5 information per token is good
            results['token_efficiency'] = min(token_efficiency / 0.5, 1.0)
        
        # Input-Output Ratio
        if input_tokens > 0:
            io_ratio = output_tokens / input_tokens
            # Ideal ratio depends on task, but 2-5x is often reasonable
            if 2 <= io_ratio <= 5:
                results['input_output_ratio_score'] = 1.0
            elif io_ratio < 2:
                results['input_output_ratio_score'] = io_ratio / 2.0
            else:
                results['input_output_ratio_score'] = max(0, 1 - ((io_ratio - 5) / 10.0))
        
        # Redundancy Detection
        redundancy_score = self._calculate_redundancy(output_text)
        results['redundancy_score'] = 1.0 - redundancy_score
        
        # Compression Ratio (semantic density)
        semantic_density = self._calculate_semantic_density(output_text)
        results['semantic_density'] = semantic_density
        
        return results
    
    async def _evaluate_resource_utilization(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate system resource utilization
        """
        results = {}
        
        # CPU Utilization
        cpu_usage = self.resource_monitor.get_cpu_usage()
        cpu_efficiency = self._calculate_resource_efficiency(cpu_usage)
        results['cpu_efficiency'] = cpu_efficiency
        
        # Memory Utilization
        memory_usage = self.resource_monitor.get_memory_usage()
        memory_efficiency = self._calculate_resource_efficiency(memory_usage)
        results['memory_efficiency'] = memory_efficiency
        
        # GPU Utilization (if available)
        gpu_usage = self.resource_monitor.get_gpu_usage()
        if gpu_usage is not None:
            gpu_efficiency = self._calculate_resource_efficiency(gpu_usage)
            results['gpu_efficiency'] = gpu_efficiency
        
        # Disk I/O Efficiency
        disk_io = self.resource_monitor.get_disk_io()
        disk_efficiency = self._calculate_io_efficiency(disk_io)
        results['disk_io_efficiency'] = disk_efficiency
        
        # Network I/O Efficiency
        network_io = self.resource_monitor.get_network_io()
        network_efficiency = self._calculate_io_efficiency(network_io)
        results['network_io_efficiency'] = network_efficiency
        
        return results
    
    async def _evaluate_cost_efficiency(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate cost efficiency metrics
        """
        results = {}
        
        # Cost per Token
        total_tokens = len(context.get('input_text', '').split()) + len(context.get('output_text', '').split())
        cost_per_request = context.get('cost_per_request', 0.01)  # Default cost
        
        if total_tokens > 0:
            cost_per_token = cost_per_request / total_tokens
            # Normalize assuming $0.001 per token is reasonable
            cost_efficiency = max(0, 1 - (cost_per_token / 0.001))
            results['cost_per_token_efficiency'] = cost_efficiency
        
        # Cost per Quality Unit
        quality_score = context.get('quality_score', 0.5)
        if quality_score > 0:
            cost_per_quality = cost_per_request / quality_score
            # Normalize based on expected cost-quality trade-off
            cost_quality_efficiency = max(0, 1 - (cost_per_quality / 0.02))
            results['cost_quality_efficiency'] = cost_quality_efficiency
        
        # Infrastructure Cost Efficiency
        infrastructure_cost = context.get('infrastructure_cost', 0.0)
        if infrastructure_cost > 0:
            requests_per_dollar = 1.0 / infrastructure_cost
            # Normalize assuming 100 requests per dollar is good
            infra_efficiency = min(requests_per_dollar / 100.0, 1.0)
            results['infrastructure_cost_efficiency'] = infra_efficiency
        
        return results
    
    async def _evaluate_energy_efficiency(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate energy efficiency metrics
        """
        results = {}
        
        # Energy per Token
        energy_consumption = context.get('energy_consumption', 0.0)  # kWh
        total_tokens = len(context.get('input_text', '').split()) + len(context.get('output_text', '').split())
        
        if total_tokens > 0 and energy_consumption > 0:
            energy_per_token = energy_consumption / total_tokens
            # Normalize assuming 0.001 kWh per token is reasonable
            energy_efficiency = max(0, 1 - (energy_per_token / 0.001))
            results['energy_per_token_efficiency'] = energy_efficiency
        
        # Carbon Footprint
        carbon_footprint = context.get('carbon_footprint', 0.0)  # kg CO2
        if carbon_footprint > 0:
            carbon_per_request = carbon_footprint
            # Normalize assuming 0.01 kg CO2 per request is reasonable
            carbon_efficiency = max(0, 1 - (carbon_per_request / 0.01))
            results['carbon_efficiency'] = carbon_efficiency
        
        # Power Usage Effectiveness (PUE)
        pue = context.get('pue', 1.5)  # Default PUE
        # Lower PUE is better, 1.0 is ideal, 2.0 is poor
        pue_efficiency = max(0, 2.0 - pue)
        results['pue_efficiency'] = pue_efficiency
        
        return results
    
    async def _evaluate_caching_efficiency(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate caching and optimization efficiency
        """
        results = {}
        
        # Cache Hit Rate
        cache_hits = context.get('cache_hits', 0)
        cache_misses = context.get('cache_misses', 1)
        total_requests = cache_hits + cache_misses
        
        hit_rate = cache_hits / total_requests if total_requests > 0 else 0
        results['cache_hit_rate'] = hit_rate
        
        # Cache Performance Improvement
        cached_response_time = context.get('cached_response_time', 0.1)
        uncached_response_time = context.get('uncached_response_time', 1.0)
        
        if uncached_response_time > 0:
            performance_improvement = 1 - (cached_response_time / uncached_response_time)
            results['cache_performance_improvement'] = max(0, performance_improvement)
        
        # Cache Memory Efficiency
        cache_memory_usage = context.get('cache_memory_usage', 0)
        available_memory = context.get('available_memory', 1)
        
        if available_memory > 0:
            memory_usage_ratio = cache_memory_usage / available_memory
            # Optimal cache usage is around 20-30% of available memory
            if 0.2 <= memory_usage_ratio <= 0.3:
                cache_memory_efficiency = 1.0
            elif memory_usage_ratio < 0.2:
                cache_memory_efficiency = memory_usage_ratio / 0.2
            else:
                cache_memory_efficiency = max(0, 1 - ((memory_usage_ratio - 0.3) / 0.7))
            
            results['cache_memory_efficiency'] = cache_memory_efficiency
        
        return results
    
    async def _evaluate_load_balancing(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate load balancing efficiency
        """
        results = {}
        
        # Load Distribution Evenness
        server_loads = context.get('server_loads', [1.0])  # List of server load percentages
        
        if len(server_loads) > 1:
            load_variance = np.var(server_loads)
            # Lower variance indicates better load distribution
            load_distribution_score = max(0, 1 - (load_variance / 0.25))
            results['load_distribution_score'] = load_distribution_score
        
        # Failover Efficiency
        failover_time = context.get('failover_time', 0.0)
        if failover_time >= 0:
            # Lower failover time is better
            failover_efficiency = max(0, 1 - (failover_time / 10.0))
            results['failover_efficiency'] = failover_efficiency
        
        # Health Check Efficiency
        health_check_overhead = context.get('health_check_overhead', 0.01)
        # Lower overhead is better
        health_check_efficiency = max(0, 1 - (health_check_overhead / 0.1))
        results['health_check_efficiency'] = health_check_efficiency
        
        return results
    
    async def _evaluate_memory_management(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate memory management efficiency
        """
        results = {}
        
        # Memory Leak Detection
        memory_growth_rate = context.get('memory_growth_rate', 0.0)
        # Negative or zero growth is good
        memory_leak_score = max(0, 1 - max(0, memory_growth_rate))
        results['memory_leak_score'] = memory_leak_score
        
        # Garbage Collection Efficiency
        gc_pause_time = context.get('gc_pause_time', 0.01)
        gc_frequency = context.get('gc_frequency', 1.0)
        
        # Lower pause time and reasonable frequency is better
        gc_efficiency = max(0, 1 - (gc_pause_time / 0.1)) * max(0, 1 - abs(gc_frequency - 1) / 2)
        results['garbage_collection_efficiency'] = gc_efficiency
        
        # Memory Fragmentation
        memory_fragmentation = context.get('memory_fragmentation', 0.1)
        fragmentation_score = max(0, 1 - (memory_fragmentation / 0.5))
        results['memory_fragmentation_score'] = fragmentation_score
        
        return results
    
    async def _evaluate_concurrency_efficiency(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate concurrency and parallelization efficiency
        """
        results = {}
        
        # Thread Utilization
        active_threads = context.get('active_threads', 1)
        available_threads = context.get('available_threads', 4)
        
        if available_threads > 0:
            thread_utilization = active_threads / available_threads
            # Optimal utilization is around 70-90%
            if 0.7 <= thread_utilization <= 0.9:
                thread_efficiency = 1.0
            elif thread_utilization < 0.7:
                thread_efficiency = thread_utilization / 0.7
            else:
                thread_efficiency = max(0, 1 - ((thread_utilization - 0.9) / 0.1))
            
            results['thread_efficiency'] = thread_efficiency
        
        # Lock Contention
        lock_contention_ratio = context.get('lock_contention_ratio', 0.0)
        lock_efficiency = max(0, 1 - (lock_contention_ratio / 0.2))
        results['lock_efficiency'] = lock_efficiency
        
        # Parallel Processing Speedup
        sequential_time = context.get('sequential_time', 1.0)
        parallel_time = context.get('parallel_time', 1.0)
        
        if sequential_time > 0:
            speedup = sequential_time / parallel_time
            # Theoretical maximum speedup equals number of cores
            cores = context.get('cpu_cores', 4)
            speedup_efficiency = min(speedup / cores, 1.0)
            results['parallel_speedup_efficiency'] = speedup_efficiency
        
        return results
    
    async def _evaluate_rate_limiting_efficiency(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate API rate limiting efficiency
        """
        results = {}
        
        # Rate Limit Adherence
        requests_made = context.get('requests_made', 1)
        rate_limit = context.get('rate_limit', 100)
        
        adherence_ratio = min(requests_made / rate_limit, 1.0)
        results['rate_limit_adherence'] = adherence_ratio
        
        # Throttling Effectiveness
        throttled_requests = context.get('throttled_requests', 0)
        total_requests = context.get('total_requests', 1)
        
        throttling_rate = throttled_requests / total_requests if total_requests > 0 else 0
        # Lower throttling rate indicates better capacity planning
        throttling_efficiency = max(0, 1 - (throttling_rate / 0.1))
        results['throttling_efficiency'] = throttling_efficiency
        
        # Burst Handling
        burst_capacity = context.get('burst_capacity', 10)
        peak_requests = context.get('peak_requests', 1)
        
        burst_handling = min(burst_capacity / peak_requests, 1.0) if peak_requests > 0 else 1.0
        results['burst_handling_efficiency'] = burst_handling
        
        return results
    
    async def _evaluate_error_handling_efficiency(self, context: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate error handling efficiency
        """
        results = {}
        
        # Error Recovery Time
        error_recovery_time = context.get('error_recovery_time', 0.0)
        if error_recovery_time >= 0:
            recovery_efficiency = max(0, 1 - (error_recovery_time / 5.0))
            results['error_recovery_efficiency'] = recovery_efficiency
        
        # Retry Logic Efficiency
        retry_attempts = context.get('retry_attempts', 0)
        successful_retries = context.get('successful_retries', 0)
        
        if retry_attempts > 0:
            retry_success_rate = successful_retries / retry_attempts
            results['retry_efficiency'] = retry_success_rate
        
        # Circuit Breaker Effectiveness
        circuit_breaker_activations = context.get('circuit_breaker_activations', 0)
        prevented_failures = context.get('prevented_failures', 0)
        
        if circuit_breaker_activations > 0:
            circuit_breaker_efficiency = prevented_failures / circuit_breaker_activations
            results['circuit_breaker_efficiency'] = min(circuit_breaker_efficiency, 1.0)
        
        return results
    
    # Helper methods
    def _estimate_information_content(self, text: str) -> float:
        """
        Estimate information content using entropy
        """
        if not text:
            return 0.0
        
        # Calculate character frequency
        char_counts = {}
        for char in text.lower():
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        total_chars = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / total_chars
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        # Normalize entropy (max entropy for English is ~4.5 bits)
        normalized_entropy = min(entropy / 4.5, 1.0)
        
        return normalized_entropy
    
    def _calculate_redundancy(self, text: str) -> float:
        """
        Calculate text redundancy score
        """
        words = text.lower().split()
        if len(words) <= 1:
            return 0.0
        
        # Count unique words
        unique_words = set(words)
        redundancy = 1.0 - (len(unique_words) / len(words))
        
        return redundancy
    
    def _calculate_semantic_density(self, text: str) -> float:
        """
        Calculate semantic density (content words vs function words)
        """
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Common function words
        function_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this',
            'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them'
        }
        
        content_words = [word for word in words if word not in function_words]
        density = len(content_words) / len(words)
        
        return density
    
    def _calculate_resource_efficiency(self, usage_percentage: float) -> float:
        """
        Calculate resource efficiency score
        Optimal usage is around 70-80%
        """
        if 70 <= usage_percentage <= 80:
            return 1.0
        elif usage_percentage < 70:
            # Underutilization
            return usage_percentage / 70.0
        else:
            # Overutilization
            return max(0, 1 - ((usage_percentage - 80) / 20.0))
    
    def _calculate_io_efficiency(self, io_metrics: Dict[str, float]) -> float:
        """
        Calculate I/O efficiency score
        """
        if not io_metrics:
            return 0.5
        
        # Example metrics: read_rate, write_rate, latency
        read_rate = io_metrics.get('read_rate', 0)
        write_rate = io_metrics.get('write_rate', 0)
        latency = io_metrics.get('latency', 1.0)
        
        # Higher throughput and lower latency is better
        throughput_score = min((read_rate + write_rate) / 1000.0, 1.0)  # MB/s
        latency_score = max(0, 1 - (latency / 10.0))  # ms
        
        return (throughput_score + latency_score) / 2


class ResourceMonitor:
    """
    System resource monitoring utility
    """
    
    def __init__(self):
        self.process = psutil.Process()
    
    def get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception as e:
            logger.warning(f"Could not get CPU usage: {e}")
            return 50.0
    
    def get_memory_usage(self) -> float:
        """Get memory usage percentage"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception as e:
            logger.warning(f"Could not get memory usage: {e}")
            return 50.0
    
    def get_gpu_usage(self) -> Optional[float]:
        """Get GPU usage percentage (if available)"""
        try:
            # This would require GPU monitoring libraries like nvidia-ml-py
            # For now, return None
            return None
        except Exception as e:
            logger.warning(f"Could not get GPU usage: {e}")
            return None
    
    def get_disk_io(self) -> Dict[str, float]:
        """Get disk I/O metrics"""
        try:
            disk_io = psutil.disk_io_counters()
            return {
                'read_rate': disk_io.read_bytes / (1024 * 1024),  # MB
                'write_rate': disk_io.write_bytes / (1024 * 1024),  # MB
                'latency': 1.0  # Placeholder
            }
        except Exception as e:
            logger.warning(f"Could not get disk I/O: {e}")
            return {'read_rate': 0, 'write_rate': 0, 'latency': 1.0}
    
    def get_network_io(self) -> Dict[str, float]:
        """Get network I/O metrics"""
        try:
            network_io = psutil.net_io_counters()
            return {
                'read_rate': network_io.bytes_recv / (1024 * 1024),  # MB
                'write_rate': network_io.bytes_sent / (1024 * 1024),  # MB
                'latency': 1.0  # Placeholder
            }
        except Exception as e:
            logger.warning(f"Could not get network I/O: {e}")
            return {'read_rate': 0, 'write_rate': 0, 'latency': 1.0}
