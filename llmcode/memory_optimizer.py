"""
Memory Optimizer Module for Llmcode

This module provides memory optimization strategies for large dependency graphs
and analysis results, enabling efficient handling of large codebases.
"""

import os
import sys
import gc
import time
import psutil
import pickle
import mmap
import json
from typing import Dict, List, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import weakref
import threading
from contextlib import contextmanager

from .static_analysis import StaticAnalyzer
from .code_structure import CodeStructureAnalyzer
from .relationship_mapper import RelationshipMapper
from .dependency_graph import DependencyGraphAnalyzer


@dataclass
class MemoryUsage:
    """Memory usage statistics"""
    current_rss: int = 0  # Current resident set size in bytes
    peak_rss: int = 0     # Peak resident set size in bytes
    current_vms: int = 0  # Current virtual memory size in bytes
    peak_vms: int = 0     # Peak virtual memory size in bytes
    shared_memory: int = 0 # Shared memory usage in bytes
    text_memory: int = 0   # Text (code) memory usage in bytes
    data_memory: int = 0   # Data memory usage in bytes
    
    def to_mb(self) -> Dict[str, float]:
        """Convert memory usage to MB"""
        return {
            'current_rss_mb': self.current_rss / (1024 * 1024),
            'peak_rss_mb': self.peak_rss / (1024 * 1024),
            'current_vms_mb': self.current_vms / (1024 * 1024),
            'peak_vms_mb': self.peak_vms / (1024 * 1024),
            'shared_memory_mb': self.shared_memory / (1024 * 1024),
            'text_memory_mb': self.text_memory / (1024 * 1024),
            'data_memory_mb': self.data_memory / (1024 * 1024),
        }


@dataclass
class MemoryStats:
    """Comprehensive memory statistics"""
    usage: MemoryUsage
    object_counts: Dict[str, int] = field(default_factory=dict)
    cache_sizes: Dict[str, int] = field(default_factory=dict)
    graph_stats: Dict[str, int] = field(default_factory=dict)
    optimization_stats: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'memory_usage': self.usage.to_mb(),
            'object_counts': self.object_counts,
            'cache_sizes': self.cache_sizes,
            'graph_stats': self.graph_stats,
            'optimization_stats': self.optimization_stats,
            'timestamp': time.time()
        }


class MemoryMonitor:
    """Monitor memory usage in real-time"""
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize memory monitor
        
        Args:
            update_interval: Interval in seconds for memory updates
        """
        self.update_interval = update_interval
        self.process = psutil.Process()
        self.initial_memory = self._get_memory_usage()
        self.peak_memory = self.initial_memory
        self.monitoring = False
        self.monitor_thread = None
        self.lock = threading.Lock()
    
    def _get_memory_usage(self) -> MemoryUsage:
        """Get current memory usage"""
        try:
            memory_info = self.process.memory_info()
            return MemoryUsage(
                current_rss=memory_info.rss,
                peak_rss=memory_info.rss,  # Will be updated by monitoring
                current_vms=memory_info.vms,
                peak_vms=memory_info.vms,  # Will be updated by monitoring
                shared_memory=getattr(memory_info, 'shared', 0),
                text_memory=getattr(memory_info, 'text', 0),
                data_memory=getattr(memory_info, 'data', 0)
            )
        except Exception:
            return MemoryUsage()
    
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                current_memory = self._get_memory_usage()
                
                with self.lock:
                    # Update peak values
                    if current_memory.current_rss > self.peak_memory.current_rss:
                        self.peak_memory.current_rss = current_memory.current_rss
                    if current_memory.current_vms > self.peak_memory.current_vms:
                        self.peak_memory.current_vms = current_memory.current_vms
                
                time.sleep(self.update_interval)
                
            except Exception:
                time.sleep(self.update_interval)
    
    def get_current_memory(self) -> MemoryUsage:
        """Get current memory usage"""
        with self.lock:
            return self._get_memory_usage()
    
    def get_peak_memory(self) -> MemoryUsage:
        """Get peak memory usage"""
        with self.lock:
            return MemoryUsage(
                current_rss=self.peak_memory.current_rss,
                peak_rss=self.peak_memory.current_rss,
                current_vms=self.peak_memory.current_vms,
                peak_vms=self.peak_memory.current_vms,
                shared_memory=self.peak_memory.shared_memory,
                text_memory=self.peak_memory.text_memory,
                data_memory=self.peak_memory.data_memory
            )


class DependencyGraphOptimizer:
    """Optimize memory usage for large dependency graphs"""
    
    def __init__(self, 
                 max_memory_mb: int = 1024,
                 compression_threshold: int = 1000,
                 enable_sparse_storage: bool = True,
                 enable_memory_mapping: bool = True):
        """
        Initialize dependency graph optimizer
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            compression_threshold: Threshold for graph compression
            enable_sparse_storage: Whether to use sparse storage for large graphs
            enable_memory_mapping: Whether to use memory mapping for large graphs
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.compression_threshold = compression_threshold
        self.enable_sparse_storage = enable_sparse_storage
        self.enable_memory_mapping = enable_memory_mapping
        
        # Memory monitoring
        self.memory_monitor = MemoryMonitor()
        
        # Graph storage
        self.graph_cache = {}
        self.compressed_graphs = {}
        self.sparse_graphs = {}
        self.mapped_graphs = {}
        
        # Optimization stats
        self.optimization_stats = {
            'compressions': 0,
            'sparsifications': 0,
            'memory_mappings': 0,
            'cache_evictions': 0,
            'memory_savings_mb': 0
        }
    
    def _check_memory_pressure(self) -> bool:
        """Check if memory usage is approaching limits"""
        current_memory = self.memory_monitor.get_current_memory()
        return current_memory.current_rss > (self.max_memory_bytes * 0.8)
    
    def _compress_graph(self, graph: Dict[str, Set[str]]) -> Dict[str, List[str]]:
        """Compress graph by converting sets to lists"""
        compressed = {}
        for node, dependencies in graph.items():
            compressed[node] = list(dependencies)
        return compressed
    
    def _decompress_graph(self, compressed_graph: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        """Decompress graph by converting lists back to sets"""
        decompressed = {}
        for node, dependencies in compressed_graph.items():
            decompressed[node] = set(dependencies)
        return decompressed
    
    def _create_sparse_graph(self, graph: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """Create sparse representation of graph"""
        if len(graph) < self.compression_threshold:
            return graph
        
        # Remove nodes with no dependencies
        sparse_graph = {}
        for node, dependencies in graph.items():
            if dependencies:  # Only keep nodes with dependencies
                sparse_graph[node] = dependencies
        
        self.optimization_stats['sparsifications'] += 1
        return sparse_graph
    
    def _memory_map_graph(self, graph_id: str, graph: Dict[str, Any]) -> bool:
        """Store graph using memory mapping"""
        if not self.enable_memory_mapping:
            return False
        
        try:
            # Create temporary file for memory mapping
            temp_file = Path(f"/tmp/llmcode_graph_{graph_id}.tmp")
            
            # Serialize graph to file
            with open(temp_file, 'wb') as f:
                pickle.dump(graph, f)
            
            # Memory map the file
            with open(temp_file, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Store the memory-mapped data
                    self.mapped_graphs[graph_id] = {
                        'file_path': temp_file,
                        'size': os.path.getsize(temp_file),
                        'mapped_time': time.time()
                    }
            
            self.optimization_stats['memory_mappings'] += 1
            return True
            
        except Exception:
            return False
    
    def _evict_cache_entries(self, count: int = 1):
        """Evict oldest cache entries to free memory"""
        if not self.graph_cache:
            return
        
        # Simple LRU eviction
        cache_items = list(self.graph_cache.items())
        for i in range(min(count, len(cache_items))):
            key, _ = cache_items[i]
            del self.graph_cache[key]
            self.optimization_stats['cache_evictions'] += 1
    
    def optimize_dependency_graph(self, 
                                graph_id: str,
                                graph: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """
        Optimize dependency graph for memory usage
        
        Args:
            graph_id: Unique identifier for the graph
            graph: Dependency graph to optimize
            
        Returns:
            Optimized graph
        """
        # Check memory pressure
        if self._check_memory_pressure():
            self._evict_cache_entries(5)  # Evict 5 entries
        
        # Start memory monitoring
        self.memory_monitor.start_monitoring()
        
        try:
            initial_memory = self.memory_monitor.get_current_memory()
            
            # Apply optimizations based on graph size
            optimized_graph = graph.copy()
            
            # Sparse storage for large graphs
            if self.enable_sparse_storage and len(graph) > self.compression_threshold:
                optimized_graph = self._create_sparse_graph(optimized_graph)
            
            # Compress if still large
            if len(optimized_graph) > self.compression_threshold:
                compressed = self._compress_graph(optimized_graph)
                self.compressed_graphs[graph_id] = compressed
                self.optimization_stats['compressions'] += 1
            
            # Memory mapping for very large graphs
            if len(optimized_graph) > self.compression_threshold * 2:
                if self._memory_map_graph(graph_id, optimized_graph):
                    # Remove from memory cache since it's memory-mapped
                    if graph_id in self.graph_cache:
                        del self.graph_cache[graph_id]
            
            # Cache the optimized graph
            self.graph_cache[graph_id] = optimized_graph
            
            # Calculate memory savings
            final_memory = self.memory_monitor.get_current_memory()
            memory_saved = initial_memory.current_rss - final_memory.current_rss
            if memory_saved > 0:
                self.optimization_stats['memory_savings_mb'] += memory_saved / (1024 * 1024)
            
            return optimized_graph
            
        finally:
            self.memory_monitor.stop_monitoring()
    
    def get_optimized_graph(self, graph_id: str) -> Optional[Dict[str, Set[str]]]:
        """Get optimized graph by ID"""
        # Check memory-mapped graphs first
        if graph_id in self.mapped_graphs:
            try:
                mapped_info = self.mapped_graphs[graph_id]
                with open(mapped_info['file_path'], 'rb') as f:
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        graph = pickle.loads(mm.read())
                        return graph
            except Exception:
                # Remove corrupted mapping
                del self.mapped_graphs[graph_id]
        
        # Check compressed graphs
        if graph_id in self.compressed_graphs:
            return self._decompress_graph(self.compressed_graphs[graph_id])
        
        # Check regular cache
        return self.graph_cache.get(graph_id)
    
    def clear_optimization_cache(self):
        """Clear all optimization caches"""
        self.graph_cache.clear()
        self.compressed_graphs.clear()
        self.sparse_graphs.clear()
        
        # Clean up memory-mapped files
        for graph_id, mapped_info in self.mapped_graphs.items():
            try:
                file_path = mapped_info['file_path']
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass
        
        self.mapped_graphs.clear()
        
        # Force garbage collection
        gc.collect()
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        current_memory = self.memory_monitor.get_current_memory()
        peak_memory = self.memory_monitor.get_peak_memory()
        
        return {
            'optimization_stats': self.optimization_stats,
            'current_memory_mb': current_memory.current_rss / (1024 * 1024),
            'peak_memory_mb': peak_memory.current_rss / (1024 * 1024),
            'cache_sizes': {
                'graph_cache': len(self.graph_cache),
                'compressed_graphs': len(self.compressed_graphs),
                'sparse_graphs': len(self.sparse_graphs),
                'mapped_graphs': len(self.mapped_graphs)
            },
            'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
            'memory_pressure': self._check_memory_pressure()
        }


class MemoryAwareAnalyzer:
    """Memory-aware analyzer that optimizes memory usage during analysis"""
    
    def __init__(self,
                 max_memory_mb: int = 1024,
                 batch_size: int = 100,
                 enable_streaming: bool = True,
                 enable_gc_optimization: bool = True):
        """
        Initialize memory-aware analyzer
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            batch_size: Number of files to process in each batch
            enable_streaming: Whether to enable streaming analysis
            enable_gc_optimization: Whether to optimize garbage collection
        """
        self.max_memory_mb = max_memory_mb
        self.batch_size = batch_size
        self.enable_streaming = enable_streaming
        self.enable_gc_optimization = enable_gc_optimization
        
        # Initialize analyzers
        self.static_analyzer = StaticAnalyzer()
        self.structure_analyzer = CodeStructureAnalyzer()
        self.relationship_mapper = RelationshipMapper()
        self.dependency_analyzer = DependencyGraphAnalyzer()
        
        # Memory optimization
        self.graph_optimizer = DependencyGraphOptimizer(max_memory_mb=max_memory_mb)
        self.memory_monitor = MemoryMonitor()
        
        # Analysis state
        self.current_batch = []
        self.batch_results = []
        self.dependency_graph = {}
        
        # Weak references for large objects
        self.weak_refs = weakref.WeakValueDictionary()
    
    def _check_memory_usage(self) -> bool:
        """Check if memory usage is within limits"""
        current_memory = self.memory_monitor.get_current_memory()
        current_mb = current_memory.current_rss / (1024 * 1024)
        return current_mb < self.max_memory_mb
    
    def _optimize_memory_usage(self):
        """Optimize memory usage"""
        if self.enable_gc_optimization:
            gc.collect()
        
        # Clear weak references that are no longer valid
        invalid_keys = [key for key, ref in self.weak_refs.items() if ref() is None]
        for key in invalid_keys:
            del self.weak_refs[key]
    
    def _process_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of files with memory optimization"""
        batch_results = []
        
        for file_path in file_paths:
            if not self._check_memory_usage():
                self._optimize_memory_usage()
                
                # If still over limit, process smaller batches
                if not self._check_memory_usage():
                    # Split batch and process recursively
                    mid = len(file_paths) // 2
                    first_half = self._process_batch(file_paths[:mid])
                    second_half = self._process_batch(file_paths[mid:])
                    return first_half + second_half
            
            try:
                # Analyze file
                result = {
                    'file_path': file_path,
                    'static': self.static_analyzer.analyze_file(file_path),
                    'structure': self.structure_analyzer.analyze_module(file_path),
                    'dependencies': self.dependency_analyzer.analyze_file(file_path),
                    'relationships': self.relationship_mapper.analyze_file(file_path)
                }
                
                batch_results.append(result)
                
                # Update dependency graph
                if 'dependencies' in result and 'imports' in result['dependencies']:
                    self.dependency_graph[file_path] = set(result['dependencies']['imports'])
                
                # Optimize dependency graph periodically
                if len(batch_results) % 10 == 0:
                    self.dependency_graph = self.graph_optimizer.optimize_dependency_graph(
                        f"batch_{len(batch_results)}",
                        self.dependency_graph
                    )
                
            except Exception as e:
                batch_results.append({
                    'file_path': file_path,
                    'error': str(e)
                })
        
        return batch_results
    
    def analyze_files_streaming(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze files with streaming to optimize memory usage
        
        Args:
            file_paths: List of file paths to analyze
            
        Returns:
            Analysis results with memory optimization
        """
        self.memory_monitor.start_monitoring()
        
        try:
            all_results = []
            
            if self.enable_streaming:
                # Process files in batches
                for i in range(0, len(file_paths), self.batch_size):
                    batch = file_paths[i:i + self.batch_size]
                    batch_results = self._process_batch(batch)
                    all_results.extend(batch_results)
                    
                    # Optimize memory between batches
                    self._optimize_memory_usage()
                    
                    # Yield intermediate results if needed
                    if i % (self.batch_size * 5) == 0:  # Every 5 batches
                        yield {
                            'batch_number': i // self.batch_size + 1,
                            'total_batches': (len(file_paths) + self.batch_size - 1) // self.batch_size,
                            'batch_results': batch_results,
                            'memory_usage': self.memory_monitor.get_current_memory().to_mb()
                        }
            else:
                # Process all files at once
                all_results = self._process_batch(file_paths)
            
            # Final optimization of dependency graph
            self.dependency_graph = self.graph_optimizer.optimize_dependency_graph(
                "final_graph",
                self.dependency_graph
            )
            
            # Compile final results
            final_results = {
                'files': all_results,
                'dependency_graph': self.dependency_graph,
                'memory_stats': self.memory_monitor.get_peak_memory().to_mb(),
                'optimization_stats': self.graph_optimizer.get_optimization_stats(),
                'total_files': len(file_paths),
                'successful_files': sum(1 for r in all_results if 'error' not in r),
                'failed_files': sum(1 for r in all_results if 'error' in r)
            }
            
            yield final_results
            
        finally:
            self.memory_monitor.stop_monitoring()
    
    def get_memory_stats(self) -> MemoryStats:
        """Get comprehensive memory statistics"""
        current_memory = self.memory_monitor.get_current_memory()
        peak_memory = self.memory_monitor.get_peak_memory()
        
        # Count objects in memory
        object_counts = {
            'dependency_graph_nodes': len(self.dependency_graph),
            'batch_results': len(self.batch_results),
            'weak_refs': len(self.weak_refs)
        }
        
        # Cache sizes
        cache_sizes = {
            'graph_cache': len(self.graph_optimizer.graph_cache),
            'compressed_graphs': len(self.graph_optimizer.compressed_graphs),
            'mapped_graphs': len(self.graph_optimizer.mapped_graphs)
        }
        
        # Graph statistics
        graph_stats = {
            'total_nodes': len(self.dependency_graph),
            'total_edges': sum(len(deps) for deps in self.dependency_graph.values()),
            'average_degree': sum(len(deps) for deps in self.dependency_graph.values()) / max(1, len(self.dependency_graph))
        }
        
        return MemoryStats(
            usage=current_memory,
            object_counts=object_counts,
            cache_sizes=cache_sizes,
            graph_stats=graph_stats,
            optimization_stats=self.graph_optimizer.optimization_stats
        )
    
    def clear_memory_cache(self):
        """Clear all memory caches"""
        self.graph_optimizer.clear_optimization_cache()
        self.dependency_graph.clear()
        self.batch_results.clear()
        self.weak_refs.clear()
        
        if self.enable_gc_optimization:
            gc.collect()
    
    @contextmanager
    def memory_limit_context(self, memory_limit_mb: int):
        """Context manager for temporary memory limit"""
        original_limit = self.max_memory_mb
        self.max_memory_mb = memory_limit_mb
        
        try:
            yield
        finally:
            self.max_memory_mb = original_limit


class MemoryOptimizationManager:
    """High-level manager for memory optimization"""
    
    def __init__(self,
                 max_memory_mb: int = 1024,
                 auto_optimize: bool = True,
                 monitoring_interval: float = 5.0):
        """
        Initialize memory optimization manager
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            auto_optimize: Whether to automatically optimize memory
            monitoring_interval: Interval for memory monitoring
        """
        self.max_memory_mb = max_memory_mb
        self.auto_optimize = auto_optimize
        self.monitoring_interval = monitoring_interval
        
        self.memory_analyzer = MemoryAwareAnalyzer(max_memory_mb=max_memory_mb)
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start background memory monitoring"""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                if self.auto_optimize:
                    current_memory = self.memory_analyzer.memory_monitor.get_current_memory()
                    current_mb = current_memory.current_rss / (1024 * 1024)
                    
                    # Auto-optimize if memory usage is high
                    if current_mb > self.max_memory_mb * 0.8:
                        self.memory_analyzer._optimize_memory_usage()
                
                time.sleep(self.monitoring_interval)
                
            except Exception:
                time.sleep(self.monitoring_interval)
    
    def analyze_with_memory_optimization(self, file_paths: List[str]) -> Dict[str, Any]:
        """
        Analyze files with memory optimization
        
        Args:
            file_paths: List of file paths to analyze
            
        Returns:
            Analysis results with memory optimization
        """
        # Start monitoring if auto-optimize is enabled
        if self.auto_optimize:
            self.start_monitoring()
        
        try:
            # Get streaming results
            results_generator = self.memory_analyzer.analyze_files_streaming(file_paths)
            
            # Collect all results
            final_result = None
            for result in results_generator:
                if 'batch_number' in result:
                    # Intermediate batch result
                    continue
                else:
                    # Final result
                    final_result = result
                    break
            
            return final_result or {}
            
        finally:
            if self.auto_optimize:
                self.stop_monitoring()
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory report"""
        memory_stats = self.memory_analyzer.get_memory_stats()
        optimization_stats = self.memory_analyzer.graph_optimizer.get_optimization_stats()
        
        return {
            'memory_stats': memory_stats.to_dict(),
            'optimization_stats': optimization_stats,
            'system_info': {
                'max_memory_mb': self.max_memory_mb,
                'auto_optimize': self.auto_optimize,
                'monitoring_active': self.monitoring
            }
        }
    
    def optimize_dependency_graph(self, graph: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
        """
        Optimize a dependency graph for memory usage
        
        Args:
            graph: Dependency graph to optimize
            
        Returns:
            Optimized graph
        """
        return self.memory_analyzer.graph_optimizer.optimize_dependency_graph(
            "user_graph",
            graph
        )
    
    def clear_all_caches(self):
        """Clear all memory caches"""
        self.memory_analyzer.clear_memory_cache()
