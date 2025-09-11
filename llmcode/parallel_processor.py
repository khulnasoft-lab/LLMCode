"""
Parallel Processing Module for Llmcode

This module provides parallel processing capabilities for analysis tasks,
enabling concurrent analysis of multiple files to improve performance.
"""

import os
import time
import multiprocessing
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Set, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from threading import Lock
import traceback

from .static_analysis import StaticAnalyzer
from .code_structure import CodeStructureAnalyzer
from .relationship_mapper import RelationshipMapper
from .dependency_graph import DependencyGraphAnalyzer
from .file_selector import IntelligentFileSelector
from .relevance_scorer import RelevanceScorer
from .context_optimizer import ContextOptimizer
from .task_filter import TaskFilter


@dataclass
class AnalysisTask:
    """Represents a single analysis task"""
    file_path: str
    analysis_types: Set[str]
    task_id: str
    priority: int = 0
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class AnalysisResult:
    """Represents the result of an analysis task"""
    task_id: str
    file_path: str
    results: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    execution_time: float = 0.0
    memory_usage: float = 0.0


@dataclass
class ProcessingStats:
    """Statistics about parallel processing"""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_time: float
    average_time_per_task: float
    peak_memory_usage: float
    cpu_utilization: float
    throughput: float  # tasks per second


class ParallelProcessor:
    """
    Parallel processor for concurrent analysis tasks
    """
    
    def __init__(self,
                 max_workers: Optional[int] = None,
                 use_processes: bool = False,
                 chunk_size: int = 1,
                 timeout: Optional[float] = None,
                 memory_limit_mb: Optional[int] = None,
                 enable_progress: bool = True):
        """
        Initialize parallel processor
        
        Args:
            max_workers: Maximum number of worker threads/processes
            use_processes: Whether to use processes instead of threads
            chunk_size: Number of tasks to process per worker
            timeout: Timeout for individual tasks in seconds
            memory_limit_mb: Memory limit in MB for the entire process
            enable_progress: Whether to show progress information
        """
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb
        self.enable_progress = enable_progress
        
        # Initialize analysis modules (will be created per worker if using processes)
        self.static_analyzer = StaticAnalyzer()
        self.structure_analyzer = CodeStructureAnalyzer()
        self.relationship_mapper = RelationshipMapper()
        self.dependency_analyzer = DependencyGraphAnalyzer()
        self.file_selector = IntelligentFileSelector()
        self.relevance_scorer = RelevanceScorer()
        self.context_optimizer = ContextOptimizer()
        self.task_filter = TaskFilter()
        
        # Thread safety
        self._lock = Lock()
        self._stats = {
            'tasks_started': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'start_time': 0,
            'end_time': 0,
            'memory_samples': []
        }
    
    def _create_worker_modules(self):
        """Create analysis modules for worker processes"""
        return {
            'static_analyzer': StaticAnalyzer(),
            'structure_analyzer': CodeStructureAnalyzer(),
            'relationship_mapper': RelationshipMapper(),
            'dependency_analyzer': DependencyGraphAnalyzer(),
            'file_selector': IntelligentFileSelector(),
            'relevance_scorer': RelevanceScorer(),
            'context_optimizer': ContextOptimizer(),
            'task_filter': TaskFilter(),
        }
    
    def _analyze_file_worker(self, task: AnalysisTask, modules: Dict[str, Any]) -> AnalysisResult:
        """Worker function to analyze a single file"""
        start_time = time.time()
        
        try:
            results = {}
            
            # Static analysis
            if 'static' in task.analysis_types:
                results['static'] = modules['static_analyzer'].analyze_file(task.file_path)
            
            # Structure analysis
            if 'structure' in task.analysis_types:
                results['structure'] = modules['structure_analyzer'].analyze_module(task.file_path)
            
            # Dependency analysis
            if 'dependency' in task.analysis_types:
                results['dependencies'] = modules['dependency_analyzer'].analyze_file(task.file_path)
            
            # Relationship analysis
            if 'relationship' in task.analysis_types:
                results['relationships'] = modules['relationship_mapper'].analyze_file(task.file_path)
            
            # Relevance scoring
            if 'relevance' in task.analysis_types:
                results['relevance'] = modules['relevance_scorer'].score_file(task.file_path)
            
            # Context optimization
            if 'context' in task.analysis_types:
                results['context'] = modules['context_optimizer'].optimize_file_context(task.file_path)
            
            # Task filtering
            if 'filter' in task.analysis_types:
                results['filter'] = modules['task_filter'].filter_file_content(task.file_path)
            
            execution_time = time.time() - start_time
            
            return AnalysisResult(
                task_id=task.task_id,
                file_path=task.file_path,
                results=results,
                success=True,
                execution_time=execution_time,
                memory_usage=0  # Would need psutil for accurate memory tracking
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"{str(e)}\\n{traceback.format_exc()}"
            
            return AnalysisResult(
                task_id=task.task_id,
                file_path=task.file_path,
                results={},
                success=False,
                error=error_msg,
                execution_time=execution_time,
                memory_usage=0
            )
    
    def _process_task_thread(self, task: AnalysisTask) -> AnalysisResult:
        """Process a task in the current thread (for thread-based execution)"""
        modules = {
            'static_analyzer': self.static_analyzer,
            'structure_analyzer': self.structure_analyzer,
            'relationship_mapper': self.relationship_mapper,
            'dependency_analyzer': self.dependency_analyzer,
            'file_selector': self.file_selector,
            'relevance_scorer': self.relevance_scorer,
            'context_optimizer': self.context_optimizer,
            'task_filter': self.task_filter,
        }
        
        return self._analyze_file_worker(task, modules)
    
    def _process_task_process(self, task: AnalysisTask) -> AnalysisResult:
        """Process a task in a separate process"""
        modules = self._create_worker_modules()
        return self._analyze_file_worker(task, modules)
    
    def _update_stats(self, result: AnalysisResult):
        """Update processing statistics"""
        with self._lock:
            if result.success:
                self._stats['tasks_completed'] += 1
            else:
                self._stats['tasks_failed'] += 1
            
            # Sample memory usage (would need psutil for accurate tracking)
            self._stats['memory_samples'].append(result.memory_usage)
    
    def _log_progress(self, completed: int, total: int, current_file: str = ""):
        """Log progress information"""
        if not self.enable_progress:
            return
        
        progress = (completed / total) * 100
        print(f"Progress: {completed}/{total} ({progress:.1f}%) - {current_file}")
    
    def process_tasks(self, tasks: List[AnalysisTask]) -> Tuple[List[AnalysisResult], ProcessingStats]:
        """
        Process multiple analysis tasks in parallel
        
        Args:
            tasks: List of analysis tasks to process
            
        Returns:
            Tuple of (results, processing_stats)
        """
        if not tasks:
            return [], ProcessingStats(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Sort tasks by priority
        tasks.sort(key=lambda x: x.priority, reverse=True)
        
        # Initialize stats
        self._stats['start_time'] = time.time()
        self._stats['tasks_started'] = len(tasks)
        
        results = []
        
        # Choose executor type
        ExecutorClass = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor
        
        try:
            with ExecutorClass(max_workers=self.max_workers) as executor:
                # Submit all tasks
                future_to_task = {}
                
                for task in tasks:
                    if self.use_processes:
                        future = executor.submit(self._process_task_process, task)
                    else:
                        future = executor.submit(self._process_task_thread, task)
                    
                    future_to_task[future] = task
                
                # Process completed tasks
                completed_count = 0
                for future in as_completed(future_to_task, timeout=self.timeout):
                    task = future_to_task[future]
                    
                    try:
                        result = future.result()
                        results.append(result)
                        self._update_stats(result)
                        
                        completed_count += 1
                        self._log_progress(completed_count, len(tasks), task.file_path)
                        
                    except Exception as e:
                        # Handle task execution errors
                        error_result = AnalysisResult(
                            task_id=task.task_id,
                            file_path=task.file_path,
                            results={},
                            success=False,
                            error=f"Task execution failed: {str(e)}",
                            execution_time=0,
                            memory_usage=0
                        )
                        results.append(error_result)
                        self._update_stats(error_result)
                        
                        completed_count += 1
                        self._log_progress(completed_count, len(tasks), task.file_path)
        
        except Exception as e:
            print(f"Parallel processing failed: {e}")
            # Fallback to sequential processing
            results = self._process_sequential(tasks)
        
        # Calculate final stats
        self._stats['end_time'] = time.time()
        stats = self._calculate_processing_stats()
        
        return results, stats
    
    def _process_sequential(self, tasks: List[AnalysisTask]) -> List[AnalysisResult]:
        """Fallback sequential processing"""
        results = []
        
        for i, task in enumerate(tasks):
            self._log_progress(i + 1, len(tasks), task.file_path)
            
            if self.use_processes:
                result = self._process_task_process(task)
            else:
                result = self._process_task_thread(task)
            
            results.append(result)
            self._update_stats(result)
        
        return results
    
    def _calculate_processing_stats(self) -> ProcessingStats:
        """Calculate processing statistics"""
        total_time = self._stats['end_time'] - self._stats['start_time']
        
        if self._stats['tasks_completed'] > 0:
            avg_time_per_task = total_time / self._stats['tasks_completed']
        else:
            avg_time_per_task = 0
        
        if total_time > 0:
            throughput = self._stats['tasks_completed'] / total_time
        else:
            throughput = 0
        
        # Calculate peak memory usage
        peak_memory = max(self._stats['memory_samples']) if self._stats['memory_samples'] else 0
        
        # Estimate CPU utilization (simplified)
        cpu_utilization = min(100.0, (self._stats['tasks_completed'] / max(1, self.max_workers)) * 100)
        
        return ProcessingStats(
            total_tasks=self._stats['tasks_started'],
            completed_tasks=self._stats['tasks_completed'],
            failed_tasks=self._stats['tasks_failed'],
            total_time=total_time,
            average_time_per_task=avg_time_per_task,
            peak_memory_usage=peak_memory,
            cpu_utilization=cpu_utilization,
            throughput=throughput
        )
    
    def process_files(self,
                     file_paths: List[str],
                     analysis_types: Set[str],
                     priority_map: Optional[Dict[str, int]] = None) -> Tuple[Dict[str, AnalysisResult], ProcessingStats]:
        """
        Process multiple files with parallel analysis
        
        Args:
            file_paths: List of file paths to analyze
            analysis_types: Types of analysis to perform
            priority_map: Optional mapping of file paths to priorities
            
        Returns:
            Tuple of (results_by_file, processing_stats)
        """
        # Create tasks
        tasks = []
        for i, file_path in enumerate(file_paths):
            task = AnalysisTask(
                file_path=file_path,
                analysis_types=analysis_types,
                task_id=f"task_{i}",
                priority=priority_map.get(file_path, 0) if priority_map else 0
            )
            tasks.append(task)
        
        # Process tasks
        results, stats = self.process_tasks(tasks)
        
        # Convert to file-based mapping
        results_by_file = {result.file_path: result for result in results}
        
        return results_by_file, stats
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information for parallel processing"""
        return {
            'cpu_count': multiprocessing.cpu_count(),
            'max_workers': self.max_workers,
            'use_processes': self.use_processes,
            'chunk_size': self.chunk_size,
            'timeout': self.timeout,
            'memory_limit_mb': self.memory_limit_mb,
            'enable_progress': self.enable_progress
        }
    
    def estimate_processing_time(self, num_files: int, avg_file_size_kb: float = 10) -> float:
        """
        Estimate processing time for a given number of files
        
        Args:
            num_files: Number of files to process
            avg_file_size_kb: Average file size in KB
            
        Returns:
            Estimated processing time in seconds
        """
        # Base time per file (empirical estimate)
        base_time_per_file = 0.1  # 100ms per file for small files
        
        # Size factor (larger files take longer)
        size_factor = max(1.0, avg_file_size_kb / 10.0)
        
        # Parallel speedup factor
        speedup_factor = min(self.max_workers, num_files) / max(1, num_files)
        
        # Estimated time
        estimated_time = (num_files * base_time_per_file * size_factor) / speedup_factor
        
        return estimated_time


class ParallelAnalysisManager:
    """
    High-level manager for parallel analysis operations
    """
    
    def __init__(self,
                 max_workers: Optional[int] = None,
                 use_processes: bool = False,
                 auto_tune: bool = True):
        """
        Initialize parallel analysis manager
        
        Args:
            max_workers: Maximum number of workers
            use_processes: Whether to use processes
            auto_tune: Whether to automatically tune parameters
        """
        self.auto_tune = auto_tune
        
        if auto_tune:
            max_workers = self._auto_tune_workers(max_workers, use_processes)
        
        self.processor = ParallelProcessor(
            max_workers=max_workers,
            use_processes=use_processes,
            enable_progress=True
        )
    
    def _auto_tune_workers(self, max_workers: Optional[int], use_processes: bool) -> int:
        """Automatically tune the number of workers"""
        cpu_count = multiprocessing.cpu_count()
        
        if use_processes:
            # For CPU-bound tasks, use all cores
            return max_workers or cpu_count
        else:
            # For I/O-bound tasks, use more workers than cores
            return max_workers or min(cpu_count * 2, 16)
    
    def analyze_repository(self,
                          root_dir: str,
                          file_patterns: List[str] = None,
                          analysis_types: Set[str] = None,
                          exclude_patterns: List[str] = None) -> Tuple[Dict[str, Any], ProcessingStats]:
        """
        Analyze an entire repository in parallel
        
        Args:
            root_dir: Root directory of the repository
            file_patterns: List of file patterns to include
            analysis_types: Types of analysis to perform
            exclude_patterns: List of patterns to exclude
            
        Returns:
            Tuple of (analysis_results, processing_stats)
        """
        if analysis_types is None:
            analysis_types = {'static', 'structure', 'dependency', 'relationship'}
        
        if file_patterns is None:
            file_patterns = ['*.py', '*.js', '*.ts', '*.java', '*.cpp', '*.c']
        
        if exclude_patterns is None:
            exclude_patterns = ['*.min.js', '*.pyc', '__pycache__', 'node_modules', '.git']
        
        # Find files to analyze
        file_paths = self._find_files(root_dir, file_patterns, exclude_patterns)
        
        if not file_paths:
            return {}, ProcessingStats(0, 0, 0, 0, 0, 0, 0, 0)
        
        # Create priority map based on file size and importance
        priority_map = self._create_priority_map(file_paths)
        
        # Process files
        results_by_file, stats = self.processor.process_files(
            file_paths=file_paths,
            analysis_types=analysis_types,
            priority_map=priority_map
        )
        
        # Aggregate results
        aggregated_results = self._aggregate_results(results_by_file)
        
        return aggregated_results, stats
    
    def _find_files(self, root_dir: str, file_patterns: List[str], exclude_patterns: List[str]) -> List[str]:
        """Find files matching patterns"""
        import fnmatch
        
        file_paths = []
        exclude_patterns = exclude_patterns or []
        
        for root, dirs, files in os.walk(root_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pattern) for pattern in exclude_patterns)]
            
            for file in files:
                file_path = os.path.join(root, file)
                
                # Check if file matches any pattern
                if any(fnmatch.fnmatch(file, pattern) for pattern in file_patterns):
                    # Check if file is not excluded
                    if not any(fnmatch.fnmatch(file_path, pattern) for pattern in exclude_patterns):
                        file_paths.append(file_path)
        
        return file_paths
    
    def _create_priority_map(self, file_paths: List[str]) -> Dict[str, int]:
        """Create priority map for files"""
        priority_map = {}
        
        for file_path in file_paths:
            priority = 0
            
            # Higher priority for certain file types
            if file_path.endswith('.py'):
                priority += 2
            elif file_path.endswith(('.js', '.ts')):
                priority += 1
            
            # Higher priority for main files
            filename = os.path.basename(file_path)
            if filename in ['main.py', 'index.js', 'app.py', '__init__.py']:
                priority += 3
            
            # Higher priority for smaller files (faster to process)
            try:
                file_size = os.path.getsize(file_path)
                if file_size < 1024:  # < 1KB
                    priority += 2
                elif file_size < 10240:  # < 10KB
                    priority += 1
            except OSError:
                pass
            
            priority_map[file_path] = priority
        
        return priority_map
    
    def _aggregate_results(self, results_by_file: Dict[str, AnalysisResult]) -> Dict[str, Any]:
        """Aggregate analysis results from multiple files"""
        aggregated = {
            'files': {},
            'summary': {
                'total_files': len(results_by_file),
                'successful_files': sum(1 for r in results_by_file.values() if r.success),
                'failed_files': sum(1 for r in results_by_file.values() if not r.success),
                'total_execution_time': sum(r.execution_time for r in results_by_file.values()),
                'average_execution_time': sum(r.execution_time for r in results_by_file.values()) / max(1, len(results_by_file))
            },
            'aggregated_analysis': {}
        }
        
        # Collect individual file results
        for file_path, result in results_by_file.items():
            aggregated['files'][file_path] = {
                'success': result.success,
                'execution_time': result.execution_time,
                'error': result.error,
                'results': result.results
            }
        
        # Aggregate analysis types
        analysis_types = set()
        for result in results_by_file.values():
            if result.success:
                analysis_types.update(result.results.keys())
        
        for analysis_type in analysis_types:
            type_results = []
            for result in results_by_file.values():
                if result.success and analysis_type in result.results:
                    type_results.append(result.results[analysis_type])
            
            aggregated['aggregated_analysis'][analysis_type] = {
                'file_count': len(type_results),
                'results': type_results
            }
        
        return aggregated
