"""
Task-Aware Context Filtering for LLMCode

This module provides advanced task-aware context filtering capabilities including:
- Task-specific context filtering
- Dynamic context prioritization
- Context relevance prediction
- Multi-stage filtering pipeline
- Adaptive threshold adjustment
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import networkx as nx
from enum import Enum
import json
from datetime import datetime
import math
import heapq

from llmcode.static_analysis import StaticAnalyzer, CodeEntity
from llmcode.code_structure import CodeStructureAnalyzer
from llmcode.relationship_mapper import RelationshipMapper, RelationshipType
from llmcode.dependency_graph import DependencyGraphGenerator, DependencyType
from llmcode.file_selector import IntelligentFileSelector, TaskType, FileSelection
from llmcode.relevance_scorer import RelevanceScorer, RelevanceScore, CodeSection, ScoringDimension
from llmcode.context_optimizer import ContextOptimizer, ContextWindow, OptimizationConfig, OptimizationStrategy
from llmcode.dump import dump


class FilterStage(Enum):
    """Stages of the filtering pipeline"""
    PRE_FILTERING = "pre_filtering"
    RELEVANCE_SCORING = "relevance_scoring"
    TASK_AWARE_FILTERING = "task_aware_filtering"
    CONTEXT_OPTIMIZATION = "context_optimization"
    POST_FILTERING = "post_filtering"


class FilterType(Enum):
    """Types of filters"""
    FILE_TYPE = "file_type"
    CODE_PATTERN = "code_pattern"
    DEPENDENCY = "dependency"
    SEMANTIC = "semantic"
    STRUCTURAL = "structural"
    TEMPORAL = "temporal"
    QUALITY = "quality"
    TASK_SPECIFIC = "task_specific"


@dataclass
class FilterCriteria:
    """Criteria for filtering context"""
    filter_type: FilterType
    pattern: Optional[str] = None
    threshold: float = 0.5
    enabled: bool = True
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterResult:
    """Result of filtering operation"""
    filtered_items: List[Any]
    removed_items: List[Any]
    filter_stats: Dict[str, Any]
    execution_time: float
    confidence: float


@dataclass
class TaskProfile:
    """Profile for a specific task type"""
    task_type: TaskType
    priority_filters: List[FilterCriteria] = field(default_factory=list)
    secondary_filters: List[FilterCriteria] = field(default_factory=list)
    context_requirements: Dict[str, Any] = field(default_factory=dict)
    optimization_preferences: Dict[str, Any] = field(default_factory=dict)
    relevance_thresholds: Dict[str, float] = field(default_factory=dict)


class TaskAwareFilter:
    """Advanced task-aware context filtering"""
    
    def __init__(self, root_path: str, static_analyzer: Optional[StaticAnalyzer] = None,
                 structure_analyzer: Optional[CodeStructureAnalyzer] = None,
                 relationship_mapper: Optional[RelationshipMapper] = None,
                 dependency_graph: Optional[DependencyGraphGenerator] = None,
                 file_selector: Optional[IntelligentFileSelector] = None,
                 relevance_scorer: Optional[RelevanceScorer] = None,
                 context_optimizer: Optional[ContextOptimizer] = None,
                 verbose: bool = False):
        self.root_path = Path(root_path)
        self.verbose = verbose
        
        # Initialize analyzers
        self.static_analyzer = static_analyzer or StaticAnalyzer(root_path, verbose)
        self.structure_analyzer = structure_analyzer or CodeStructureAnalyzer(root_path, self.static_analyzer, verbose)
        self.relationship_mapper = relationship_mapper or RelationshipMapper(root_path, self.static_analyzer, self.structure_analyzer, verbose)
        self.dependency_graph = dependency_graph or DependencyGraphGenerator(root_path, self.static_analyzer, verbose)
        self.file_selector = file_selector or IntelligentFileSelector(root_path, self.static_analyzer, self.structure_analyzer, self.relationship_mapper, self.dependency_graph, verbose)
        self.relevance_scorer = relevance_scorer or RelevanceScorer(root_path, self.static_analyzer, self.structure_analyzer, self.relationship_mapper, self.dependency_graph, self.file_selector, verbose)
        self.context_optimizer = context_optimizer or ContextOptimizer(root_path, self.static_analyzer, self.structure_analyzer, self.relationship_mapper, self.dependency_graph, self.file_selector, self.relevance_scorer, verbose)
        
        # Task profiles
        self.task_profiles = self._create_task_profiles()
        
        # Filter history
        self.filter_history: List[Dict[str, Any]] = []
        
        # Initialize analyzers if needed
        self._initialize_analyzers()
        
    def _initialize_analyzers(self):
        """Initialize all analyzers if they haven't been run"""
        if not self.static_analyzer.entities:
            self.static_analyzer.analyze_project()
            
        if not self.structure_analyzer.modules:
            self.structure_analyzer.analyze_code_structure()
            
        if not self.relationship_mapper.relationships:
            self.relationship_mapper.map_relationships()
            
        if not self.dependency_graph.dependency_nodes:
            self.dependency_graph.generate_dependency_graph()
            
    def _create_task_profiles(self) -> Dict[TaskType, TaskProfile]:
        """Create profiles for different task types"""
        profiles = {}
        
        # Code Review Profile
        profiles[TaskType.CODE_REVIEW] = TaskProfile(
            task_type=TaskType.CODE_REVIEW,
            priority_filters=[
                FilterCriteria(FilterType.FILE_TYPE, pattern=r'\.(py|js|ts|java|cpp|c)$', threshold=0.8),
                FilterCriteria(FilterType.CODE_PATTERN, pattern=r'(class|function|def|method)', threshold=0.7),
                FilterCriteria(FilterType.STRUCTURAL, pattern=r'(complex|large|important)', threshold=0.6),
                FilterCriteria(FilterType.QUALITY, pattern=r'(documentation|test)', threshold=0.5)
            ],
            secondary_filters=[
                FilterCriteria(FilterType.SEMANTIC, pattern=r'(code|review|quality)', threshold=0.4),
                FilterCriteria(FilterType.DEPENDENCY, threshold=0.3)
            ],
            context_requirements={
                'include_imports': True,
                'include_definitions': True,
                'include_documentation': True,
                'include_tests': True,
                'max_files': 15,
                'max_context_size': 8000
            },
            optimization_preferences={
                'strategy': OptimizationStrategy.PRIORITY_BASED,
                'compression_level': 'minimal',
                'prioritize_quality': True
            },
            relevance_thresholds={
                'overall': 0.6,
                'structural_importance': 0.7,
                'documentation_quality': 0.6
            }
        )
        
        # Bug Fix Profile
        profiles[TaskType.BUG_FIX] = TaskProfile(
            task_type=TaskType.BUG_FIX,
            priority_filters=[
                FilterCriteria(FilterType.FILE_TYPE, pattern=r'\.(py|js|ts|java|cpp|c)$', threshold=0.9),
                FilterCriteria(FilterType.CODE_PATTERN, pattern=r'(error|exception|bug|fix|issue|debug)', threshold=0.8),
                FilterCriteria(FilterType.STRUCTURAL, pattern=r'(complex|critical|core)', threshold=0.7),
                FilterCriteria(FilterType.DEPENDENCY, threshold=0.6)
            ],
            secondary_filters=[
                FilterCriteria(FilterType.SEMANTIC, pattern=r'(bug|error|exception|fix)', threshold=0.5),
                FilterCriteria(FilterType.TEMPORAL, pattern=r'(recent|changed|modified)', threshold=0.4)
            ],
            context_requirements={
                'include_imports': True,
                'include_definitions': True,
                'include_error_handling': True,
                'include_tests': True,
                'max_files': 10,
                'max_context_size': 6000
            },
            optimization_preferences={
                'strategy': OptimizationStrategy.DEPENDENCY_BASED,
                'compression_level': 'moderate',
                'prioritize_dependencies': True
            },
            relevance_thresholds={
                'overall': 0.7,
                'bug_density': 0.8,
                'dependency_centrality': 0.7
            }
        )
        
        # Feature Development Profile
        profiles[TaskType.FEATURE_DEVELOPMENT] = TaskProfile(
            task_type=TaskType.FEATURE_DEVELOPMENT,
            priority_filters=[
                FilterCriteria(FilterType.FILE_TYPE, pattern=r'\.(py|js|ts|java|cpp|c)$', threshold=0.8),
                FilterCriteria(FilterType.CODE_PATTERN, pattern=r'(class|function|def|interface|abstract)', threshold=0.7),
                FilterCriteria(FilterType.STRUCTURAL, pattern=r'(core|base|main|service)', threshold=0.6),
                FilterCriteria(FilterType.DEPENDENCY, threshold=0.5)
            ],
            secondary_filters=[
                FilterCriteria(FilterType.SEMANTIC, pattern=r'(feature|new|add|create|implement)', threshold=0.4),
                FilterCriteria(FilterType.TASK_SPECIFIC, pattern=r'(api|endpoint|controller|model)', threshold=0.3)
            ],
            context_requirements={
                'include_imports': True,
                'include_definitions': True,
                'include_interfaces': True,
                'include_architecture': True,
                'max_files': 20,
                'max_context_size': 10000
            },
            optimization_preferences={
                'strategy': OptimizationStrategy.HIERARCHICAL,
                'compression_level': 'minimal',
                'prioritize_structure': True
            },
            relevance_thresholds={
                'overall': 0.6,
                'structural_importance': 0.8,
                'task_relevance': 0.7
            }
        )
        
        # Refactoring Profile
        profiles[TaskType.REFACTORING] = TaskProfile(
            task_type=TaskType.REFACTORING,
            priority_filters=[
                FilterCriteria(FilterType.FILE_TYPE, pattern=r'\.(py|js|ts|java|cpp|c)$', threshold=0.9),
                FilterCriteria(FilterType.CODE_PATTERN, pattern=r'(class|function|def|method)', threshold=0.8),
                FilterCriteria(FilterType.STRUCTURAL, pattern=r'(complex|large|legacy|spaghetti)', threshold=0.7),
                FilterCriteria(FilterType.QUALITY, pattern=r'(technical_debt|code_smell)', threshold=0.6)
            ],
            secondary_filters=[
                FilterCriteria(FilterType.SEMANTIC, pattern=r'(refactor|improve|optimize|cleanup)', threshold=0.5),
                FilterCriteria(FilterType.DEPENDENCY, threshold=0.4)
            ],
            context_requirements={
                'include_imports': True,
                'include_definitions': True,
                'include_dependencies': True,
                'include_complexity_metrics': True,
                'max_files': 12,
                'max_context_size': 8000
            },
            optimization_preferences={
                'strategy': OptimizationStrategy.HYBRID,
                'compression_level': 'moderate',
                'prioritize_complexity': True
            },
            relevance_thresholds={
                'overall': 0.7,
                'code_complexity': 0.8,
                'structural_importance': 0.7
            }
        )
        
        # Documentation Profile
        profiles[TaskType.DOCUMENTATION] = TaskProfile(
            task_type=TaskType.DOCUMENTATION,
            priority_filters=[
                FilterCriteria(FilterType.FILE_TYPE, pattern=r'\.(py|js|ts|java|cpp|c|md|rst)$', threshold=0.7),
                FilterCriteria(FilterType.CODE_PATTERN, pattern=r'(class|function|def|method)', threshold=0.6),
                FilterCriteria(FilterType.QUALITY, pattern=r'(documentation|docstring|comment)', threshold=0.8),
                FilterCriteria(FilterType.SEMANTIC, pattern=r'(doc|comment|explain|describe)', threshold=0.7)
            ],
            secondary_filters=[
                FilterCriteria(FilterType.FILE_TYPE, pattern=r'\.(md|rst|txt)$', threshold=0.5),
                FilterCriteria(FilterType.STRUCTURAL, pattern=r'(public|api|interface)', threshold=0.4)
            ],
            context_requirements={
                'include_imports': False,
                'include_definitions': True,
                'include_documentation': True,
                'include_examples': True,
                'max_files': 25,
                'max_context_size': 12000
            },
            optimization_preferences={
                'strategy': OptimizationStrategy.SEMANTIC_BASED,
                'compression_level': 'none',
                'prioritize_documentation': True
            },
            relevance_thresholds={
                'overall': 0.5,
                'documentation_quality': 0.8,
                'semantic_similarity': 0.7
            }
        )
        
        # Testing Profile
        profiles[TaskType.TESTING] = TaskProfile(
            task_type=TaskType.TESTING,
            priority_filters=[
                FilterCriteria(FilterType.FILE_TYPE, pattern=r'\.(py|js|ts|java|cpp|c)$', threshold=0.8),
                FilterCriteria(FilterType.CODE_PATTERN, pattern=r'(test|spec|mock|fixture)', threshold=0.9),
                FilterCriteria(FilterType.FILE_TYPE, pattern=r'test_.*|.*_test\.|.*test\.', threshold=0.8),
                FilterCriteria(FilterType.QUALITY, pattern=r'(assert|verify|validate)', threshold=0.7)
            ],
            secondary_filters=[
                FilterCriteria(FilterType.SEMANTIC, pattern=r'(test|assert|verify|validate)', threshold=0.6),
                FilterCriteria(FilterType.DEPENDENCY, threshold=0.5)
            ],
            context_requirements={
                'include_imports': True,
                'include_definitions': True,
                'include_tests': True,
                'include_test_data': True,
                'max_files': 15,
                'max_context_size': 8000
            },
            optimization_preferences={
                'strategy': OptimizationStrategy.SEMANTIC_BASED,
                'compression_level': 'minimal',
                'prioritize_tests': True
            },
            relevance_thresholds={
                'overall': 0.6,
                'test_coverage': 0.8,
                'task_relevance': 0.7
            }
        )
        
        # Performance Optimization Profile
        profiles[TaskType.PERFORMANCE_OPTIMIZATION] = TaskProfile(
            task_type=TaskType.PERFORMANCE_OPTIMIZATION,
            priority_filters=[
                FilterCriteria(FilterType.FILE_TYPE, pattern=r'\.(py|js|ts|java|cpp|c)$', threshold=0.9),
                FilterCriteria(FilterType.CODE_PATTERN, pattern=r'(loop|algorithm|data_structure|cache)', threshold=0.8),
                FilterCriteria(FilterType.STRUCTURAL, pattern=r'(complex|performance|critical|hot_path)', threshold=0.7),
                FilterCriteria(FilterType.QUALITY, pattern=r'(performance|speed|memory|cpu)', threshold=0.6)
            ],
            secondary_filters=[
                FilterCriteria(FilterType.SEMANTIC, pattern=r'(performance|optimize|speed|efficient)', threshold=0.5),
                FilterCriteria(FilterType.TEMPORAL, pattern=r'(frequent|called|executed)', threshold=0.4)
            ],
            context_requirements={
                'include_imports': True,
                'include_definitions': True,
                'include_algorithms': True,
                'include_performance_metrics': True,
                'max_files': 10,
                'max_context_size': 6000
            },
            optimization_preferences={
                'strategy': OptimizationStrategy.PRIORITY_BASED,
                'compression_level': 'moderate',
                'prioritize_performance': True
            },
            relevance_thresholds={
                'overall': 0.7,
                'code_complexity': 0.8,
                'structural_importance': 0.7
            }
        )
        
        # Security Audit Profile
        profiles[TaskType.SECURITY_AUDIT] = TaskProfile(
            task_type=TaskType.SECURITY_AUDIT,
            priority_filters=[
                FilterCriteria(FilterType.FILE_TYPE, pattern=r'\.(py|js|ts|java|cpp|c)$', threshold=0.9),
                FilterCriteria(FilterType.CODE_PATTERN, pattern=r'(auth|security|validation|sanitization)', threshold=0.8),
                FilterCriteria(FilterType.CODE_PATTERN, pattern=r'(password|token|key|secret|credential)', threshold=0.9),
                FilterCriteria(FilterType.STRUCTURAL, pattern=r'(input|validation|security)', threshold=0.7)
            ],
            secondary_filters=[
                FilterCriteria(FilterType.SEMANTIC, pattern=r'(security|safe|validate|sanitize)', threshold=0.6),
                FilterCriteria(FilterType.DEPENDENCY, threshold=0.5)
            ],
            context_requirements={
                'include_imports': True,
                'include_definitions': True,
                'include_security_code': True,
                'include_validation': True,
                'max_files': 8,
                'max_context_size': 5000
            },
            optimization_preferences={
                'strategy': OptimizationStrategy.DEPENDENCY_BASED,
                'compression_level': 'minimal',
                'prioritize_security': True
            },
            relevance_thresholds={
                'overall': 0.8,
                'dependency_centrality': 0.7,
                'task_relevance': 0.8
            }
        )
        
        # Dependency Update Profile
        profiles[TaskType.DEPENDENCY_UPDATE] = TaskProfile(
            task_type=TaskType.DEPENDENCY_UPDATE,
            priority_filters=[
                FilterCriteria(FilterType.FILE_TYPE, pattern=r'\.(py|js|ts|java|cpp|c)$', threshold=0.9),
                FilterCriteria(FilterType.CODE_PATTERN, pattern=r'(import|require|include|using)', threshold=0.9),
                FilterCriteria(FilterType.DEPENDENCY, threshold=0.8),
                FilterCriteria(FilterType.FILE_TYPE, pattern=r'requirements\.txt|package\.json|pom\.xml|build\.gradle', threshold=0.9)
            ],
            secondary_filters=[
                FilterCriteria(FilterType.SEMANTIC, pattern=r'(dependency|package|library|module)', threshold=0.6),
                FilterCriteria(FilterType.STRUCTURAL, pattern=r'(configuration|setup|build)', threshold=0.5)
            ],
            context_requirements={
                'include_imports': True,
                'include_dependencies': True,
                'include_configuration': True,
                'include_build_files': True,
                'max_files': 12,
                'max_context_size': 6000
            },
            optimization_preferences={
                'strategy': OptimizationStrategy.DEPENDENCY_BASED,
                'compression_level': 'minimal',
                'prioritize_dependencies': True
            },
            relevance_thresholds={
                'overall': 0.7,
                'dependency_centrality': 0.9,
                'task_relevance': 0.8
            }
        )
        
        # Architecture Analysis Profile
        profiles[TaskType.ARCHITECTURE_ANALYSIS] = TaskProfile(
            task_type=TaskType.ARCHITECTURE_ANALYSIS,
            priority_filters=[
                FilterCriteria(FilterType.FILE_TYPE, pattern=r'\.(py|js|ts|java|cpp|c)$', threshold=0.8),
                FilterCriteria(FilterType.CODE_PATTERN, pattern=r'(class|interface|abstract|base)', threshold=0.8),
                FilterCriteria(FilterType.STRUCTURAL, pattern=r'(architecture|design|pattern|structure)', threshold=0.7),
                FilterCriteria(FilterType.DEPENDENCY, threshold=0.6)
            ],
            secondary_filters=[
                FilterCriteria(FilterType.SEMANTIC, pattern=r'(architecture|design|pattern|structure)', threshold=0.5),
                FilterCriteria(FilterType.FILE_TYPE, pattern=r'config|setup|main', threshold=0.4)
            ],
            context_requirements={
                'include_imports': True,
                'include_definitions': True,
                'include_architecture': True,
                'include_patterns': True,
                'max_files': 20,
                'max_context_size': 10000
            },
            optimization_preferences={
                'strategy': OptimizationStrategy.HIERARCHICAL,
                'compression_level': 'minimal',
                'prioritize_structure': True
            },
            relevance_thresholds={
                'overall': 0.6,
                'structural_importance': 0.9,
                'dependency_centrality': 0.7
            }
        )
        
        return profiles
        
    def filter_context(self, file_paths: List[str], task_type: TaskType, 
                      query: Optional[str] = None,
                      custom_filters: Optional[List[FilterCriteria]] = None) -> FilterResult:
        """Filter context based on task type and query"""
        start_time = datetime.now()
        
        # Get task profile
        task_profile = self.task_profiles.get(task_type, self.task_profiles[TaskType.CODE_REVIEW])
        
        # Apply custom filters if provided
        if custom_filters:
            task_profile.priority_filters.extend(custom_filters)
            
        # Multi-stage filtering pipeline
        filtered_items = file_paths.copy()
        removed_items = []
        filter_stats = {}
        
        # Stage 1: Pre-filtering
        filtered_items, stage1_removed, stage1_stats = self._apply_pre_filtering(
            filtered_items, task_profile, query
        )
        removed_items.extend(stage1_removed)
        filter_stats['pre_filtering'] = stage1_stats
        
        # Stage 2: Relevance scoring
        scored_items = self._apply_relevance_scoring(filtered_items, task_type, query)
        filter_stats['relevance_scoring'] = {
            'total_scored': len(scored_items),
            'average_score': sum(score.relevance_score for score in scored_items) / len(scored_items) if scored_items else 0
        }
        
        # Stage 3: Task-aware filtering
        filtered_items, stage3_removed, stage3_stats = self._apply_task_aware_filtering(
            scored_items, task_profile, query
        )
        removed_items.extend(stage3_removed)
        filter_stats['task_aware_filtering'] = stage3_stats
        
        # Stage 4: Context optimization
        optimized_context = self._apply_context_optimization(
            [item.file_path for item in filtered_items], task_type, query, task_profile
        )
        filter_stats['context_optimization'] = {
            'optimized_windows': len(optimized_context),
            'total_size': sum(window.size for window in optimized_context)
        }
        
        # Stage 5: Post-filtering
        final_items, stage5_removed, stage5_stats = self._apply_post_filtering(
            optimized_context, task_profile, query
        )
        removed_items.extend(stage5_removed)
        filter_stats['post_filtering'] = stage5_stats
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Calculate confidence
        confidence = self._calculate_filtering_confidence(filter_stats, task_profile)
        
        # Record filtering operation
        self._record_filtering_operation(
            file_paths, task_type, query, filtered_items, removed_items, 
            filter_stats, execution_time, confidence
        )
        
        return FilterResult(
            filtered_items=final_items,
            removed_items=removed_items,
            filter_stats=filter_stats,
            execution_time=execution_time,
            confidence=confidence
        )
        
    def _apply_pre_filtering(self, items: List[str], task_profile: TaskProfile, 
                           query: Optional[str]) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """Apply pre-filtering stage"""
        filtered_items = []
        removed_items = []
        stats = {
            'total_items': len(items),
            'filtered_by_type': 0,
            'filtered_by_pattern': 0,
            'filtered_by_size': 0
        }
        
        for item in items:
            keep_item = True
            
            # File type filtering
            if task_profile.priority_filters:
                type_filter = next((f for f in task_profile.priority_filters if f.filter_type == FilterType.FILE_TYPE), None)
                if type_filter and type_filter.enabled:
                    if not self._matches_file_type(item, type_filter.pattern):
                        keep_item = False
                        stats['filtered_by_type'] += 1
                        
            # Pattern filtering
            if keep_item and task_profile.priority_filters:
                pattern_filter = next((f for f in task_profile.priority_filters if f.filter_type == FilterType.CODE_PATTERN), None)
                if pattern_filter and pattern_filter.enabled:
                    if not self._matches_code_pattern(item, pattern_filter.pattern):
                        keep_item = False
                        stats['filtered_by_pattern'] += 1
                        
            # Size filtering
            if keep_item:
                try:
                    file_size = os.path.getsize(item)
                    if file_size > 1024 * 1024:  # 1MB limit
                        keep_item = False
                        stats['filtered_by_size'] += 1
                except:
                    pass
                    
            if keep_item:
                filtered_items.append(item)
            else:
                removed_items.append(item)
                
        return filtered_items, removed_items, stats
        
    def _matches_file_type(self, file_path: str, pattern: str) -> bool:
        """Check if file matches type pattern"""
        try:
            return bool(re.search(pattern, file_path, re.IGNORECASE))
        except:
            return True
            
    def _matches_code_pattern(self, file_path: str, pattern: str) -> bool:
        """Check if file contains code pattern"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                return bool(re.search(pattern, content, re.IGNORECASE))
        except:
            return True
            
    def _apply_relevance_scoring(self, items: List[str], task_type: TaskType, 
                               query: Optional[str]) -> List[FileSelection]:
        """Apply relevance scoring stage"""
        from llmcode.file_selector import SelectionCriteria
        
        criteria = SelectionCriteria(
            task_type=task_type,
            max_files=len(items),
            context_window_limit=100000  # Large limit for scoring
        )
        
        return self.file_selector.select_files(criteria, query)
        
    def _apply_task_aware_filtering(self, scored_items: List[FileSelection], 
                                  task_profile: TaskProfile, 
                                  query: Optional[str]) -> Tuple[List[FileSelection], List[FileSelection], Dict[str, Any]]:
        """Apply task-aware filtering stage"""
        filtered_items = []
        removed_items = []
        stats = {
            'total_scored': len(scored_items),
            'filtered_by_threshold': 0,
            'filtered_by_relevance': 0,
            'filtered_by_requirements': 0
        }
        
        # Apply relevance thresholds
        overall_threshold = task_profile.relevance_thresholds.get('overall', 0.5)
        
        for item in scored_items:
            keep_item = True
            
            # Overall relevance threshold
            if item.relevance_score < overall_threshold:
                keep_item = False
                stats['filtered_by_threshold'] += 1
                
            # Task-specific relevance
            if keep_item:
                task_relevance = self._calculate_task_relevance(item.file_path, task_profile, query)
                if task_relevance < task_profile.relevance_thresholds.get('task_relevance', 0.5):
                    keep_item = False
                    stats['filtered_by_relevance'] += 1
                    
            # Context requirements
            if keep_item:
                if not self._meets_context_requirements(item.file_path, task_profile):
                    keep_item = False
                    stats['filtered_by_requirements'] += 1
                    
            if keep_item:
                filtered_items.append(item)
            else:
                removed_items.append(item)
                
        return filtered_items, removed_items, stats
        
    def _calculate_task_relevance(self, file_path: str, task_profile: TaskProfile, 
                                query: Optional[str]) -> float:
        """Calculate task-specific relevance for a file"""
        relevance = 0.0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Check for task-specific patterns
            for filter_criteria in task_profile.priority_filters:
                if filter_criteria.filter_type == FilterType.CODE_PATTERN:
                    if filter_criteria.pattern:
                        matches = re.findall(filter_criteria.pattern, content, re.IGNORECASE)
                        if matches:
                            relevance += filter_criteria.weight * 0.3
                            
            # Check for semantic relevance
            if query:
                query_lower = query.lower()
                content_lower = content.lower()
                
                # Simple keyword matching
                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                
                if query_words and content_words:
                    overlap = len(query_words.intersection(content_words)) / len(query_words)
                    relevance += overlap * 0.4
                    
            # File name relevance
            file_name = Path(file_path).name.lower()
            task_keywords = {
                TaskType.CODE_REVIEW: ['code', 'review', 'quality'],
                TaskType.BUG_FIX: ['bug', 'fix', 'error', 'exception'],
                TaskType.FEATURE_DEVELOPMENT: ['feature', 'new', 'add', 'create'],
                TaskType.REFACTORING: ['refactor', 'improve', 'cleanup'],
                TaskType.DOCUMENTATION: ['doc', 'readme', 'guide'],
                TaskType.TESTING: ['test', 'spec', 'mock'],
                TaskType.PERFORMANCE_OPTIMIZATION: ['perf', 'optim', 'speed'],
                TaskType.SECURITY_AUDIT: ['security', 'auth', 'validation'],
                TaskType.DEPENDENCY_UPDATE: ['dep', 'require', 'package'],
                TaskType.ARCHITECTURE_ANALYSIS: ['arch', 'design', 'structure']
            }
            
            keywords = task_keywords.get(task_profile.task_type, [])
            for keyword in keywords:
                if keyword in file_name:
                    relevance += 0.3
                    
        except Exception as e:
            if self.verbose:
                dump(f"Error calculating task relevance for {file_path}: {e}")
                
        return min(1.0, relevance)
        
    def _meets_context_requirements(self, file_path: str, task_profile: TaskProfile) -> bool:
        """Check if file meets context requirements"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            requirements = task_profile.context_requirements
            
            # Check for required content types
            if requirements.get('include_imports', False):
                if not re.search(r'(import|from|#include|require)', content):
                    return False
                    
            if requirements.get('include_definitions', False):
                if not re.search(r'(class|function|def|method|interface)', content):
                    return False
                    
            if requirements.get('include_documentation', False):
                if not re.search(r'("""|\'\'\'|//|/\*|#)', content):
                    return False
                    
            if requirements.get('include_tests', False):
                if not re.search(r'(test|spec|assert|mock)', content, re.IGNORECASE):
                    return False
                    
            return True
            
        except:
            return True
            
    def _apply_context_optimization(self, file_paths: List[str], task_type: TaskType, 
                                  query: Optional[str], task_profile: TaskProfile) -> List[ContextWindow]:
        """Apply context optimization stage"""
        # Create optimization config from task profile
        config = OptimizationConfig(
            max_window_size=task_profile.context_requirements.get('max_context_size', 4000),
            strategy=task_profile.optimization_preferences.get('strategy', OptimizationStrategy.HYBRID),
            compression_level=CompressionLevel(task_profile.optimization_preferences.get('compression_level', 'moderate')),
            include_metadata=True,
            prioritize_imports=task_profile.context_requirements.get('include_imports', True),
            prioritize_definitions=task_profile.context_requirements.get('include_definitions', True),
            prioritize_tests=task_profile.context_requirements.get('include_tests', False),
            prioritize_docs=task_profile.context_requirements.get('include_documentation', False)
        )
        
        return self.context_optimizer.optimize_context(file_paths, task_type, query, config)
        
    def _apply_post_filtering(self, optimized_context: List[ContextWindow], 
                            task_profile: TaskProfile, 
                            query: Optional[str]) -> Tuple[List[ContextWindow], List[ContextWindow], Dict[str, Any]]:
        """Apply post-filtering stage"""
        filtered_windows = []
        removed_windows = []
        stats = {
            'total_windows': len(optimized_context),
            'filtered_by_size': 0,
            'filtered_by_quality': 0,
            'filtered_by_relevance': 0
        }
        
        max_files = task_profile.context_requirements.get('max_files', 20)
        
        for window in optimized_context:
            keep_window = True
            
            # Size filtering
            if window.size > task_profile.context_requirements.get('max_context_size', 4000) * 1.2:
                keep_window = False
                stats['filtered_by_size'] += 1
                
            # File count filtering
            if keep_window and len(window.included_files) > max_files:
                keep_window = False
                stats['filtered_by_quality'] += 1
                
            # Priority filtering
            if keep_window and window.priority_score < 0.3:
                keep_window = False
                stats['filtered_by_relevance'] += 1
                
            if keep_window:
                filtered_windows.append(window)
            else:
                removed_windows.append(window)
                
        return filtered_windows, removed_windows, stats
        
    def _calculate_filtering_confidence(self, filter_stats: Dict[str, Any], 
                                      task_profile: TaskProfile) -> float:
        """Calculate confidence in filtering results"""
        confidence = 0.0
        
        # Base confidence from filter stages
        total_stages = len(filter_stats)
        successful_stages = sum(1 for stats in filter_stats.values() if stats.get('total_items', 0) > 0)
        
        if total_stages > 0:
            confidence += (successful_stages / total_stages) * 0.4
            
        # Confidence from filtering ratios
        total_items = filter_stats.get('pre_filtering', {}).get('total_items', 0)
        if total_items > 0:
            # Good filtering removes some but not all items
            total_removed = sum(stats.get('filtered_by_type', 0) + stats.get('filtered_by_pattern', 0) 
                              for stats in filter_stats.values())
            removal_ratio = total_removed / total_items
            
            if 0.2 <= removal_ratio <= 0.8:  # Good removal ratio
                confidence += 0.3
                
        # Confidence from final results
        final_windows = filter_stats.get('context_optimization', {}).get('optimized_windows', 0)
        if final_windows > 0:
            confidence += 0.3
            
        return min(1.0, confidence)
        
    def _record_filtering_operation(self, input_files: List[str], task_type: TaskType, 
                                  query: Optional[str], filtered_items: List[Any],
                                  removed_items: List[Any], filter_stats: Dict[str, Any],
                                  execution_time: float, confidence: float):
        """Record filtering operation in history"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'input_files': len(input_files),
            'task_type': task_type.value,
            'query': query,
            'filtered_items': len(filtered_items),
            'removed_items': len(removed_items),
            'filter_stats': filter_stats,
            'execution_time': execution_time,
            'confidence': confidence
        }
        
        self.filter_history.append(record)
        
        # Keep only last 100 operations
        if len(self.filter_history) > 100:
            self.filter_history = self.filter_history[-100:]
            
    def get_filtering_statistics(self) -> Dict[str, Any]:
        """Get statistics about filtering operations"""
        if not self.filter_history:
            return {'message': 'No filtering history available'}
            
        stats = {
            'total_operations': len(self.filter_history),
            'average_execution_time': sum(op['execution_time'] for op in self.filter_history) / len(self.filter_history),
            'average_confidence': sum(op['confidence'] for op in self.filter_history) / len(self.filter_history),
            'task_type_distribution': defaultdict(int),
            'average_filtering_ratio': sum(op['removed_items'] / (op['filtered_items'] + op['removed_items']) 
                                         for op in self.filter_history) / len(self.filter_history),
            'recent_operations': self.filter_history[-5:]
        }
        
        # Analyze task type distribution
        for op in self.filter_history:
            task_type = op['task_type']
            stats['task_type_distribution'][task_type] += 1
            
        return stats
        
    def recommend_task_profile(self, query: str, file_paths: List[str]) -> TaskType:
        """Recommend task type based on query and files"""
        query_lower = query.lower()
        
        # Score each task type
        task_scores = {}
        
        for task_type, profile in self.task_profiles.items():
            score = 0.0
            
            # Query-based scoring
            task_keywords = {
                TaskType.CODE_REVIEW: ['review', 'quality', 'check', 'analyze'],
                TaskType.BUG_FIX: ['bug', 'error', 'fix', 'issue', 'problem'],
                TaskType.FEATURE_DEVELOPMENT: ['feature', 'new', 'add', 'create', 'implement'],
                TaskType.REFACTORING: ['refactor', 'improve', 'cleanup', 'optimize'],
                TaskType.DOCUMENTATION: ['doc', 'document', 'explain', 'guide'],
                TaskType.TESTING: ['test', 'spec', 'mock', 'verify'],
                TaskType.PERFORMANCE_OPTIMIZATION: ['performance', 'speed', 'fast', 'optimize'],
                TaskType.SECURITY_AUDIT: ['security', 'safe', 'vulnerability', 'audit'],
                TaskType.DEPENDENCY_UPDATE: ['dependency', 'package', 'update', 'install'],
                TaskType.ARCHITECTURE_ANALYSIS: ['architecture', 'design', 'structure', 'pattern']
            }
            
            keywords = task_keywords.get(task_type, [])
            for keyword in keywords:
                if keyword in query_lower:
                    score += 1.0
                    
            # File-based scoring
            for file_path in file_paths:
                file_name = Path(file_path).name.lower()
                
                for keyword in keywords:
                    if keyword in file_name:
                        score += 0.5
                        
            task_scores[task_type] = score
            
        # Return task type with highest score
        if task_scores:
            return max(task_scores, key=task_scores.get)
        else:
            return TaskType.CODE_REVIEW  # Default
            
    def create_custom_filter(self, filter_type: FilterType, pattern: Optional[str] = None,
                           threshold: float = 0.5, weight: float = 1.0,
                           metadata: Optional[Dict[str, Any]] = None) -> FilterCriteria:
        """Create a custom filter criteria"""
        return FilterCriteria(
            filter_type=filter_type,
            pattern=pattern,
            threshold=threshold,
            enabled=True,
            weight=weight,
            metadata=metadata or {}
        )
