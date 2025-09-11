"""
Intelligent File Selection Algorithm for LLMCode

This module provides intelligent file selection capabilities including:
- Task-aware file selection
- Relevance-based file ranking
- Context optimization for file selection
- Multi-criteria decision making
- Adaptive selection strategies
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

from llmcode.static_analysis import StaticAnalyzer, CodeEntity
from llmcode.code_structure import CodeStructureAnalyzer
from llmcode.relationship_mapper import RelationshipMapper, RelationshipType
from llmcode.dependency_graph import DependencyGraphGenerator, DependencyType
from llmcode.dump import dump


class TaskType(Enum):
    """Types of tasks for file selection"""
    CODE_REVIEW = "code_review"
    BUG_FIX = "bug_fix"
    FEATURE_DEVELOPMENT = "feature_development"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SECURITY_AUDIT = "security_audit"
    DEPENDENCY_UPDATE = "dependency_update"
    ARCHITECTURE_ANALYSIS = "architecture_analysis"


class SelectionStrategy(Enum):
    """File selection strategies"""
    RELEVANCE_BASED = "relevance_based"
    DEPENDENCY_BASED = "dependency_based"
    STRUCTURE_BASED = "structure_based"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


@dataclass
class FileFeatures:
    """Features extracted from a file for selection"""
    file_path: str
    size: int
    complexity_score: float
    dependency_count: int
    relationship_count: int
    change_frequency: float
    bug_density: float
    test_coverage: float
    documentation_score: float
    architectural_importance: float
    recency_score: float
    language: str
    file_type: str
    module_affinity: float
    centrality_score: float


@dataclass
class SelectionCriteria:
    """Criteria for file selection"""
    task_type: TaskType
    max_files: int = 10
    min_relevance_score: float = 0.3
    include_tests: bool = False
    include_docs: bool = False
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    strategy: SelectionStrategy = SelectionStrategy.HYBRID
    context_window_limit: int = 4000
    prioritize_recent_changes: bool = False
    prioritize_high_complexity: bool = False
    prioritize_critical_dependencies: bool = False


@dataclass
class FileSelection:
    """Result of file selection"""
    file_path: str
    relevance_score: float
    features: FileFeatures
    selection_reason: str
    rank: int
    estimated_context_size: int


class IntelligentFileSelector:
    """Intelligent file selection algorithm"""
    
    def __init__(self, root_path: str, static_analyzer: Optional[StaticAnalyzer] = None,
                 structure_analyzer: Optional[CodeStructureAnalyzer] = None,
                 relationship_mapper: Optional[RelationshipMapper] = None,
                 dependency_graph: Optional[DependencyGraphGenerator] = None,
                 verbose: bool = False):
        self.root_path = Path(root_path)
        self.verbose = verbose
        
        # Initialize analyzers
        self.static_analyzer = static_analyzer or StaticAnalyzer(root_path, verbose)
        self.structure_analyzer = structure_analyzer or CodeStructureAnalyzer(root_path, self.static_analyzer, verbose)
        self.relationship_mapper = relationship_mapper or RelationshipMapper(root_path, self.static_analyzer, self.structure_analyzer, verbose)
        self.dependency_graph = dependency_graph or DependencyGraphGenerator(root_path, self.static_analyzer, verbose)
        
        # File features cache
        self.file_features: Dict[str, FileFeatures] = {}
        self.selection_history: List[Dict[str, Any]] = []
        
        # Task-specific weights
        self.task_weights = {
            TaskType.CODE_REVIEW: {
                'complexity_score': 0.3,
                'change_frequency': 0.2,
                'bug_density': 0.2,
                'documentation_score': 0.15,
                'architectural_importance': 0.15
            },
            TaskType.BUG_FIX: {
                'bug_density': 0.4,
                'complexity_score': 0.2,
                'dependency_count': 0.15,
                'relationship_count': 0.15,
                'centrality_score': 0.1
            },
            TaskType.FEATURE_DEVELOPMENT: {
                'architectural_importance': 0.3,
                'dependency_count': 0.2,
                'relationship_count': 0.2,
                'complexity_score': 0.15,
                'module_affinity': 0.15
            },
            TaskType.REFACTORING: {
                'complexity_score': 0.4,
                'dependency_count': 0.2,
                'relationship_count': 0.2,
                'architectural_importance': 0.2
            },
            TaskType.DOCUMENTATION: {
                'documentation_score': 0.5,
                'architectural_importance': 0.2,
                'complexity_score': 0.15,
                'change_frequency': 0.15
            },
            TaskType.TESTING: {
                'test_coverage': 0.4,
                'bug_density': 0.2,
                'complexity_score': 0.2,
                'relationship_count': 0.2
            },
            TaskType.PERFORMANCE_OPTIMIZATION: {
                'complexity_score': 0.3,
                'dependency_count': 0.25,
                'relationship_count': 0.25,
                'centrality_score': 0.2
            },
            TaskType.SECURITY_AUDIT: {
                'dependency_count': 0.3,
                'complexity_score': 0.25,
                'architectural_importance': 0.25,
                'centrality_score': 0.2
            },
            TaskType.DEPENDENCY_UPDATE: {
                'dependency_count': 0.5,
                'architectural_importance': 0.3,
                'relationship_count': 0.2
            },
            TaskType.ARCHITECTURE_ANALYSIS: {
                'architectural_importance': 0.4,
                'dependency_count': 0.2,
                'relationship_count': 0.2,
                'complexity_score': 0.2
            }
        }
        
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
            
    def select_files(self, criteria: SelectionCriteria, 
                    query: Optional[str] = None, 
                    context_files: Optional[List[str]] = None) -> List[FileSelection]:
        """Select files based on criteria and query"""
        # Extract features for all files
        self._extract_file_features()
        
        # Filter files based on patterns
        candidate_files = self._filter_files(criteria)
        
        # Score files based on task and query
        scored_files = self._score_files(candidate_files, criteria, query, context_files)
        
        # Rank and select files
        selected_files = self._rank_and_select_files(scored_files, criteria)
        
        # Record selection history
        self._record_selection(criteria, selected_files, query)
        
        return selected_files
        
    def _extract_file_features(self):
        """Extract features from all files in the project"""
        self.file_features.clear()
        
        # Get all source files
        source_files = self._get_source_files()
        
        for file_path in source_files:
            try:
                features = self._extract_features_for_file(file_path)
                self.file_features[file_path] = features
                
            except Exception as e:
                if self.verbose:
                    dump(f"Error extracting features for {file_path}: {e}")
                    
    def _get_source_files(self) -> List[str]:
        """Get all source files in the project"""
        extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h']
        source_files = []
        
        for ext in extensions:
            source_files.extend([str(f) for f in self.root_path.rglob(f'*{ext}') if f.is_file()])
            
        return source_files
        
    def _extract_features_for_file(self, file_path: str) -> FileFeatures:
        """Extract features for a specific file"""
        file_path_obj = Path(file_path)
        
        # Basic file features
        size = file_path_obj.stat().st_size if file_path_obj.exists() else 0
        language = self._detect_language(file_path)
        file_type = self._detect_file_type(file_path)
        
        # Complexity analysis
        complexity_score = self._calculate_complexity_score(file_path)
        
        # Dependency analysis
        dependency_count = self._count_dependencies(file_path)
        
        # Relationship analysis
        relationship_count = self._count_relationships(file_path)
        
        # Change frequency (simulated - in real implementation would use git history)
        change_frequency = self._calculate_change_frequency(file_path)
        
        # Bug density (simulated - in real implementation would use issue tracking)
        bug_density = self._calculate_bug_density(file_path)
        
        # Test coverage (simulated - in real implementation would use coverage tools)
        test_coverage = self._calculate_test_coverage(file_path)
        
        # Documentation score
        documentation_score = self._calculate_documentation_score(file_path)
        
        # Architectural importance
        architectural_importance = self._calculate_architectural_importance(file_path)
        
        # Recency score
        recency_score = self._calculate_recency_score(file_path)
        
        # Module affinity
        module_affinity = self._calculate_module_affinity(file_path)
        
        # Centrality score
        centrality_score = self._calculate_centrality_score(file_path)
        
        return FileFeatures(
            file_path=file_path,
            size=size,
            complexity_score=complexity_score,
            dependency_count=dependency_count,
            relationship_count=relationship_count,
            change_frequency=change_frequency,
            bug_density=bug_density,
            test_coverage=test_coverage,
            documentation_score=documentation_score,
            architectural_importance=architectural_importance,
            recency_score=recency_score,
            language=language,
            file_type=file_type,
            module_affinity=module_affinity,
            centrality_score=centrality_score
        )
        
    def _detect_language(self, file_path: str) -> str:
        """Detect the programming language of a file"""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'cpp'
        }
        return language_map.get(ext, 'unknown')
        
    def _detect_file_type(self, file_path: str) -> str:
        """Detect the type of file (source, test, docs, config)"""
        path_lower = file_path.lower()
        
        if any(keyword in path_lower for keyword in ['test', 'spec']):
            return 'test'
        elif any(keyword in path_lower for keyword in ['doc', 'readme', 'md']):
            return 'documentation'
        elif any(keyword in path_lower for keyword in ['config', 'conf', 'ini', 'yaml', 'yml', 'json']):
            return 'configuration'
        else:
            return 'source'
            
    def _calculate_complexity_score(self, file_path: str) -> float:
        """Calculate complexity score for a file"""
        # Use structure analyzer complexity metrics
        if hasattr(self.structure_analyzer, 'complexity_metrics'):
            metrics = self.structure_analyzer.complexity_metrics
            file_metrics = metrics.get(file_path, {})
            
            cyclomatic = file_metrics.get('cyclomatic_complexity', 1)
            maintainability = file_metrics.get('maintainability_index', 100)
            
            # Normalize to 0-1 scale
            complexity = min(1.0, cyclomatic / 20.0)  # Assume 20 is high complexity
            maintainability_factor = (100 - maintainability) / 100.0
            
            return (complexity + maintainability_factor) / 2.0
            
        return 0.5  # Default if no metrics available
        
    def _count_dependencies(self, file_path: str) -> int:
        """Count dependencies for a file"""
        count = 0
        
        # Count from static analyzer
        for entity in self.static_analyzer.entities.values():
            if entity.file_path == file_path and entity.type == 'import':
                count += len(entity.dependencies)
                
        # Count from dependency graph
        for node in self.dependency_graph.dependency_nodes.values():
            if node.file_path == file_path:
                count += node.import_count
                
        return count
        
    def _count_relationships(self, file_path: str) -> int:
        """Count relationships for a file"""
        count = 0
        
        # Count from relationship mapper
        for rel in self.relationship_mapper.relationships:
            if (file_path in rel.source or file_path in rel.target):
                count += 1
                
        return count
        
    def _calculate_change_frequency(self, file_path: str) -> float:
        """Calculate change frequency (simulated)"""
        # In real implementation, this would use git history
        # For now, use a simple heuristic based on file type and size
        file_type = self._detect_file_type(file_path)
        
        base_frequency = {
            'source': 0.5,
            'test': 0.3,
            'documentation': 0.2,
            'configuration': 0.1
        }.get(file_type, 0.3)
        
        # Adjust based on size (larger files change less frequently)
        size_factor = min(1.0, 10000 / max(1, self.file_features.get(file_path, FileFeatures('', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, '', '', 0, 0)).size))
        
        return base_frequency * size_factor
        
    def _calculate_bug_density(self, file_path: str) -> float:
        """Calculate bug density (simulated)"""
        # In real implementation, this would use issue tracking data
        # For now, use complexity as a proxy
        complexity = self._calculate_complexity_score(file_path)
        return complexity * 0.3  # Assume 30% of complexity contributes to bug density
        
    def _calculate_test_coverage(self, file_path: str) -> float:
        """Calculate test coverage (simulated)"""
        # In real implementation, this would use coverage tools
        # For now, use file type as a proxy
        file_type = self._detect_file_type(file_path)
        
        if file_type == 'test':
            return 1.0
        elif file_type == 'source':
            return 0.6  # Assume 60% coverage for source files
        else:
            return 0.0
            
    def _calculate_documentation_score(self, file_path: str) -> float:
        """Calculate documentation score"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Count documentation lines
            doc_lines = 0
            total_lines = len(content.split('\n'))
            
            # Look for docstrings and comments
            if file_path.endswith('.py'):
                # Count docstrings and comments
                docstring_pattern = r'""".*?"""|\'\'\'.*?\'\'\''
                comment_pattern = r'#.*$'
                
                docstrings = len(re.findall(docstring_pattern, content, re.DOTALL))
                comments = len(re.findall(comment_pattern, content, re.MULTILINE))
                
                doc_lines = docstrings + comments
                
            elif file_path.endswith(('.js', '.ts', '.java', '.cpp', '.c')):
                # Count comments
                comment_patterns = [r'//.*$', r'/\*.*?\*/']
                comments = 0
                for pattern in comment_patterns:
                    comments += len(re.findall(pattern, content, re.MULTILINE | re.DOTALL))
                    
                doc_lines = comments
                
            if total_lines > 0:
                return min(1.0, doc_lines / total_lines)
                
        except Exception:
            pass
            
        return 0.0
        
    def _calculate_architectural_importance(self, file_path: str) -> float:
        """Calculate architectural importance"""
        importance = 0.0
        
        # Check if file is in important directories
        path_parts = Path(file_path).parts
        important_dirs = ['src', 'lib', 'core', 'main', 'app']
        
        for part in path_parts:
            if part.lower() in important_dirs:
                importance += 0.3
                
        # Check dependency centrality
        centrality = self._calculate_centrality_score(file_path)
        importance += centrality * 0.4
        
        # Check module importance
        if hasattr(self.structure_analyzer, 'modules'):
            for module in self.structure_analyzer.modules.values():
                if file_path in module.files:
                    importance += module.importance_score * 0.3
                    break
                    
        return min(1.0, importance)
        
    def _calculate_recency_score(self, file_path: str) -> float:
        """Calculate recency score based on file modification time"""
        try:
            file_path_obj = Path(file_path)
            if file_path_obj.exists():
                mod_time = file_path_obj.stat().st_mtime
                current_time = datetime.now().timestamp()
                
                # Calculate days since last modification
                days_since_mod = (current_time - mod_time) / (24 * 3600)
                
                # More recent files get higher scores
                return max(0.0, 1.0 - days_since_mod / 365.0)  # Decay over a year
                
        except Exception:
            pass
            
        return 0.0
        
    def _calculate_module_affinity(self, file_path: str) -> float:
        """Calculate module affinity score"""
        # Check if file belongs to a well-defined module
        if hasattr(self.structure_analyzer, 'modules'):
            for module in self.structure_analyzer.modules.values():
                if file_path in module.files:
                    return module.cohesion_score
                    
        return 0.0
        
    def _calculate_centrality_score(self, file_path: str) -> float:
        """Calculate centrality score in dependency graph"""
        if not self.dependency_graph.dependency_graph.nodes:
            return 0.0
            
        try:
            # Find nodes related to this file
            file_nodes = [node for node in self.dependency_graph.dependency_graph.nodes 
                         if file_path in node]
            
            if not file_nodes:
                return 0.0
                
            # Calculate degree centrality
            centrality = nx.degree_centrality(self.dependency_graph.dependency_graph)
            
            # Return maximum centrality among file nodes
            return max(centrality.get(node, 0.0) for node in file_nodes)
            
        except Exception:
            return 0.0
            
    def _filter_files(self, criteria: SelectionCriteria) -> List[str]:
        """Filter files based on criteria"""
        candidate_files = []
        
        for file_path, features in self.file_features.items():
            # Check exclude patterns
            if any(re.search(pattern, file_path, re.IGNORECASE) for pattern in criteria.exclude_patterns):
                continue
                
            # Check include patterns
            if criteria.include_patterns and not any(re.search(pattern, file_path, re.IGNORECASE) for pattern in criteria.include_patterns):
                continue
                
            # Check file type filters
            if not criteria.include_tests and features.file_type == 'test':
                continue
                
            if not criteria.include_docs and features.file_type == 'documentation':
                continue
                
            candidate_files.append(file_path)
            
        return candidate_files
        
    def _score_files(self, candidate_files: List[str], criteria: SelectionCriteria,
                    query: Optional[str] = None, context_files: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """Score files based on criteria and query"""
        scored_files = []
        
        # Get weights for the task type
        weights = self.task_weights.get(criteria.task_type, {})
        
        for file_path in candidate_files:
            features = self.file_features[file_path]
            score = 0.0
            
            # Calculate base score using task weights
            for feature_name, weight in weights.items():
                feature_value = getattr(features, feature_name, 0.0)
                score += feature_value * weight
                
            # Apply query relevance boost
            if query:
                query_score = self._calculate_query_relevance(file_path, query)
                score += query_score * 0.3  # 30% boost for query relevance
                
            # Apply context relevance boost
            if context_files:
                context_score = self._calculate_context_relevance(file_path, context_files)
                score += context_score * 0.2  # 20% boost for context relevance
                
            # Apply strategy-specific adjustments
            if criteria.strategy == SelectionStrategy.DEPENDENCY_BASED:
                score = score * 0.7 + features.dependency_count * 0.3
            elif criteria.strategy == SelectionStrategy.STRUCTURE_BASED:
                score = score * 0.7 + features.architectural_importance * 0.3
            elif criteria.strategy == SelectionStrategy.RELEVANCE_BASED:
                score = score * 0.8 + max(0.0, (features.documentation_score + features.test_coverage) / 2.0) * 0.2
                
            # Apply priority adjustments
            if criteria.prioritize_recent_changes:
                score += features.recency_score * 0.2
                
            if criteria.prioritize_high_complexity:
                score += features.complexity_score * 0.2
                
            if criteria.prioritize_critical_dependencies:
                score += features.centrality_score * 0.2
                
            scored_files.append((file_path, score))
            
        return scored_files
        
    def _calculate_query_relevance(self, file_path: str, query: str) -> float:
        """Calculate relevance score for a query"""
        if not query:
            return 0.0
            
        relevance = 0.0
        query_lower = query.lower()
        
        # Check file name
        file_name = Path(file_path).name.lower()
        if query_lower in file_name:
            relevance += 0.4
            
        # Check file path
        file_path_lower = file_path.lower()
        if query_lower in file_path_lower:
            relevance += 0.3
            
        # Check file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().lower()
                
            # Count query occurrences
            occurrences = content.count(query_lower)
            max_occurrences = len(content.split()) // 10  # Normalize by file size
            relevance += min(0.3, occurrences / max(max_occurrences, 1))
            
        except Exception:
            pass
            
        return min(1.0, relevance)
        
    def _calculate_context_relevance(self, file_path: str, context_files: List[str]) -> float:
        """Calculate relevance score based on context files"""
        if not context_files:
            return 0.0
            
        relevance = 0.0
        
        # Check direct dependencies
        for context_file in context_files:
            # Check if file_path depends on context_file or vice versa
            for edge in self.dependency_graph.dependency_edges:
                if ((edge.source == file_path and context_file in edge.target) or
                    (edge.source == context_file and file_path in edge.target)):
                    relevance += 0.3
                    
        # Check relationship proximity
        for rel in self.relationship_mapper.relationships:
            if (file_path in rel.source and any(context_file in rel.target for context_file in context_files)) or \
               (file_path in rel.target and any(context_file in rel.source for context_file in context_files)):
                relevance += 0.2
                
        # Check module affinity
        file_module = None
        context_modules = set()
        
        if hasattr(self.structure_analyzer, 'modules'):
            for module_id, module in self.structure_analyzer.modules.items():
                if file_path in module.files:
                    file_module = module_id
                if any(context_file in module.files for context_file in context_files):
                    context_modules.add(module_id)
                    
            if file_module in context_modules:
                relevance += 0.5
                
        return min(1.0, relevance)
        
    def _rank_and_select_files(self, scored_files: List[Tuple[str, float]], 
                              criteria: SelectionCriteria) -> List[FileSelection]:
        """Rank and select files based on scores"""
        # Sort by score (descending)
        scored_files.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by minimum relevance score
        filtered_files = [(path, score) for path, score in scored_files if score >= criteria.min_relevance_score]
        
        # Select top files
        selected_files = []
        total_context_size = 0
        
        for rank, (file_path, score) in enumerate(filtered_files[:criteria.max_files]):
            features = self.file_features[file_path]
            
            # Estimate context size
            estimated_size = min(features.size, criteria.context_window_limit // 4)  # Conservative estimate
            
            # Check if we have space
            if total_context_size + estimated_size > criteria.context_window_limit:
                break
                
            total_context_size += estimated_size
            
            # Generate selection reason
            reason = self._generate_selection_reason(file_path, score, criteria.task_type)
            
            selection = FileSelection(
                file_path=file_path,
                relevance_score=score,
                features=features,
                selection_reason=reason,
                rank=rank + 1,
                estimated_context_size=estimated_size
            )
            
            selected_files.append(selection)
            
        return selected_files
        
    def _generate_selection_reason(self, file_path: str, score: float, task_type: TaskType) -> str:
        """Generate a human-readable reason for file selection"""
        features = self.file_features[file_path]
        reasons = []
        
        # High score reasons
        if score > 0.8:
            reasons.append("High relevance score")
        elif score > 0.6:
            reasons.append("Good relevance score")
            
        # Task-specific reasons
        if task_type == TaskType.CODE_REVIEW:
            if features.complexity_score > 0.7:
                reasons.append("High complexity - needs review")
            if features.documentation_score < 0.3:
                reasons.append("Low documentation - needs attention")
                
        elif task_type == TaskType.BUG_FIX:
            if features.bug_density > 0.5:
                reasons.append("High bug density")
            if features.change_frequency > 0.6:
                reasons.append("Frequently changed - potential bug source")
                
        elif task_type == TaskType.FEATURE_DEVELOPMENT:
            if features.architectural_importance > 0.7:
                reasons.append("Architecturally important")
            if features.dependency_count > 10:
                reasons.append("High dependency - good extension point")
                
        elif task_type == TaskType.REFACTORING:
            if features.complexity_score > 0.8:
                reasons.append("Very high complexity - refactoring candidate")
            if features.dependency_count > 15:
                reasons.append("High coupling - refactoring needed")
                
        # Feature-based reasons
        if features.centrality_score > 0.7:
            reasons.append("High centrality in dependency graph")
        if features.recency_score > 0.8:
            reasons.append("Recently modified")
        if features.test_coverage < 0.3:
            reasons.append("Low test coverage")
            
        if not reasons:
            reasons.append("General relevance to task")
            
        return "; ".join(reasons)
        
    def _record_selection(self, criteria: SelectionCriteria, selected_files: List[FileSelection], query: Optional[str]):
        """Record selection history for learning"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'criteria': criteria.__dict__,
            'query': query,
            'selected_files': [(sel.file_path, sel.relevance_score) for sel in selected_files],
            'total_files': len(selected_files)
        }
        
        self.selection_history.append(record)
        
        # Keep only last 100 selections
        if len(self.selection_history) > 100:
            self.selection_history = self.selection_history[-100:]
            
    def get_selection_recommendations(self, task_type: TaskType, query: Optional[str] = None) -> Dict[str, Any]:
        """Get recommendations for file selection based on history"""
        if not self.selection_history:
            return {'message': 'No selection history available'}
            
        # Analyze past selections for similar tasks
        similar_selections = [
            sel for sel in self.selection_history 
            if sel['criteria']['task_type'] == task_type
        ]
        
        if not similar_selections:
            return {'message': f'No history for task type: {task_type.value}'}
            
        # Find commonly selected files
        file_counts = Counter()
        for sel in similar_selections:
            for file_path, score in sel['selected_files']:
                file_counts[file_path] += 1
                
        # Get top recommended files
        top_files = file_counts.most_common(10)
        
        return {
            'task_type': task_type.value,
            'similar_selections_count': len(similar_selections),
            'recommended_files': [
                {
                    'file_path': file_path,
                    'selection_frequency': count,
                    'average_score': sum(score for sel in similar_selections 
                                       for fp, score in sel['selected_files'] 
                                       if fp == file_path) / count
                }
                for file_path, count in top_files
            ]
        }
        
    def optimize_selection_criteria(self, criteria: SelectionCriteria) -> SelectionCriteria:
        """Optimize selection criteria based on historical performance"""
        if not self.selection_history:
            return criteria
            
        # Analyze successful selections
        successful_selections = [
            sel for sel in self.selection_history 
            if sel['total_files'] >= 3 and sel['total_files'] <= 15
        ]
        
        if not successful_selections:
            return criteria
            
        # Calculate optimal parameters
        avg_files = sum(sel['total_files'] for sel in successful_selections) / len(successful_selections)
        
        # Create optimized criteria
        optimized_criteria = SelectionCriteria(
            task_type=criteria.task_type,
            max_files=max(3, min(20, int(avg_files))),
            min_relevance_score=criteria.min_relevance_score,
            include_tests=criteria.include_tests,
            include_docs=criteria.include_docs,
            exclude_patterns=criteria.exclude_patterns,
            include_patterns=criteria.include_patterns,
            strategy=criteria.strategy,
            context_window_limit=criteria.context_window_limit,
            prioritize_recent_changes=criteria.prioritize_recent_changes,
            prioritize_high_complexity=criteria.prioritize_high_complexity,
            prioritize_critical_dependencies=criteria.prioritize_critical_dependencies
        )
        
        return optimized_criteria
