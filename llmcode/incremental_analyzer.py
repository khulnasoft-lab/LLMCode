"""
Incremental Analysis Module for Llmcode

This module provides incremental analysis capabilities for large codebases,
tracking file changes and only analyzing modified files to improve performance.
"""

import os
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .static_analysis import StaticAnalyzer
from .code_structure import CodeStructureAnalyzer
from .relationship_mapper import RelationshipMapper
from .dependency_graph import DependencyGraphAnalyzer
from .file_selector import IntelligentFileSelector
from .relevance_scorer import RelevanceScorer
from .context_optimizer import ContextOptimizer
from .task_filter import TaskFilter


@dataclass
class FileMetadata:
    """Metadata for tracking file analysis state"""
    path: str
    size: int
    last_modified: float
    content_hash: str
    analysis_version: str
    last_analyzed: float
    analysis_types: Set[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['analysis_types'] = list(self.analysis_types)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FileMetadata':
        """Create from dictionary"""
        data['analysis_types'] = set(data['analysis_types'])
        return cls(**data)


@dataclass
class AnalysisDelta:
    """Represents changes in analysis results"""
    added_files: Set[str]
    modified_files: Set[str]
    deleted_files: Set[str]
    affected_dependencies: Set[str]
    
    def has_changes(self) -> bool:
        """Check if there are any changes"""
        return bool(self.added_files or self.modified_files or self.deleted_files)


class IncrementalAnalyzer:
    """
    Incremental analyzer that tracks file changes and performs selective analysis
    """
    
    def __init__(self, 
                 cache_dir: str = ".llmcode_cache",
                 analysis_version: str = "1.0",
                 enable_hashing: bool = True,
                 max_cache_age: int = 86400):  # 24 hours
        """
        Initialize incremental analyzer
        
        Args:
            cache_dir: Directory for storing analysis cache
            analysis_version: Version identifier for analysis compatibility
            enable_hashing: Whether to use content hashing for change detection
            max_cache_age: Maximum age of cache entries in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.analysis_version = analysis_version
        self.enable_hashing = enable_hashing
        self.max_cache_age = max_cache_age
        
        # Initialize analysis modules
        self.static_analyzer = StaticAnalyzer()
        self.structure_analyzer = CodeStructureAnalyzer()
        self.relationship_mapper = RelationshipMapper()
        self.dependency_analyzer = DependencyGraphAnalyzer()
        self.file_selector = IntelligentFileSelector()
        self.relevance_scorer = RelevanceScorer()
        self.context_optimizer = ContextOptimizer()
        self.task_filter = TaskFilter()
        
        # File tracking
        self.file_metadata: Dict[str, FileMetadata] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load existing cache
        self._load_cache()
    
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file content"""
        if not self.enable_hashing:
            return ""
        
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except (IOError, OSError):
            return ""
    
    def _get_file_metadata(self, file_path: str) -> Optional[FileMetadata]:
        """Get current metadata for a file"""
        try:
            stat = os.stat(file_path)
            return FileMetadata(
                path=file_path,
                size=stat.st_size,
                last_modified=stat.st_mtime,
                content_hash=self._compute_file_hash(file_path),
                analysis_version=self.analysis_version,
                last_analyzed=0.0,
                analysis_types=set()
            )
        except (IOError, OSError):
            return None
    
    def _has_file_changed(self, file_path: str, metadata: FileMetadata) -> bool:
        """Check if a file has changed since last analysis"""
        current_meta = self._get_file_metadata(file_path)
        if current_meta is None:
            return True  # File was deleted
        
        # Check if file was modified
        if current_meta.last_modified > metadata.last_modified:
            if self.enable_hashing:
                return current_meta.content_hash != metadata.content_hash
            return True
        
        # Check if analysis version changed
        if current_meta.analysis_version != metadata.analysis_version:
            return True
        
        return False
    
    def _get_cache_file_path(self) -> Path:
        """Get path to cache file"""
        return self.cache_dir / "incremental_analysis_cache.json"
    
    def _load_cache(self) -> None:
        """Load analysis cache from disk"""
        cache_file = self._get_cache_file_path()
        if not cache_file.exists():
            return
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Load file metadata
            self.file_metadata = {
                path: FileMetadata.from_dict(meta_data)
                for path, meta_data in data.get('file_metadata', {}).items()
            }
            
            # Load dependency graph
            self.dependency_graph = {
                path: set(deps)
                for path, deps in data.get('dependency_graph', {}).items()
            }
            
            # Clean up old cache entries
            self._cleanup_old_cache()
            
        except (json.JSONDecodeError, IOError, OSError):
            # Cache is corrupted, start fresh
            self.file_metadata = {}
            self.dependency_graph = {}
    
    def _save_cache(self) -> None:
        """Save analysis cache to disk"""
        cache_file = self._get_cache_file_path()
        
        try:
            data = {
                'file_metadata': {
                    path: meta.to_dict()
                    for path, meta in self.file_metadata.items()
                },
                'dependency_graph': {
                    path: list(deps)
                    for path, deps in self.dependency_graph.items()
                },
                'cache_version': self.analysis_version,
                'timestamp': time.time()
            }
            
            # Write to temporary file first
            temp_file = cache_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomic rename
            temp_file.replace(cache_file)
            
        except (IOError, OSError):
            pass  # Failed to save cache, continue without it
    
    def _cleanup_old_cache(self) -> None:
        """Remove old cache entries"""
        current_time = time.time()
        files_to_remove = []
        
        for file_path, metadata in self.file_metadata.items():
            if (current_time - metadata.last_analyzed) > self.max_cache_age:
                files_to_remove.append(file_path)
        
        for file_path in files_to_remove:
            del self.file_metadata[file_path]
            if file_path in self.dependency_graph:
                del self.dependency_graph[file_path]
    
    def _detect_changes(self, file_paths: List[str]) -> AnalysisDelta:
        """Detect changes in file set"""
        delta = AnalysisDelta(
            added_files=set(),
            modified_files=set(),
            deleted_files=set(),
            affected_dependencies=set()
        )
        
        # Get current file set
        current_files = set()
        for file_path in file_paths:
            if os.path.exists(file_path):
                current_files.add(file_path)
        
        # Find deleted files
        cached_files = set(self.file_metadata.keys())
        delta.deleted_files = cached_files - current_files
        
        # Find added and modified files
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
                
            if file_path not in self.file_metadata:
                delta.added_files.add(file_path)
            elif self._has_file_changed(file_path, self.file_metadata[file_path]):
                delta.modified_files.add(file_path)
        
        # Find affected dependencies
        changed_files = delta.added_files | delta.modified_files | delta.deleted_files
        for changed_file in changed_files:
            # Files that depend on changed files
            for file_path, deps in self.dependency_graph.items():
                if changed_file in deps:
                    delta.affected_dependencies.add(file_path)
        
        return delta
    
    def _update_file_metadata(self, file_path: str, analysis_types: Set[str]) -> None:
        """Update metadata for analyzed file"""
        metadata = self._get_file_metadata(file_path)
        if metadata is None:
            return
        
        metadata.last_analyzed = time.time()
        metadata.analysis_types.update(analysis_types)
        self.file_metadata[file_path] = metadata
    
    def _update_dependency_graph(self, file_path: str, dependencies: Set[str]) -> None:
        """Update dependency graph"""
        self.dependency_graph[file_path] = dependencies
    
    def analyze_incremental(self, 
                          file_paths: List[str],
                          analysis_types: Set[str],
                          force_refresh: bool = False) -> Dict[str, Any]:
        """
        Perform incremental analysis on files
        
        Args:
            file_paths: List of files to analyze
            analysis_types: Types of analysis to perform
            force_refresh: Whether to force refresh all analysis
            
        Returns:
            Dictionary containing analysis results and metadata
        """
        # Detect changes
        delta = self._detect_changes(file_paths)
        
        # If no changes and not forcing refresh, return cached results
        if not delta.has_changes() and not force_refresh:
            return self._get_cached_results(analysis_types)
        
        # Determine files to analyze
        files_to_analyze = (
            delta.added_files | 
            delta.modified_files | 
            delta.affected_dependencies
        )
        
        # Remove deleted files from cache
        for deleted_file in delta.deleted_files:
            if deleted_file in self.file_metadata:
                del self.file_metadata[deleted_file]
            if deleted_file in self.dependency_graph:
                del self.dependency_graph[deleted_file]
        
        # Perform analysis on changed files
        results = {}
        
        for file_path in files_to_analyze:
            if not os.path.exists(file_path):
                continue
                
            try:
                file_results = self._analyze_file(file_path, analysis_types)
                results[file_path] = file_results
                
                # Update metadata
                self._update_file_metadata(file_path, analysis_types)
                
                # Update dependency graph if dependency analysis was performed
                if 'dependency' in analysis_types and 'dependencies' in file_results:
                    dependencies = set(file_results['dependencies'].get('imports', []))
                    self._update_dependency_graph(file_path, dependencies)
                
            except Exception as e:
                # Log error but continue with other files
                print(f"Error analyzing {file_path}: {e}")
                results[file_path] = {'error': str(e)}
        
        # Save updated cache
        self._save_cache()
        
        return {
            'results': results,
            'delta': {
                'added_files': list(delta.added_files),
                'modified_files': list(delta.modified_files),
                'deleted_files': list(delta.deleted_files),
                'affected_dependencies': list(delta.affected_dependencies)
            },
            'files_analyzed': len(files_to_analyze),
            'total_files': len(file_paths),
            'timestamp': time.time()
        }
    
    def _analyze_file(self, file_path: str, analysis_types: Set[str]) -> Dict[str, Any]:
        """Analyze a single file with specified analysis types"""
        results = {}
        
        # Static analysis
        if 'static' in analysis_types:
            results['static'] = self.static_analyzer.analyze_file(file_path)
        
        # Structure analysis
        if 'structure' in analysis_types:
            results['structure'] = self.structure_analyzer.analyze_module(file_path)
        
        # Dependency analysis
        if 'dependency' in analysis_types:
            results['dependencies'] = self.dependency_analyzer.analyze_file(file_path)
        
        # Relationship analysis
        if 'relationship' in analysis_types:
            results['relationships'] = self.relationship_mapper.analyze_file(file_path)
        
        return results
    
    def _get_cached_results(self, analysis_types: Set[str]) -> Dict[str, Any]:
        """Get cached analysis results"""
        # This would typically load cached results from disk
        # For now, return empty results
        return {
            'results': {},
            'delta': {
                'added_files': [],
                'modified_files': [],
                'deleted_files': [],
                'affected_dependencies': []
            },
            'files_analyzed': 0,
            'total_files': len(self.file_metadata),
            'timestamp': time.time(),
            'cached': True
        }
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get statistics about the incremental analysis"""
        current_time = time.time()
        
        # Count files by analysis type
        analysis_counts = {}
        for metadata in self.file_metadata.values():
            for analysis_type in metadata.analysis_types:
                analysis_counts[analysis_type] = analysis_counts.get(analysis_type, 0) + 1
        
        # Calculate cache age statistics
        cache_ages = [current_time - meta.last_analyzed for meta in self.file_metadata.values()]
        
        return {
            'total_files': len(self.file_metadata),
            'analysis_counts': analysis_counts,
            'cache_age_stats': {
                'oldest': min(cache_ages) if cache_ages else 0,
                'newest': max(cache_ages) if cache_ages else 0,
                'average': sum(cache_ages) / len(cache_ages) if cache_ages else 0
            },
            'dependency_graph_size': len(self.dependency_graph),
            'total_dependencies': sum(len(deps) for deps in self.dependency_graph.values()),
            'analysis_version': self.analysis_version
        }
    
    def clear_cache(self) -> None:
        """Clear all cached analysis data"""
        self.file_metadata = {}
        self.dependency_graph = {}
        
        # Remove cache file
        cache_file = self._get_cache_file_path()
        if cache_file.exists():
            cache_file.unlink()
    
    def invalidate_file(self, file_path: str) -> None:
        """Invalidate cache for a specific file"""
        if file_path in self.file_metadata:
            del self.file_metadata[file_path]
        if file_path in self.dependency_graph:
            del self.dependency_graph[file_path]
