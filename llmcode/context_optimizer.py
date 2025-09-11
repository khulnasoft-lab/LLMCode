"""
Context Window Optimization Strategies for LLMCode

This module provides advanced context optimization capabilities including:
- Context window size optimization
- Content prioritization and compression
- Hierarchical context representation
- Adaptive context selection
- Context compression and summarization
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
from llmcode.relevance_scorer import RelevanceScorer, RelevanceScore, CodeSection
from llmcode.dump import dump


class OptimizationStrategy(Enum):
    """Context optimization strategies"""
    PRIORITY_BASED = "priority_based"
    HIERARCHICAL = "hierarchical"
    DEPENDENCY_BASED = "dependency_based"
    SEMANTIC_BASED = "semantic_based"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


class CompressionLevel(Enum):
    """Levels of context compression"""
    NONE = "none"  # No compression
    MINIMAL = "minimal"  # Remove comments and whitespace
    MODERATE = "moderate"  # Remove non-essential code
    AGGRESSIVE = "aggressive"  # Heavy compression with summarization
    EXTREME = "extreme"  # Maximum compression, metadata only


@dataclass
class ContextWindow:
    """Represents a context window with optimized content"""
    window_id: str
    content: str
    size: int
    max_size: int
    included_files: List[str] = field(default_factory=list)
    included_sections: List[str] = field(default_factory=list)
    compression_ratio: float = 1.0
    priority_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextChunk:
    """Represents a chunk of context with metadata"""
    chunk_id: str
    content: str
    size: int
    priority: float
    file_path: str
    section_info: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    compression_level: CompressionLevel = CompressionLevel.NONE


@dataclass
class OptimizationConfig:
    """Configuration for context optimization"""
    max_window_size: int = 4000
    strategy: OptimizationStrategy = OptimizationStrategy.HYBRID
    compression_level: CompressionLevel = CompressionLevel.MODERATE
    include_metadata: bool = True
    prioritize_imports: bool = True
    prioritize_definitions: bool = True
    prioritize_tests: bool = False
    prioritize_docs: bool = False
    min_chunk_size: int = 100
    max_chunk_size: int = 1000
    overlap_ratio: float = 0.1
    adaptive_threshold: float = 0.7


class ContextOptimizer:
    """Advanced context window optimization"""
    
    def __init__(self, root_path: str, static_analyzer: Optional[StaticAnalyzer] = None,
                 structure_analyzer: Optional[CodeStructureAnalyzer] = None,
                 relationship_mapper: Optional[RelationshipMapper] = None,
                 dependency_graph: Optional[DependencyGraphGenerator] = None,
                 file_selector: Optional[IntelligentFileSelector] = None,
                 relevance_scorer: Optional[RelevanceScorer] = None,
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
        
        # Context optimization history
        self.optimization_history: List[Dict[str, Any]] = []
        
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
            
    def optimize_context(self, file_paths: List[str], task_type: TaskType, 
                         query: Optional[str] = None,
                         config: Optional[OptimizationConfig] = None) -> List[ContextWindow]:
        """Optimize context for given files and task"""
        if config is None:
            config = OptimizationConfig()
            
        # Select relevant files
        from llmcode.file_selector import SelectionCriteria
        criteria = SelectionCriteria(
            task_type=task_type,
            max_files=20,
            context_window_limit=config.max_window_size
        )
        
        selected_files = self.file_selector.select_files(criteria, query)
        
        # Extract context chunks
        chunks = self._extract_context_chunks(selected_files, task_type, query, config)
        
        # Optimize chunks based on strategy
        optimized_chunks = self._optimize_chunks(chunks, config)
        
        # Create context windows
        context_windows = self._create_context_windows(optimized_chunks, config)
        
        # Record optimization
        self._record_optimization(file_paths, task_type, query, config, context_windows)
        
        return context_windows
        
    def _extract_context_chunks(self, selected_files: List[FileSelection], task_type: TaskType,
                              query: Optional[str], config: OptimizationConfig) -> List[ContextChunk]:
        """Extract context chunks from selected files"""
        chunks = []
        
        for file_selection in selected_files:
            file_path = file_selection.file_path
            
            try:
                # Get relevance scores for sections
                section_scores = self.relevance_scorer.score_sections(file_path, task_type, query)
                
                # Create chunks for each section
                for score in section_scores:
                    chunk = self._create_chunk_from_section(score, file_selection.relevance_score, config)
                    chunks.append(chunk)
                    
                # Add file-level chunks for imports and overall structure
                file_chunks = self._create_file_level_chunks(file_path, task_type, config)
                chunks.extend(file_chunks)
                
            except Exception as e:
                if self.verbose:
                    dump(f"Error extracting chunks from {file_path}: {e}")
                    
        return chunks
        
    def _create_chunk_from_section(self, score: RelevanceScore, file_relevance: float,
                                 config: OptimizationConfig) -> ContextChunk:
        """Create a context chunk from a scored section"""
        section = score.section
        
        # Calculate chunk priority
        priority = (score.overall_score * 0.7 + file_relevance * 0.3)
        
        # Apply compression if needed
        compressed_content = self._compress_content(section.content, config.compression_level)
        
        chunk = ContextChunk(
            chunk_id=f"{section.file_path}:{section.start_line}-{section.end_line}",
            content=compressed_content,
            size=len(compressed_content),
            priority=priority,
            file_path=section.file_path,
            section_info={
                'name': section.name,
                'type': section.section_type,
                'start_line': section.start_line,
                'end_line': section.end_line,
                'relevance_score': score.overall_score,
                'dimension_scores': {dim.value: score for dim, score in score.dimension_scores.items()}
            },
            dependencies=section.metadata.get('dependencies', []),
            compression_level=config.compression_level
        )
        
        return chunk
        
    def _create_file_level_chunks(self, file_path: str, task_type: TaskType,
                                config: OptimizationConfig) -> List[ContextChunk]:
        """Create file-level context chunks"""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Import chunk
            if config.prioritize_imports:
                import_chunk = self._create_import_chunk(file_path, content, config)
                if import_chunk:
                    chunks.append(import_chunk)
                    
            # File summary chunk
            if config.include_metadata:
                summary_chunk = self._create_file_summary_chunk(file_path, content, task_type, config)
                if summary_chunk:
                    chunks.append(summary_chunk)
                    
        except Exception as e:
            if self.verbose:
                dump(f"Error creating file-level chunks for {file_path}: {e}")
                
        return chunks
        
    def _create_import_chunk(self, file_path: str, content: str, config: OptimizationConfig) -> Optional[ContextChunk]:
        """Create a chunk for imports"""
        import_lines = []
        lines = content.split('\n')
        
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith('import ') or stripped.startswith('from ') or
                stripped.startswith('#include') or stripped.startswith('require')):
                import_lines.append(line)
            elif import_lines and stripped and not stripped.startswith('#'):
                break  # End of import section
                
        if import_lines:
            import_content = '\n'.join(import_lines)
            compressed_content = self._compress_content(import_content, config.compression_level)
            
            return ContextChunk(
                chunk_id=f"{file_path}:imports",
                content=compressed_content,
                size=len(compressed_content),
                priority=0.8,  # High priority for imports
                file_path=file_path,
                section_info={'type': 'imports'},
                compression_level=config.compression_level
            )
            
        return None
        
    def _create_file_summary_chunk(self, file_path: str, content: str, task_type: TaskType,
                                 config: OptimizationConfig) -> Optional[ContextChunk]:
        """Create a file summary chunk"""
        # Get file statistics
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Count entities
        file_entities = [e for e in self.static_analyzer.entities.values() if e.file_path == file_path]
        
        summary = {
            'file_path': file_path,
            'total_lines': len(lines),
            'non_empty_lines': len(non_empty_lines),
            'entities': {
                'classes': len([e for e in file_entities if e.type == 'class']),
                'functions': len([e for e in file_entities if e.type == 'function']),
                'methods': len([e for e in file_entities if e.type == 'method']),
                'imports': len([e for e in file_entities if e.type == 'import'])
            },
            'language': self._detect_language(file_path),
            'task_type': task_type.value
        }
        
        summary_content = json.dumps(summary, indent=2)
        
        return ContextChunk(
            chunk_id=f"{file_path}:summary",
            content=summary_content,
            size=len(summary_content),
            priority=0.6,  # Medium priority for summary
            file_path=file_path,
            section_info={'type': 'summary'},
            compression_level=CompressionLevel.NONE
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
        
    def _compress_content(self, content: str, compression_level: CompressionLevel) -> str:
        """Compress content based on compression level"""
        if compression_level == CompressionLevel.NONE:
            return content
            
        compressed = content
        
        # Remove comments
        if compression_level in [CompressionLevel.MODERATE, CompressionLevel.AGGRESSIVE, CompressionLevel.EXTREME]:
            compressed = self._remove_comments(compressed)
            
        # Remove excessive whitespace
        if compression_level in [CompressionLevel.MINIMAL, CompressionLevel.MODERATE, CompressionLevel.AGGRESSIVE, CompressionLevel.EXTREME]:
            compressed = self._remove_excessive_whitespace(compressed)
            
        # Remove non-essential code
        if compression_level in [CompressionLevel.AGGRESSIVE, CompressionLevel.EXTREME]:
            compressed = self._remove_non_essential_code(compressed)
            
        # Summarize for extreme compression
        if compression_level == CompressionLevel.EXTREME:
            compressed = self._summarize_content(compressed)
            
        return compressed
        
    def _remove_comments(self, content: str) -> str:
        """Remove comments from content"""
        # Python comments
        content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
        
        # Multi-line comments
        content = re.sub(r'""".*?"""', '', content, flags=re.DOTALL)
        content = re.sub(r"'''.*?'''", '', content, flags=re.DOTALL)
        
        # JavaScript/Java/C++ comments
        content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
        
        return content
        
    def _remove_excessive_whitespace(self, content: str) -> str:
        """Remove excessive whitespace"""
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove leading/trailing whitespace
            cleaned = line.strip()
            
            # Keep non-empty lines and lines that are just whitespace (for structure)
            if cleaned or line.isspace():
                cleaned_lines.append(cleaned)
                
        return '\n'.join(cleaned_lines)
        
    def _remove_non_essential_code(self, content: str) -> str:
        """Remove non-essential code for aggressive compression"""
        lines = content.split('\n')
        essential_lines = []
        
        # Keep imports, class/function definitions, and important statements
        for line in lines:
            stripped = line.strip()
            if (stripped.startswith(('import ', 'from ', 'class ', 'def ', 'async def ')) or
                stripped.startswith(('#include', 'require', 'function', 'class')) or
                re.match(r'^\s*(if|for|while|try|except|finally|return)\b', stripped)):
                essential_lines.append(line)
                
        return '\n'.join(essential_lines)
        
    def _summarize_content(self, content: str) -> str:
        """Create a summary of content for extreme compression"""
        lines = content.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        if len(non_empty_lines) <= 5:
            return content
            
        # Take first 2 and last 2 lines, plus middle if needed
        summary_lines = []
        summary_lines.extend(non_empty_lines[:2])
        
        if len(non_empty_lines) > 6:
            summary_lines.append("...")  # Ellipsis for omitted content
            summary_lines.extend(non_empty_lines[-2:])
        else:
            summary_lines.extend(non_empty_lines[2:])
            
        return '\n'.join(summary_lines)
        
    def _optimize_chunks(self, chunks: List[ContextChunk], config: OptimizationConfig) -> List[ContextChunk]:
        """Optimize chunks based on strategy"""
        if config.strategy == OptimizationStrategy.PRIORITY_BASED:
            return self._optimize_priority_based(chunks, config)
        elif config.strategy == OptimizationStrategy.HIERARCHICAL:
            return self._optimize_hierarchical(chunks, config)
        elif config.strategy == OptimizationStrategy.DEPENDENCY_BASED:
            return self._optimize_dependency_based(chunks, config)
        elif config.strategy == OptimizationStrategy.SEMANTIC_BASED:
            return self._optimize_semantic_based(chunks, config)
        elif config.strategy == OptimizationStrategy.ADAPTIVE:
            return self._optimize_adaptive(chunks, config)
        elif config.strategy == OptimizationStrategy.HYBRID:
            return self._optimize_hybrid(chunks, config)
        else:
            return chunks
            
    def _optimize_priority_based(self, chunks: List[ContextChunk], config: OptimizationConfig) -> List[ContextChunk]:
        """Optimize chunks based on priority"""
        # Sort chunks by priority
        chunks.sort(key=lambda x: x.priority, reverse=True)
        
        # Select top chunks within size limit
        selected_chunks = []
        total_size = 0
        
        for chunk in chunks:
            if total_size + chunk.size <= config.max_window_size:
                selected_chunks.append(chunk)
                total_size += chunk.size
            else:
                break
                
        return selected_chunks
        
    def _optimize_hierarchical(self, chunks: List[ContextChunk], config: OptimizationConfig) -> List[ContextChunk]:
        """Optimize chunks using hierarchical approach"""
        # Group chunks by file
        file_chunks = defaultdict(list)
        for chunk in chunks:
            file_chunks[chunk.file_path].append(chunk)
            
        # Sort files by total priority
        file_priorities = {}
        for file_path, file_chunk_list in file_chunks.items():
            file_priority = sum(chunk.priority for chunk in file_chunk_list) / len(file_chunk_list)
            file_priorities[file_path] = file_priority
            
        # Sort files by priority
        sorted_files = sorted(file_priorities.keys(), key=lambda x: file_priorities[x], reverse=True)
        
        # Select chunks from top files
        selected_chunks = []
        total_size = 0
        
        for file_path in sorted_files:
            file_chunk_list = file_chunks[file_path]
            
            # Sort chunks within file by priority
            file_chunk_list.sort(key=lambda x: x.priority, reverse=True)
            
            for chunk in file_chunk_list:
                if total_size + chunk.size <= config.max_window_size:
                    selected_chunks.append(chunk)
                    total_size += chunk.size
                else:
                    break
                    
            if total_size >= config.max_window_size * 0.9:  # Stop at 90% to allow for overhead
                break
                
        return selected_chunks
        
    def _optimize_dependency_based(self, chunks: List[ContextChunk], config: OptimizationConfig) -> List[ContextChunk]:
        """Optimize chunks based on dependencies"""
        # Build dependency graph
        chunk_graph = nx.DiGraph()
        
        # Add chunks as nodes
        for chunk in chunks:
            chunk_graph.add_node(chunk.chunk_id, chunk=chunk)
            
        # Add edges based on dependencies
        for chunk in chunks:
            for dep in chunk.dependencies:
                # Find chunks that provide this dependency
                for other_chunk in chunks:
                    if (other_chunk.file_path != chunk.file_path and 
                        other_chunk.section_info and 
                        other_chunk.section_info.get('name') == dep):
                        chunk_graph.add_edge(chunk.chunk_id, other_chunk.chunk_id)
                        
        # Calculate centrality
        centrality = nx.degree_centrality(chunk_graph)
        
        # Update chunk priorities based on centrality
        for chunk in chunks:
            chunk_centrality = centrality.get(chunk.chunk_id, 0.0)
            chunk.priority = chunk.priority * 0.7 + chunk_centrality * 0.3
            
        # Use priority-based selection
        return self._optimize_priority_based(chunks, config)
        
    def _optimize_semantic_based(self, chunks: List[ContextChunk], config: OptimizationConfig) -> List[ContextChunk]:
        """Optimize chunks based on semantic similarity"""
        # Calculate semantic similarity between chunks
        similarity_matrix = {}
        
        for i, chunk1 in enumerate(chunks):
            for j, chunk2 in enumerate(chunks[i+1:], i+1):
                similarity = self._calculate_semantic_similarity(chunk1.content, chunk2.content)
                similarity_matrix[(chunk1.chunk_id, chunk2.chunk_id)] = similarity
                
        # Cluster chunks based on similarity
        clusters = self._cluster_chunks_by_similarity(chunks, similarity_matrix, config.adaptive_threshold)
        
        # Select chunks from diverse clusters
        selected_chunks = []
        total_size = 0
        
        # Sort clusters by average priority
        cluster_priorities = {}
        for cluster_id, cluster_chunks in clusters.items():
            avg_priority = sum(chunk.priority for chunk in cluster_chunks) / len(cluster_chunks)
            cluster_priorities[cluster_id] = avg_priority
            
        sorted_clusters = sorted(clusters.keys(), key=lambda x: cluster_priorities[x], reverse=True)
        
        for cluster_id in sorted_clusters:
            cluster_chunks = clusters[cluster_id]
            
            # Sort chunks within cluster by priority
            cluster_chunks.sort(key=lambda x: x.priority, reverse=True)
            
            for chunk in cluster_chunks:
                if total_size + chunk.size <= config.max_window_size:
                    selected_chunks.append(chunk)
                    total_size += chunk.size
                else:
                    break
                    
            if total_size >= config.max_window_size * 0.9:
                break
                
        return selected_chunks
        
    def _calculate_semantic_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity between two contents"""
        # Simple word-based similarity
        words1 = set(re.findall(r'\b\w+\b', content1.lower()))
        words2 = set(re.findall(r'\b\w+\b', content2.lower()))
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def _cluster_chunks_by_similarity(self, chunks: List[ContextChunk], 
                                    similarity_matrix: Dict[Tuple[str, str], float],
                                    threshold: float) -> Dict[int, List[ContextChunk]]:
        """Cluster chunks based on similarity"""
        clusters = {}
        cluster_id = 0
        
        for chunk in chunks:
            assigned = False
            
            # Try to assign to existing cluster
            for existing_cluster_id, cluster_chunks in clusters.items():
                for existing_chunk in cluster_chunks:
                    similarity = similarity_matrix.get((chunk.chunk_id, existing_chunk.chunk_id), 0.0)
                    reverse_similarity = similarity_matrix.get((existing_chunk.chunk_id, chunk.chunk_id), 0.0)
                    max_similarity = max(similarity, reverse_similarity)
                    
                    if max_similarity >= threshold:
                        clusters[existing_cluster_id].append(chunk)
                        assigned = True
                        break
                        
                if assigned:
                    break
                    
            # Create new cluster if not assigned
            if not assigned:
                clusters[cluster_id] = [chunk]
                cluster_id += 1
                
        return clusters
        
    def _optimize_adaptive(self, chunks: List[ContextChunk], config: OptimizationConfig) -> List[ContextChunk]:
        """Optimize chunks using adaptive strategy"""
        # Analyze chunk characteristics
        avg_priority = sum(chunk.priority for chunk in chunks) / len(chunks)
        high_priority_chunks = [chunk for chunk in chunks if chunk.priority > avg_priority]
        
        # Use different strategies based on chunk distribution
        if len(high_priority_chunks) > len(chunks) * 0.7:
            # Many high-priority chunks - use priority-based
            return self._optimize_priority_based(chunks, config)
        elif len(chunks) > 50:
            # Many chunks - use hierarchical
            return self._optimize_hierarchical(chunks, config)
        else:
            # Few chunks - use semantic-based
            return self._optimize_semantic_based(chunks, config)
            
    def _optimize_hybrid(self, chunks: List[ContextChunk], config: OptimizationConfig) -> List[ContextChunk]:
        """Optimize chunks using hybrid strategy"""
        # Combine multiple strategies
        priority_chunks = self._optimize_priority_based(chunks, config)
        hierarchical_chunks = self._optimize_hierarchical(chunks, config)
        semantic_chunks = self._optimize_semantic_based(chunks, config)
        
        # Combine results with weights
        chunk_scores = defaultdict(float)
        
        for chunk in priority_chunks:
            chunk_scores[chunk.chunk_id] += 0.4
            
        for chunk in hierarchical_chunks:
            chunk_scores[chunk.chunk_id] += 0.3
            
        for chunk in semantic_chunks:
            chunk_scores[chunk.chunk_id] += 0.3
            
        # Select top chunks by combined score
        sorted_chunks = sorted(chunks, key=lambda x: chunk_scores[x.chunk_id], reverse=True)
        
        selected_chunks = []
        total_size = 0
        
        for chunk in sorted_chunks:
            if total_size + chunk.size <= config.max_window_size:
                selected_chunks.append(chunk)
                total_size += chunk.size
            else:
                break
                
        return selected_chunks
        
    def _create_context_windows(self, chunks: List[ContextChunk], config: OptimizationConfig) -> List[ContextWindow]:
        """Create context windows from optimized chunks"""
        windows = []
        
        # Single window if chunks fit
        total_size = sum(chunk.size for chunk in chunks)
        
        if total_size <= config.max_window_size:
            window = self._create_single_window(chunks, config)
            windows.append(window)
        else:
            # Multiple windows with overlap
            windows = self._create_multiple_windows(chunks, config)
            
        return windows
        
    def _create_single_window(self, chunks: List[ContextChunk], config: OptimizationConfig) -> ContextWindow:
        """Create a single context window"""
        # Sort chunks by priority
        chunks.sort(key=lambda x: x.priority, reverse=True)
        
        # Build content
        content_parts = []
        included_files = set()
        included_sections = set()
        
        for chunk in chunks:
            content_parts.append(f"# {chunk.chunk_id}\n{chunk.content}\n")
            included_files.add(chunk.file_path)
            if chunk.section_info:
                included_sections.add(f"{chunk.file_path}:{chunk.section_info.get('name', 'unknown')}")
                
        content = '\n'.join(content_parts)
        
        return ContextWindow(
            window_id="window_1",
            content=content,
            size=len(content),
            max_size=config.max_window_size,
            included_files=list(included_files),
            included_sections=list(included_sections),
            compression_ratio=1.0,
            priority_score=sum(chunk.priority for chunk in chunks) / len(chunks) if chunks else 0.0,
            metadata={
                'strategy': config.strategy.value,
                'compression_level': config.compression_level.value,
                'chunk_count': len(chunks)
            }
        )
        
    def _create_multiple_windows(self, chunks: List[ContextChunk], config: OptimizationConfig) -> List[ContextWindow]:
        """Create multiple context windows with overlap"""
        # Sort chunks by priority
        chunks.sort(key=lambda x: x.priority, reverse=True)
        
        windows = []
        window_id = 1
        
        # Create windows with sliding window approach
        chunk_count = len(chunks)
        window_size = max(1, int(chunk_count * 0.6))  # 60% of chunks per window
        overlap = max(1, int(window_size * config.overlap_ratio))
        
        for start_idx in range(0, chunk_count, window_size - overlap):
            end_idx = min(start_idx + window_size, chunk_count)
            window_chunks = chunks[start_idx:end_idx]
            
            # Build window content
            content_parts = []
            included_files = set()
            included_sections = set()
            
            for chunk in window_chunks:
                content_parts.append(f"# {chunk.chunk_id}\n{chunk.content}\n")
                included_files.add(chunk.file_path)
                if chunk.section_info:
                    included_sections.add(f"{chunk.file_path}:{chunk.section_info.get('name', 'unknown')}")
                    
            content = '\n'.join(content_parts)
            
            window = ContextWindow(
                window_id=f"window_{window_id}",
                content=content,
                size=len(content),
                max_size=config.max_window_size,
                included_files=list(included_files),
                included_sections=list(included_sections),
                compression_ratio=1.0,
                priority_score=sum(chunk.priority for chunk in window_chunks) / len(window_chunks) if window_chunks else 0.0,
                metadata={
                    'strategy': config.strategy.value,
                    'compression_level': config.compression_level.value,
                    'chunk_count': len(window_chunks),
                    'window_index': window_id,
                    'total_windows': math.ceil(chunk_count / (window_size - overlap))
                }
            )
            
            windows.append(window)
            window_id += 1
            
        return windows
        
    def _record_optimization(self, file_paths: List[str], task_type: TaskType, query: Optional[str],
                           config: OptimizationConfig, windows: List[ContextWindow]):
        """Record optimization history"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'file_paths': file_paths,
            'task_type': task_type.value,
            'query': query,
            'config': config.__dict__,
            'windows': [
                {
                    'window_id': window.window_id,
                    'size': window.size,
                    'included_files': len(window.included_files),
                    'included_sections': len(window.included_sections),
                    'priority_score': window.priority_score
                }
                for window in windows
            ],
            'total_windows': len(windows),
            'total_size': sum(window.size for window in windows)
        }
        
        self.optimization_history.append(record)
        
        # Keep only last 100 optimizations
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
            
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get statistics about context optimization"""
        if not self.optimization_history:
            return {'message': 'No optimization history available'}
            
        stats = {
            'total_optimizations': len(self.optimization_history),
            'average_windows_per_optimization': sum(len(opt['windows']) for opt in self.optimization_history) / len(self.optimization_history),
            'average_size_per_window': sum(window['size'] for opt in self.optimization_history for window in opt['windows']) / sum(len(opt['windows']) for opt in self.optimization_history),
            'strategy_usage': defaultdict(int),
            'compression_level_usage': defaultdict(int),
            'task_type_distribution': defaultdict(int),
            'recent_optimizations': self.optimization_history[-5:]
        }
        
        # Analyze strategy usage
        for opt in self.optimization_history:
            strategy = opt['config']['strategy']
            stats['strategy_usage'][strategy] += 1
            
        # Analyze compression level usage
        for opt in self.optimization_history:
            compression = opt['config']['compression_level']
            stats['compression_level_usage'][compression] += 1
            
        # Analyze task type distribution
        for opt in self.optimization_history:
            task_type = opt['task_type']
            stats['task_type_distribution'][task_type] += 1
            
        return stats
        
    def recommend_optimization_config(self, task_type: TaskType, file_count: int, 
                                   estimated_content_size: int) -> OptimizationConfig:
        """Recommend optimization configuration based on task and content"""
        config = OptimizationConfig()
        
        # Adjust strategy based on task type
        if task_type in [TaskType.CODE_REVIEW, TaskType.BUG_FIX]:
            config.strategy = OptimizationStrategy.PRIORITY_BASED
        elif task_type in [TaskType.ARCHITECTURE_ANALYSIS, TaskType.FEATURE_DEVELOPMENT]:
            config.strategy = OptimizationStrategy.HIERARCHICAL
        elif task_type in [TaskType.DOCUMENTATION, TaskType.TESTING]:
            config.strategy = OptimizationStrategy.SEMANTIC_BASED
        else:
            config.strategy = OptimizationStrategy.HYBRID
            
        # Adjust compression level based on content size
        if estimated_content_size > config.max_window_size * 2:
            config.compression_level = CompressionLevel.AGGRESSIVE
        elif estimated_content_size > config.max_window_size * 1.5:
            config.compression_level = CompressionLevel.MODERATE
        else:
            config.compression_level = CompressionLevel.MINIMAL
            
        # Adjust priorities based on task
        if task_type == TaskType.DOCUMENTATION:
            config.prioritize_docs = True
            config.prioritize_definitions = False
        elif task_type == TaskType.TESTING:
            config.prioritize_tests = True
        elif task_type == TaskType.DEPENDENCY_UPDATE:
            config.prioritize_imports = True
            
        return config
