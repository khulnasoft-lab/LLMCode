"""
Relevance Scoring for Code Sections for LLMCode

This module provides advanced relevance scoring capabilities including:
- Code section relevance analysis
- Multi-dimensional scoring metrics
- Context-aware relevance calculation
- Semantic similarity analysis
- Task-specific relevance weighting
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

from llmcode.static_analysis import StaticAnalyzer, CodeEntity
from llmcode.code_structure import CodeStructureAnalyzer
from llmcode.relationship_mapper import RelationshipMapper, RelationshipType
from llmcode.dependency_graph import DependencyGraphGenerator, DependencyType
from llmcode.file_selector import IntelligentFileSelector, TaskType
from llmcode.dump import dump


class ScoringDimension(Enum):
    """Dimensions for relevance scoring"""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    STRUCTURAL_IMPORTANCE = "structural_importance"
    DEPENDENCY_CENTRALITY = "dependency_centrality"
    TASK_RELEVANCE = "task_relevance"
    CONTEXT_PROXIMITY = "context_proximity"
    CODE_COMPLEXITY = "code_complexity"
    CHANGE_FREQUENCY = "change_frequency"
    DOCUMENTATION_QUALITY = "documentation_quality"
    TEST_COVERAGE = "test_coverage"
    BUG_DENSITY = "bug_density"


@dataclass
class CodeSection:
    """Represents a section of code"""
    file_path: str
    start_line: int
    end_line: int
    section_type: str  # function, class, method, block, etc.
    name: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RelevanceScore:
    """Relevance score for a code section"""
    section: CodeSection
    overall_score: float
    dimension_scores: Dict[ScoringDimension, float] = field(default_factory=dict)
    confidence: float
    explanation: str
    ranking_factors: List[str] = field(default_factory=list)


@dataclass
class ScoringWeights:
    """Weights for different scoring dimensions"""
    weights: Dict[ScoringDimension, float] = field(default_factory=dict)
    
    def __post_init__(self):
        # Set default weights
        default_weights = {
            ScoringDimension.SEMANTIC_SIMILARITY: 0.25,
            ScoringDimension.STRUCTURAL_IMPORTANCE: 0.20,
            ScoringDimension.DEPENDENCY_CENTRALITY: 0.15,
            ScoringDimension.TASK_RELEVANCE: 0.20,
            ScoringDimension.CONTEXT_PROXIMITY: 0.10,
            ScoringDimension.CODE_COMPLEXITY: 0.05,
            ScoringDimension.CHANGE_FREQUENCY: 0.02,
            ScoringDimension.DOCUMENTATION_QUALITY: 0.02,
            ScoringDimension.TEST_COVERAGE: 0.01,
            ScoringDimension.BUG_DENSITY: 0.00
        }
        
        for dim, weight in default_weights.items():
            if dim not in self.weights:
                self.weights[dim] = weight
                
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for dim in self.weights:
                self.weights[dim] /= total_weight


class RelevanceScorer:
    """Advanced relevance scoring for code sections"""
    
    def __init__(self, root_path: str, static_analyzer: Optional[StaticAnalyzer] = None,
                 structure_analyzer: Optional[CodeStructureAnalyzer] = None,
                 relationship_mapper: Optional[RelationshipMapper] = None,
                 dependency_graph: Optional[DependencyGraphGenerator] = None,
                 file_selector: Optional[IntelligentFileSelector] = None,
                 verbose: bool = False):
        self.root_path = Path(root_path)
        self.verbose = verbose
        
        # Initialize analyzers
        self.static_analyzer = static_analyzer or StaticAnalyzer(root_path, verbose)
        self.structure_analyzer = structure_analyzer or CodeStructureAnalyzer(root_path, self.static_analyzer, verbose)
        self.relationship_mapper = relationship_mapper or RelationshipMapper(root_path, self.static_analyzer, self.structure_analyzer, verbose)
        self.dependency_graph = dependency_graph or DependencyGraphGenerator(root_path, self.static_analyzer, verbose)
        self.file_selector = file_selector or IntelligentFileSelector(root_path, self.static_analyzer, self.structure_analyzer, self.relationship_mapper, self.dependency_graph, verbose)
        
        # Task-specific scoring weights
        self.task_weights = {
            TaskType.CODE_REVIEW: ScoringWeights({
                ScoringDimension.CODE_COMPLEXITY: 0.30,
                ScoringDimension.DOCUMENTATION_QUALITY: 0.25,
                ScoringDimension.STRUCTURAL_IMPORTANCE: 0.20,
                ScoringDimension.SEMANTIC_SIMILARITY: 0.15,
                ScoringDimension.BUG_DENSITY: 0.10
            }),
            TaskType.BUG_FIX: ScoringWeights({
                ScoringDimension.BUG_DENSITY: 0.35,
                ScoringDimension.DEPENDENCY_CENTRALITY: 0.25,
                ScoringDimension.CHANGE_FREQUENCY: 0.20,
                ScoringDimension.CODE_COMPLEXITY: 0.15,
                ScoringDimension.TEST_COVERAGE: 0.05
            }),
            TaskType.FEATURE_DEVELOPMENT: ScoringWeights({
                ScoringDimension.STRUCTURAL_IMPORTANCE: 0.30,
                ScoringDimension.DEPENDENCY_CENTRALITY: 0.25,
                ScoringDimension.TASK_RELEVANCE: 0.20,
                ScoringDimension.SEMANTIC_SIMILARITY: 0.15,
                ScoringDimension.CONTEXT_PROXIMITY: 0.10
            }),
            TaskType.REFACTORING: ScoringWeights({
                ScoringDimension.CODE_COMPLEXITY: 0.40,
                ScoringDimension.DEPENDENCY_CENTRALITY: 0.25,
                ScoringDimension.STRUCTURAL_IMPORTANCE: 0.20,
                ScoringDimension.DOCUMENTATION_QUALITY: 0.10,
                ScoringDimension.TEST_COVERAGE: 0.05
            }),
            TaskType.DOCUMENTATION: ScoringWeights({
                ScoringDimension.DOCUMENTATION_QUALITY: 0.50,
                ScoringDimension.SEMANTIC_SIMILARITY: 0.25,
                ScoringDimension.STRUCTURAL_IMPORTANCE: 0.15,
                ScoringDimension.CONTEXT_PROXIMITY: 0.10
            }),
            TaskType.TESTING: ScoringWeights({
                ScoringDimension.TEST_COVERAGE: 0.40,
                ScoringDimension.BUG_DENSITY: 0.25,
                ScoringDimension.CODE_COMPLEXITY: 0.20,
                ScoringDimension.DEPENDENCY_CENTRALITY: 0.15
            }),
            TaskType.PERFORMANCE_OPTIMIZATION: ScoringWeights({
                ScoringDimension.CODE_COMPLEXITY: 0.35,
                ScoringDimension.DEPENDENCY_CENTRALITY: 0.25,
                ScoringDimension.STRUCTURAL_IMPORTANCE: 0.20,
                ScoringDimension.CHANGE_FREQUENCY: 0.15,
                ScoringDimension.CONTEXT_PROXIMITY: 0.05
            }),
            TaskType.SECURITY_AUDIT: ScoringWeights({
                ScoringDimension.DEPENDENCY_CENTRALITY: 0.30,
                ScoringDimension.STRUCTURAL_IMPORTANCE: 0.25,
                ScoringDimension.CODE_COMPLEXITY: 0.20,
                ScoringDimension.BUG_DENSITY: 0.15,
                ScoringDimension.CHANGE_FREQUENCY: 0.10
            }),
            TaskType.DEPENDENCY_UPDATE: ScoringWeights({
                ScoringDimension.DEPENDENCY_CENTRALITY: 0.50,
                ScoringDimension.STRUCTURAL_IMPORTANCE: 0.25,
                ScoringDimension.TASK_RELEVANCE: 0.15,
                ScoringDimension.CONTEXT_PROXIMITY: 0.10
            }),
            TaskType.ARCHITECTURE_ANALYSIS: ScoringWeights({
                ScoringDimension.STRUCTURAL_IMPORTANCE: 0.40,
                ScoringDimension.DEPENDENCY_CENTRALITY: 0.30,
                ScoringDimension.TASK_RELEVANCE: 0.20,
                ScoringDimension.SEMANTIC_SIMILARITY: 0.10
            })
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
            
    def score_sections(self, file_path: str, task_type: TaskType, query: Optional[str] = None,
                      context_sections: Optional[List[CodeSection]] = None,
                      max_sections: int = 20) -> List[RelevanceScore]:
        """Score code sections in a file for relevance"""
        # Extract code sections from the file
        sections = self._extract_code_sections(file_path)
        
        # Score each section
        scored_sections = []
        
        for section in sections:
            score = self._score_section(section, task_type, query, context_sections)
            scored_sections.append(score)
            
        # Sort by overall score
        scored_sections.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Return top sections
        return scored_sections[:max_sections]
        
    def _extract_code_sections(self, file_path: str) -> List[CodeSection]:
        """Extract code sections from a file"""
        sections = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            # Extract sections based on entities from static analyzer
            file_entities = [e for e in self.static_analyzer.entities.values() if e.file_path == file_path]
            
            for entity in file_entities:
                if entity.type in ['function', 'method', 'class']:
                    section = CodeSection(
                        file_path=file_path,
                        start_line=entity.line_number,
                        end_line=min(entity.line_number + 50, len(lines)),  # Estimate end line
                        section_type=entity.type,
                        name=entity.name,
                        content='\n'.join(lines[entity.line_number-1:min(entity.line_number+49, len(lines))]),
                        metadata={
                            'entity_type': entity.type,
                            'dependencies': entity.dependencies,
                            'parent': entity.parent
                        }
                    )
                    sections.append(section)
                    
            # If no entities found, create sections based on structure
            if not sections:
                sections = self._create_structural_sections(file_path, content)
                
        except Exception as e:
            if self.verbose:
                dump(f"Error extracting sections from {file_path}: {e}")
                
        return sections
        
    def _create_structural_sections(self, file_path: str, content: str) -> List[CodeSection]:
        """Create sections based on code structure when entities are not available"""
        sections = []
        lines = content.split('\n')
        
        # Simple section creation based on indentation and blank lines
        current_section = []
        section_start = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Start new section at major boundaries
            if (stripped.startswith('def ') or stripped.startswith('class ') or 
                stripped.startswith('async def ') or not stripped):
                
                if current_section:
                    section_content = '\n'.join(current_section)
                    sections.append(CodeSection(
                        file_path=file_path,
                        start_line=section_start + 1,
                        end_line=i,
                        section_type='block',
                        name=f"section_{section_start + 1}",
                        content=section_content,
                        metadata={'generated': True}
                    ))
                    
                current_section = [line]
                section_start = i
            else:
                current_section.append(line)
                
        # Add final section
        if current_section:
            section_content = '\n'.join(current_section)
            sections.append(CodeSection(
                file_path=file_path,
                start_line=section_start + 1,
                end_line=len(lines),
                section_type='block',
                name=f"section_{section_start + 1}",
                content=section_content,
                metadata={'generated': True}
            ))
            
        return sections
        
    def _score_section(self, section: CodeSection, task_type: TaskType, query: Optional[str] = None,
                      context_sections: Optional[List[CodeSection]] = None) -> RelevanceScore:
        """Score a single code section for relevance"""
        # Get task-specific weights
        weights = self.task_weights.get(task_type, ScoringWeights())
        
        # Calculate dimension scores
        dimension_scores = {}
        
        dimension_scores[ScoringDimension.SEMANTIC_SIMILARITY] = self._calculate_semantic_similarity(section, query)
        dimension_scores[ScoringDimension.STRUCTURAL_IMPORTANCE] = self._calculate_structural_importance(section)
        dimension_scores[ScoringDimension.DEPENDENCY_CENTRALITY] = self._calculate_dependency_centrality(section)
        dimension_scores[ScoringDimension.TASK_RELEVANCE] = self._calculate_task_relevance(section, task_type)
        dimension_scores[ScoringDimension.CONTEXT_PROXIMITY] = self._calculate_context_proximity(section, context_sections)
        dimension_scores[ScoringDimension.CODE_COMPLEXITY] = self._calculate_code_complexity(section)
        dimension_scores[ScoringDimension.CHANGE_FREQUENCY] = self._calculate_change_frequency(section)
        dimension_scores[ScoringDimension.DOCUMENTATION_QUALITY] = self._calculate_documentation_quality(section)
        dimension_scores[ScoringDimension.TEST_COVERAGE] = self._calculate_test_coverage(section)
        dimension_scores[ScoringDimension.BUG_DENSITY] = self._calculate_bug_density(section)
        
        # Calculate overall score
        overall_score = sum(score * weights.weights[dim] for dim, score in dimension_scores.items())
        
        # Calculate confidence
        confidence = self._calculate_confidence(dimension_scores)
        
        # Generate explanation
        explanation = self._generate_explanation(section, dimension_scores, weights)
        
        # Generate ranking factors
        ranking_factors = self._generate_ranking_factors(section, dimension_scores)
        
        return RelevanceScore(
            section=section,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            confidence=confidence,
            explanation=explanation,
            ranking_factors=ranking_factors
        )
        
    def _calculate_semantic_similarity(self, section: CodeSection, query: Optional[str]) -> float:
        """Calculate semantic similarity between section and query"""
        if not query:
            return 0.0
            
        query_lower = query.lower()
        content_lower = section.content.lower()
        name_lower = section.name.lower()
        
        # Calculate similarity metrics
        similarity_scores = []
        
        # Name similarity
        name_similarity = self._calculate_string_similarity(name_lower, query_lower)
        similarity_scores.append(name_similarity * 0.4)  # 40% weight for name
        
        # Content similarity
        content_similarity = self._calculate_string_similarity(content_lower, query_lower)
        similarity_scores.append(content_similarity * 0.3)  # 30% weight for content
        
        # Keyword overlap
        query_keywords = set(query_lower.split())
        content_keywords = set(content_lower.split())
        
        if query_keywords and content_keywords:
            overlap = len(query_keywords.intersection(content_keywords)) / len(query_keywords)
            similarity_scores.append(overlap * 0.3)  # 30% weight for keyword overlap
            
        return min(1.0, sum(similarity_scores))
        
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        # Simple word-based similarity
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
        
    def _calculate_structural_importance(self, section: CodeSection) -> float:
        """Calculate structural importance of the section"""
        importance = 0.0
        
        # Section type importance
        type_importance = {
            'class': 0.8,
            'function': 0.6,
            'method': 0.5,
            'block': 0.3
        }
        importance += type_importance.get(section.section_type, 0.3)
        
        # Name-based importance
        important_keywords = ['main', 'init', 'setup', 'config', 'core', 'base', 'abstract']
        name_lower = section.name.lower()
        
        if any(keyword in name_lower for keyword in important_keywords):
            importance += 0.3
            
        # Size-based importance (larger sections are often more important)
        content_lines = len(section.content.split('\n'))
        size_factor = min(1.0, content_lines / 50.0)  # Normalize to 50 lines
        importance += size_factor * 0.2
        
        # Metadata-based importance
        if 'entity_type' in section.metadata:
            if section.metadata['entity_type'] == 'class':
                importance += 0.2
                
        return min(1.0, importance)
        
    def _calculate_dependency_centrality(self, section: CodeSection) -> float:
        """Calculate dependency centrality of the section"""
        centrality = 0.0
        
        # Check if section corresponds to an entity
        entity_key = f"{section.file_path}:{section.name}"
        
        if entity_key in self.relationship_mapper.entity_relationships:
            entity_rel = self.relationship_mapper.entity_relationships[entity_key]
            
            # Count relationships
            total_relationships = (len(entity_rel.incoming_relationships) + 
                                 len(entity_rel.outgoing_relationships))
            
            # Normalize centrality
            centrality = min(1.0, total_relationships / 10.0)  # Assume 10 is high
            
        # Check dependencies in metadata
        if 'dependencies' in section.metadata:
            dep_count = len(section.metadata['dependencies'])
            centrality += min(0.5, dep_count / 10.0)  # Additional weight for dependencies
            
        return min(1.0, centrality)
        
    def _calculate_task_relevance(self, section: CodeSection, task_type: TaskType) -> float:
        """Calculate task-specific relevance"""
        relevance = 0.0
        
        # Task-specific patterns
        task_patterns = {
            TaskType.CODE_REVIEW: ['complex', 'large', 'important', 'critical'],
            TaskType.BUG_FIX: ['error', 'exception', 'bug', 'fix', 'issue'],
            TaskType.FEATURE_DEVELOPMENT: ['new', 'add', 'create', 'implement', 'feature'],
            TaskType.REFACTORING: ['refactor', 'improve', 'optimize', 'cleanup'],
            TaskType.DOCUMENTATION: ['doc', 'comment', 'explain', 'describe'],
            TaskType.TESTING: ['test', 'assert', 'verify', 'validate'],
            TaskType.PERFORMANCE_OPTIMIZATION: ['performance', 'speed', 'optimize', 'efficient'],
            TaskType.SECURITY_AUDIT: ['security', 'safe', 'validate', 'check', 'protect'],
            TaskType.DEPENDENCY_UPDATE: ['import', 'require', 'dependency', 'package'],
            TaskType.ARCHITECTURE_ANALYSIS: ['architecture', 'design', 'structure', 'pattern']
        }
        
        patterns = task_patterns.get(task_type, [])
        content_lower = section.content.lower()
        name_lower = section.name.lower()
        
        # Check for task-relevant keywords
        for pattern in patterns:
            if pattern in content_lower or pattern in name_lower:
                relevance += 0.3
                
        # Section type relevance
        type_relevance = {
            TaskType.CODE_REVIEW: {'class': 0.3, 'function': 0.2},
            TaskType.BUG_FIX: {'function': 0.3, 'method': 0.2},
            TaskType.FEATURE_DEVELOPMENT: {'class': 0.3, 'function': 0.2},
            TaskType.REFACTORING: {'class': 0.4, 'function': 0.3},
            TaskType.DOCUMENTATION: {'function': 0.2, 'class': 0.1},
            TaskType.TESTING: {'function': 0.4, 'method': 0.3},
            TaskType.PERFORMANCE_OPTIMIZATION: {'function': 0.3, 'method': 0.2},
            TaskType.SECURITY_AUDIT: {'function': 0.3, 'class': 0.2},
            TaskType.DEPENDENCY_UPDATE: {'block': 0.3, 'function': 0.2},
            TaskType.ARCHITECTURE_ANALYSIS: {'class': 0.4, 'function': 0.2}
        }
        
        type_scores = type_relevance.get(task_type, {})
        relevance += type_scores.get(section.section_type, 0.0)
        
        return min(1.0, relevance)
        
    def _calculate_context_proximity(self, section: CodeSection, 
                                   context_sections: Optional[List[CodeSection]]) -> float:
        """Calculate proximity to context sections"""
        if not context_sections:
            return 0.0
            
        proximity = 0.0
        
        # Calculate line distance
        for context_section in context_sections:
            if context_section.file_path == section.file_path:
                distance = abs(section.start_line - context_section.start_line)
                
                # Closer sections get higher proximity
                if distance < 10:
                    proximity += 0.5
                elif distance < 50:
                    proximity += 0.3
                elif distance < 100:
                    proximity += 0.1
                    
        # Check for shared dependencies
        section_deps = set(section.metadata.get('dependencies', []))
        
        for context_section in context_sections:
            context_deps = set(context_section.metadata.get('dependencies', []))
            
            if section_deps.intersection(context_deps):
                proximity += 0.3
                
        return min(1.0, proximity / len(context_sections) if context_sections else 0.0)
        
    def _calculate_code_complexity(self, section: CodeSection) -> float:
        """Calculate code complexity of the section"""
        complexity = 0.0
        
        # Line-based complexity
        lines = section.content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if non_empty_lines:
            # More lines = higher complexity
            line_complexity = min(1.0, len(non_empty_lines) / 100.0)
            complexity += line_complexity * 0.3
            
        # Structural complexity
        complexity_indicators = [
            r'\bif\b', r'\belse\b', r'\belif\b',  # Conditionals
            r'\bfor\b', r'\bwhile\b',  # Loops
            r'\btry\b', r'\bexcept\b', r'\bfinally\b',  # Exception handling
            r'\bdef\b', r'\bclass\b',  # Definitions
            r'\blambda\b',  # Lambda functions
            r'\[.*for.*in.*\]',  # List comprehensions
            r'\{.*for.*in.*\}',  # Dict comprehensions
        ]
        
        content_lower = section.content.lower()
        indicator_count = 0
        
        for pattern in complexity_indicators:
            matches = re.findall(pattern, content_lower)
            indicator_count += len(matches)
            
        # Normalize complexity indicators
        indicator_complexity = min(1.0, indicator_count / 20.0)
        complexity += indicator_complexity * 0.4
        
        # Nesting complexity
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
                
        nesting_complexity = min(1.0, max_indent / 20.0)  # Assume 20 spaces is high nesting
        complexity += nesting_complexity * 0.3
        
        return min(1.0, complexity)
        
    def _calculate_change_frequency(self, section: CodeSection) -> float:
        """Calculate change frequency of the section"""
        # This is a simplified version - in practice, you'd use git history
        frequency = 0.0
        
        # Section type frequency
        type_frequency = {
            'function': 0.6,
            'method': 0.5,
            'class': 0.3,
            'block': 0.4
        }
        frequency += type_frequency.get(section.section_type, 0.3)
        
        # Size-based frequency (smaller sections change more frequently)
        content_lines = len(section.content.split('\n'))
        size_factor = max(0.0, 1.0 - content_lines / 100.0)  # Inverse relationship
        frequency += size_factor * 0.4
        
        # Name-based frequency
        high_freq_keywords = ['temp', 'test', 'debug', 'fix', 'quick']
        name_lower = section.name.lower()
        
        if any(keyword in name_lower for keyword in high_freq_keywords):
            frequency += 0.3
            
        return min(1.0, frequency)
        
    def _calculate_documentation_quality(self, section: CodeSection) -> float:
        """Calculate documentation quality of the section"""
        quality = 0.0
        
        lines = section.content.split('\n')
        
        # Check for docstrings/comments
        doc_lines = 0
        total_lines = len(lines)
        
        if section.section_type in ['function', 'method', 'class']:
            # Look for docstrings
            if total_lines > 0:
                first_line = lines[0].strip()
                if first_line.startswith('"""') or first_line.startswith("'''"):
                    quality += 0.5
                    
        # Count comment lines
        comment_patterns = [r'#.*$', r'//.*$', r'/\*.*?\*/']
        
        for line in lines:
            for pattern in comment_patterns:
                if re.search(pattern, line):
                    doc_lines += 1
                    break
                    
        # Documentation ratio
        if total_lines > 0:
            doc_ratio = doc_lines / total_lines
            quality += doc_ratio * 0.5
            
        return min(1.0, quality)
        
    def _calculate_test_coverage(self, section: CodeSection) -> float:
        """Calculate test coverage of the section"""
        # This is a simplified version - in practice, you'd use coverage tools
        coverage = 0.0
        
        # File-based coverage
        if 'test' in section.file_path.lower():
            coverage = 1.0
        elif section.section_type == 'function':
            # Assume functions have moderate coverage
            coverage = 0.6
        elif section.section_type == 'class':
            # Assume classes have lower coverage
            coverage = 0.4
        else:
            coverage = 0.2
            
        return coverage
        
    def _calculate_bug_density(self, section: CodeSection) -> float:
        """Calculate bug density of the section"""
        # This is a simplified version - in practice, you'd use bug tracking data
        density = 0.0
        
        # Complexity-based bug density
        complexity = self._calculate_code_complexity(section)
        density += complexity * 0.5
        
        # Size-based bug density
        content_lines = len(section.content.split('\n'))
        size_density = min(1.0, content_lines / 200.0)  # Larger sections have more bugs
        density += size_density * 0.3
        
        # Pattern-based bug density
        bug_patterns = [
            r'\bexcept\b.*\bpass\b',  # Bare except with pass
            r'\bprint\b',  # Debug prints
            r'\bTODO\b', r'\bFIXME\b', r'\bHACK\b',  # Technical debt
            r'\bglobal\b',  # Global variables
            r'\beval\b', r'\bexec\b',  # Dangerous functions
        ]
        
        content_lower = section.content.lower()
        pattern_count = 0
        
        for pattern in bug_patterns:
            matches = re.findall(pattern, content_lower)
            pattern_count += len(matches)
            
        pattern_density = min(1.0, pattern_count / 5.0)
        density += pattern_density * 0.2
        
        return min(1.0, density)
        
    def _calculate_confidence(self, dimension_scores: Dict[ScoringDimension, float]) -> float:
        """Calculate confidence in the relevance score"""
        # Confidence based on the distribution of scores
        scores = list(dimension_scores.values())
        
        if not scores:
            return 0.0
            
        # High variance in scores indicates lower confidence
        mean_score = sum(scores) / len(scores)
        variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
        
        # Normalize confidence (lower variance = higher confidence)
        confidence = max(0.0, 1.0 - variance)
        
        # Boost confidence if we have high scores in key dimensions
        key_dimensions = [
            ScoringDimension.SEMANTIC_SIMILARITY,
            ScoringDimension.STRUCTURAL_IMPORTANCE,
            ScoringDimension.TASK_RELEVANCE
        ]
        
        key_scores = [dimension_scores.get(dim, 0.0) for dim in key_dimensions]
        avg_key_score = sum(key_scores) / len(key_scores)
        
        confidence += avg_key_score * 0.3
        
        return min(1.0, confidence)
        
    def _generate_explanation(self, section: CodeSection, dimension_scores: Dict[ScoringDimension, float],
                            weights: ScoringWeights) -> str:
        """Generate explanation for the relevance score"""
        explanations = []
        
        # Top contributing dimensions
        sorted_dimensions = sorted(dimension_scores.items(), key=lambda x: x[1], reverse=True)
        
        for dim, score in sorted_dimensions[:3]:
            if score > 0.5:
                weight = weights.weights.get(dim, 0.0)
                explanations.append(f"{dim.value.replace('_', ' ').title()}: {score:.2f} (weight: {weight:.2f})")
                
        # Section-specific explanations
        if section.section_type == 'class':
            explanations.append("Class definition with structural importance")
        elif section.section_type == 'function':
            explanations.append("Function definition with behavioral relevance")
            
        if dimension_scores.get(ScoringDimension.SEMANTIC_SIMILARITY, 0.0) > 0.7:
            explanations.append("High semantic similarity to query")
            
        if dimension_scores.get(ScoringDimension.CODE_COMPLEXITY, 0.0) > 0.7:
            explanations.append("High complexity indicates importance")
            
        return "; ".join(explanations) if explanations else "General relevance based on multiple factors"
        
    def _generate_ranking_factors(self, section: CodeSection, 
                                dimension_scores: Dict[ScoringDimension, float]) -> List[str]:
        """Generate ranking factors for the section"""
        factors = []
        
        # High-scoring dimensions
        for dim, score in dimension_scores.items():
            if score > 0.7:
                factors.append(f"High {dim.value.replace('_', ' ')}")
                
        # Section characteristics
        if section.section_type == 'class':
            factors.append("Class definition")
        elif section.section_type == 'function':
            factors.append("Function definition")
            
        # Size factors
        content_lines = len(section.content.split('\n'))
        if content_lines > 50:
            factors.append("Large section")
        elif content_lines < 10:
            factors.append("Small section")
            
        # Dependency factors
        if 'dependencies' in section.metadata:
            dep_count = len(section.metadata['dependencies'])
            if dep_count > 5:
                factors.append("High dependency count")
                
        return factors if factors else ["Standard section"]
        
    def get_top_relevant_sections(self, file_paths: List[str], task_type: TaskType, 
                                 query: Optional[str] = None, 
                                 max_sections_per_file: int = 5,
                                 max_total_sections: int = 20) -> List[RelevanceScore]:
        """Get top relevant sections across multiple files"""
        all_scores = []
        
        for file_path in file_paths:
            file_scores = self.score_sections(file_path, task_type, query, 
                                            max_sections=max_sections_per_file)
            all_scores.extend(file_scores)
            
        # Sort by overall score
        all_scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        return all_scores[:max_total_sections]
        
    def get_section_relevance_summary(self, scores: List[RelevanceScore]) -> Dict[str, Any]:
        """Get a summary of relevance scores"""
        if not scores:
            return {'message': 'No scores available'}
            
        summary = {
            'total_sections': len(scores),
            'average_score': sum(s.overall_score for s in scores) / len(scores),
            'max_score': max(s.overall_score for s in scores),
            'min_score': min(s.overall_score for s in scores),
            'dimension_averages': {},
            'section_type_distribution': defaultdict(int),
            'top_sections': []
        }
        
        # Calculate dimension averages
        all_dimensions = set()
        for score in scores:
            all_dimensions.update(score.dimension_scores.keys())
            
        for dim in all_dimensions:
            dim_scores = [s.dimension_scores.get(dim, 0.0) for s in scores]
            summary['dimension_averages'][dim.value] = sum(dim_scores) / len(dim_scores)
            
        # Section type distribution
        for score in scores:
            summary['section_type_distribution'][score.section.section_type] += 1
            
        # Top sections
        for score in scores[:5]:
            summary['top_sections'].append({
                'file_path': score.section.file_path,
                'section_name': score.section.name,
                'score': score.overall_score,
                'explanation': score.explanation
            })
            
        return summary
