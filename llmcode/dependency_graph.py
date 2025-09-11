"""
Import Dependency Graph Generator for LLMCode

This module provides comprehensive import dependency analysis including:
- Multi-language import dependency tracking
- Dependency graph visualization
- Circular dependency detection
- Dependency impact analysis
- Module clustering and grouping
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import networkx as nx
import json
from enum import Enum

from llmcode.static_analysis import StaticAnalyzer, ImportDependency
from llmcode.dump import dump


class DependencyType(Enum):
    """Types of dependencies"""
    STANDARD_LIBRARY = "standard_library"
    THIRD_PARTY = "third_party"
    LOCAL = "local"
    BUILTIN = "builtin"
    RELATIVE = "relative"
    ABSOLUTE = "absolute"


@dataclass
class DependencyNode:
    """Represents a node in the dependency graph"""
    file_path: str
    module_name: str
    dependency_type: DependencyType
    import_count: int = 0
    exported_count: int = 0
    complexity_score: float = 0.0
    is_circular: bool = False
    cluster_id: Optional[int] = None


@dataclass
class DependencyEdge:
    """Represents an edge in the dependency graph"""
    source: str  # Source file path
    target: str  # Target module/file
    dependency_type: DependencyType
    import_count: int = 1
    strength: float = 1.0
    is_circular: bool = False


@dataclass
class DependencyCluster:
    """Represents a cluster of related modules"""
    cluster_id: int
    modules: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    cohesion_score: float = 0.0
    coupling_score: float = 0.0


class DependencyGraphGenerator:
    """Advanced import dependency graph generator"""
    
    def __init__(self, root_path: str, static_analyzer: Optional[StaticAnalyzer] = None, verbose: bool = False):
        self.root_path = Path(root_path)
        self.verbose = verbose
        self.static_analyzer = static_analyzer or StaticAnalyzer(root_path, verbose)
        
        self.dependency_nodes: Dict[str, DependencyNode] = {}
        self.dependency_edges: List[DependencyEdge] = []
        self.dependency_graph = nx.DiGraph()
        self.clusters: Dict[int, DependencyCluster] = {}
        
        # Language-specific import patterns
        self.import_patterns = {
            'python': {
                'standard_import': r'^import\s+(\w+(?:\.\w+)*)',
                'from_import': r'^from\s+(\w+(?:\.\w+)*)\s+import',
                'relative_import': r'^from\s+\.(\w+(?:\.\w+)*)\s+import'
            },
            'javascript': {
                'import': r'^import\s+(?:.*\s+from\s+)?[\'"]([^\'"]+)[\'"]',
                'require': r'^const\s+\w+\s*=\s*require\([\'"]([^\'"]+)[\'"]\)'
            },
            'java': {
                'import': r'^import\s+(?:static\s+)?([^;]+);'
            },
            'cpp': {
                'include': r'^#include\s*[<"]([^>"]+)[>"]'
            }
        }
        
        # Standard library modules by language
        self.standard_libraries = {
            'python': {
                'os', 'sys', 'json', 're', 'math', 'datetime', 'collections',
                'itertools', 'functools', 'operator', 'pathlib', 'typing',
                'dataclasses', 'enum', 'abc', 'contextlib', 'warnings',
                'logging', 'unittest', 'argparse', 'configparser', 'csv',
                'sqlite3', 'pickle', 'shelve', 'tempfile', 'shutil', 'glob'
            },
            'javascript': {
                'fs', 'path', 'http', 'https', 'url', 'querystring', 'stream',
                'events', 'buffer', 'util', 'crypto', 'zlib', 'readline',
                'timers', 'process', 'child_process', 'cluster', 'net'
            },
            'java': {
                'java.lang', 'java.util', 'java.io', 'java.net', 'java.sql',
                'java.time', 'java.math', 'java.text', 'java.nio', 'java.concurrent'
            },
            'cpp': {
                'iostream', 'vector', 'string', 'map', 'set', 'algorithm',
                'memory', 'thread', 'mutex', 'condition_variable', 'chrono',
                'fstream', 'sstream', 'cmath', 'cstdlib', 'cstring'
            }
        }
        
        # Initialize static analyzer if needed
        if not self.static_analyzer.imports:
            self.static_analyzer.analyze_project()
            
    def generate_dependency_graph(self) -> Dict[str, Any]:
        """Generate the complete dependency graph"""
        results = {
            'nodes': {},
            'edges': [],
            'graph': None,
            'clusters': {},
            'circular_dependencies': [],
            'dependency_statistics': {},
            'recommendations': []
        }
        
        # Extract dependencies from imports
        self._extract_dependencies_from_imports()
        
        # Build dependency graph
        self._build_dependency_graph()
        
        # Detect circular dependencies
        circular_deps = self._detect_circular_dependencies()
        results['circular_dependencies'] = circular_deps
        
        # Cluster modules
        self._cluster_modules()
        
        # Calculate statistics
        stats = self._calculate_dependency_statistics()
        results['dependency_statistics'] = stats
        
        # Generate recommendations
        recommendations = self._generate_dependency_recommendations()
        results['recommendations'] = recommendations
        
        # Prepare results
        results['nodes'] = {k: v.__dict__ for k, v in self.dependency_nodes.items()}
        results['edges'] = [e.__dict__ for e in self.dependency_edges]
        results['graph'] = self.dependency_graph
        results['clusters'] = {k: v.__dict__ for k, v in self.clusters.items()}
        
        return results
        
    def _extract_dependencies_from_imports(self):
        """Extract dependencies from static analyzer imports"""
        # Process existing imports from static analyzer
        for import_dep in self.static_analyzer.imports:
            self._process_import_dependency(import_dep)
            
        # Additional language-specific extraction
        self._extract_language_specific_dependencies()
        
    def _process_import_dependency(self, import_dep: ImportDependency):
        """Process a single import dependency"""
        source_file = import_dep.file_path
        
        # Determine dependency type
        dep_type = self._determine_dependency_type(import_dep)
        
        # Create or update dependency node
        target_module = import_dep.module
        node_key = f"{source_file}->{target_module}"
        
        if node_key not in self.dependency_nodes:
            self.dependency_nodes[node_key] = DependencyNode(
                file_path=source_file,
                module_name=target_module,
                dependency_type=dep_type
            )
            
        # Update node
        node = self.dependency_nodes[node_key]
        node.import_count += 1
        
        # Create dependency edge
        edge = DependencyEdge(
            source=source_file,
            target=target_module,
            dependency_type=dep_type,
            import_count=1
        )
        
        self.dependency_edges.append(edge)
        
    def _determine_dependency_type(self, import_dep: ImportDependency) -> DependencyType:
        """Determine the type of dependency"""
        if import_dep.is_standard_library:
            return DependencyType.STANDARD_LIBRARY
        elif import_dep.is_third_party:
            return DependencyType.THIRD_PARTY
        elif import_dep.is_relative:
            return DependencyType.RELATIVE
        else:
            return DependencyType.LOCAL
            
    def _extract_language_specific_dependencies(self):
        """Extract dependencies using language-specific patterns"""
        # Find source files by language
        language_files = {
            'python': list(self.root_path.rglob('*.py')),
            'javascript': list(self.root_path.rglob('*.js')) + list(self.root_path.rglob('*.ts')),
            'java': list(self.root_path.rglob('*.java')),
            'cpp': list(self.root_path.rglob('*.cpp')) + list(self.root_path.rglob('*.h'))
        }
        
        for language, files in language_files.items():
            if language not in self.import_patterns:
                continue
                
            patterns = self.import_patterns[language]
            
            for file_path in files:
                if not file_path.is_file():
                    continue
                    
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    self._extract_dependencies_from_content(content, str(file_path), language, patterns)
                    
                except Exception as e:
                    if self.verbose:
                        dump(f"Error reading {file_path}: {e}")
                        
    def _extract_dependencies_from_content(self, content: str, file_path: str, 
                                         language: str, patterns: Dict[str, str]):
        """Extract dependencies from file content using regex patterns"""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Try each pattern
            for pattern_name, pattern in patterns.items():
                matches = re.finditer(pattern, line, re.MULTILINE)
                
                for match in matches:
                    module_name = match.group(1)
                    
                    # Determine dependency type
                    dep_type = self._determine_dependency_type_from_module(module_name, language)
                    
                    # Create dependency node
                    node_key = f"{file_path}->{module_name}"
                    
                    if node_key not in self.dependency_nodes:
                        self.dependency_nodes[node_key] = DependencyNode(
                            file_path=file_path,
                            module_name=module_name,
                            dependency_type=dep_type
                        )
                        
                    # Update node
                    node = self.dependency_nodes[node_key]
                    node.import_count += 1
                    
                    # Create dependency edge
                    edge = DependencyEdge(
                        source=file_path,
                        target=module_name,
                        dependency_type=dep_type,
                        import_count=1
                    )
                    
                    self.dependency_edges.append(edge)
                    
    def _determine_dependency_type_from_module(self, module_name: str, language: str) -> DependencyType:
        """Determine dependency type from module name and language"""
        if language in self.standard_libraries:
            stdlib = self.standard_libraries[language]
            
            # Check if it's a standard library module
            if any(module_name.startswith(lib) for lib in stdlib):
                return DependencyType.STANDARD_LIBRARY
                
            # Check if it's a relative import
            if module_name.startswith('.') or module_name.startswith('..'):
                return DependencyType.RELATIVE
                
        # For third-party, check if it contains dots (usually indicates package structure)
        if '.' in module_name and not module_name.startswith('.'):
            return DependencyType.THIRD_PARTY
            
        return DependencyType.LOCAL
        
    def _build_dependency_graph(self):
        """Build the NetworkX dependency graph"""
        self.dependency_graph.clear()
        
        # Add nodes
        for node_key, node in self.dependency_nodes.items():
            self.dependency_graph.add_node(node_key, **node.__dict__)
            
        # Add edges
        for edge in self.dependency_edges:
            source_key = f"{edge.source}->{edge.target}"
            target_key = f"{edge.target}"
            
            # Add edge with attributes
            self.dependency_graph.add_edge(
                source_key,
                target_key,
                dependency_type=edge.dependency_type.value,
                import_count=edge.import_count,
                strength=edge.strength
            )
            
    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the graph"""
        circular_deps = []
        
        try:
            # Find all cycles in the graph
            cycles = nx.simple_cycles(self.dependency_graph)
            
            for cycle in cycles:
                if len(cycle) > 1:  # Ignore self-loops
                    # Convert node keys to readable format
                    readable_cycle = []
                    for node_key in cycle:
                        if node_key in self.dependency_nodes:
                            node = self.dependency_nodes[node_key]
                            readable_cycle.append(f"{Path(node.file_path).name} -> {node.module_name}")
                        else:
                            readable_cycle.append(node_key)
                            
                    circular_deps.append(readable_cycle)
                    
                    # Mark nodes as circular
                    for node_key in cycle:
                        if node_key in self.dependency_nodes:
                            self.dependency_nodes[node_key].is_circular = True
                            
        except Exception as e:
            if self.verbose:
                dump(f"Error detecting circular dependencies: {e}")
                
        return circular_deps
        
    def _cluster_modules(self):
        """Cluster modules based on dependency patterns"""
        if not self.dependency_graph.nodes:
            return
            
        try:
            # Use community detection algorithm
            import community as community_louvain
            
            # Create a simpler graph for clustering (using file paths)
            file_graph = nx.Graph()
            
            # Add edges between files that share dependencies
            file_dependencies = defaultdict(set)
            
            for edge in self.dependency_edges:
                file_dependencies[edge.source].add(edge.target)
                
            # Connect files that share similar dependencies
            files = list(file_dependencies.keys())
            for i, file1 in enumerate(files):
                for file2 in files[i+1:]:
                    deps1 = file_dependencies[file1]
                    deps2 = file_dependencies[file2]
                    
                    # Calculate Jaccard similarity
                    intersection = len(deps1.intersection(deps2))
                    union = len(deps1.union(deps2))
                    
                    if union > 0:
                        similarity = intersection / union
                        if similarity > 0.3:  # Threshold for clustering
                            file_graph.add_edge(file1, file2, weight=similarity)
                            
            # Apply Louvain community detection
            if file_graph.nodes:
                partition = community_louvain.best_partition(file_graph)
                
                # Create clusters
                clusters = defaultdict(list)
                for file_path, cluster_id in partition.items():
                    clusters[cluster_id].append(file_path)
                    
                # Create DependencyCluster objects
                for cluster_id, files in clusters.items():
                    cluster = DependencyCluster(cluster_id=cluster_id, modules=files)
                    
                    # Calculate cluster dependencies
                    cluster_deps = set()
                    cluster_dependents = set()
                    
                    for file_path in files:
                        for edge in self.dependency_edges:
                            if edge.source == file_path:
                                cluster_deps.add(edge.target)
                            if edge.target in [f.split('->')[1] for f in files]:
                                cluster_dependents.add(edge.source)
                                
                    cluster.dependencies = cluster_deps
                    cluster.dependents = cluster_dependents
                    
                    # Calculate cohesion and coupling
                    cluster.cohesion_score = self._calculate_cluster_cohesion(cluster)
                    cluster.coupling_score = self._calculate_cluster_coupling(cluster)
                    
                    self.clusters[cluster_id] = cluster
                    
                    # Update cluster_id for nodes
                    for file_path in files:
                        for node_key, node in self.dependency_nodes.items():
                            if node.file_path == file_path:
                                node.cluster_id = cluster_id
                                
        except ImportError:
            if self.verbose:
                dump("Community detection library not available, skipping clustering")
        except Exception as e:
            if self.verbose:
                dump(f"Error in module clustering: {e}")
                
    def _calculate_cluster_cohesion(self, cluster: DependencyCluster) -> float:
        """Calculate cohesion score for a cluster"""
        if len(cluster.modules) <= 1:
            return 1.0
            
        # Calculate shared dependencies
        shared_deps = 0
        total_pairs = 0
        
        for i, file1 in enumerate(cluster.modules):
            for file2 in cluster.modules[i+1:]:
                total_pairs += 1
                
                deps1 = set()
                deps2 = set()
                
                for edge in self.dependency_edges:
                    if edge.source == file1:
                        deps1.add(edge.target)
                    if edge.source == file2:
                        deps2.add(edge.target)
                        
                if deps1.intersection(deps2):
                    shared_deps += 1
                    
        return shared_deps / total_pairs if total_pairs > 0 else 0.0
        
    def _calculate_cluster_coupling(self, cluster: DependencyCluster) -> float:
        """Calculate coupling score for a cluster"""
        if not cluster.modules:
            return 0.0
            
        # Count external dependencies
        external_deps = 0
        total_deps = 0
        
        for file_path in cluster.modules:
            for edge in self.dependency_edges:
                if edge.source == file_path:
                    total_deps += 1
                    if edge.target not in [m.split('->')[1] for m in cluster.modules]:
                        external_deps += 1
                        
        return external_deps / total_deps if total_deps > 0 else 0.0
        
    def _calculate_dependency_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive dependency statistics"""
        stats = {
            'total_nodes': len(self.dependency_nodes),
            'total_edges': len(self.dependency_edges),
            'dependency_types': defaultdict(int),
            'file_statistics': {},
            'module_statistics': {},
            'cluster_statistics': {},
            'graph_metrics': {}
        }
        
        # Dependency type distribution
        for node in self.dependency_nodes.values():
            stats['dependency_types'][node.dependency_type.value] += 1
            
        stats['dependency_types'] = dict(stats['dependency_types'])
        
        # File-level statistics
        file_stats = defaultdict(lambda: {
            'total_imports': 0,
            'standard_lib_imports': 0,
            'third_party_imports': 0,
            'local_imports': 0,
            'circular_deps': 0
        })
        
        for node in self.dependency_nodes.values():
            file_path = node.file_path
            file_stats[file_path]['total_imports'] += node.import_count
            
            if node.dependency_type == DependencyType.STANDARD_LIBRARY:
                file_stats[file_path]['standard_lib_imports'] += node.import_count
            elif node.dependency_type == DependencyType.THIRD_PARTY:
                file_stats[file_path]['third_party_imports'] += node.import_count
            else:
                file_stats[file_path]['local_imports'] += node.import_count
                
            if node.is_circular:
                file_stats[file_path]['circular_deps'] += 1
                
        stats['file_statistics'] = dict(file_stats)
        
        # Module-level statistics
        module_stats = defaultdict(lambda: {
            'imported_by': 0,
            'import_count': 0,
            'files_using': set()
        })
        
        for edge in self.dependency_edges:
            module_stats[edge.target]['imported_by'] += 1
            module_stats[edge.target]['files_using'].add(edge.source)
            
        for node in self.dependency_nodes.values():
            module_stats[node.module_name]['import_count'] += node.import_count
            
        # Convert sets to counts
        for module_name, module_stat in module_stats.items():
            module_stat['files_using'] = len(module_stat['files_using'])
            
        stats['module_statistics'] = dict(module_stats)
        
        # Cluster statistics
        if self.clusters:
            cluster_stats = {}
            for cluster_id, cluster in self.clusters.items():
                cluster_stats[cluster_id] = {
                    'module_count': len(cluster.modules),
                    'cohesion_score': cluster.cohesion_score,
                    'coupling_score': cluster.coupling_score,
                    'dependencies_count': len(cluster.dependencies),
                    'dependents_count': len(cluster.dependents)
                }
                
            stats['cluster_statistics'] = cluster_stats
            
        # Graph metrics
        if self.dependency_graph.number_of_nodes() > 0:
            stats['graph_metrics'] = {
                'density': nx.density(self.dependency_graph),
                'average_degree': sum(dict(self.dependency_graph.degree()).values()) / self.dependency_graph.number_of_nodes(),
                'number_of_components': nx.number_weakly_connected_components(self.dependency_graph)
            }
            
        return stats
        
    def _generate_dependency_recommendations(self) -> List[str]:
        """Generate recommendations based on dependency analysis"""
        recommendations = []
        
        # Check for circular dependencies
        circular_deps = self._detect_circular_dependencies()
        if circular_deps:
            recommendations.append(f"Found {len(circular_deps)} circular dependencies that should be resolved")
            
        # Check for high coupling
        high_coupling_files = []
        for file_path, stats in self._calculate_dependency_statistics()['file_statistics'].items():
            if stats['total_imports'] > 20:  # Threshold
                high_coupling_files.append(file_path)
                
        if high_coupling_files:
            recommendations.append(f"Consider reducing imports in highly coupled files: {high_coupling_files[:5]}")
            
        # Check for cluster cohesion
        low_cohesion_clusters = [
            cluster_id for cluster_id, cluster in self.clusters.items() 
            if cluster.cohesion_score < 0.3
        ]
        
        if low_cohesion_clusters:
            recommendations.append(f"Consider reorganizing modules in low-cohesion clusters: {low_cohesion_clusters}")
            
        # Check for standard library usage
        stdlib_usage = sum(1 for node in self.dependency_nodes.values() 
                          if node.dependency_type == DependencyType.STANDARD_LIBRARY)
        
        if stdlib_usage < len(self.dependency_nodes) * 0.3:
            recommendations.append("Consider using more standard library modules to reduce third-party dependencies")
            
        return recommendations
        
    def get_file_dependencies(self, file_path: str) -> Dict[str, Any]:
        """Get detailed dependency information for a specific file"""
        file_deps = {
            'file_path': file_path,
            'dependencies': [],
            'statistics': {
                'total_imports': 0,
                'standard_lib_imports': 0,
                'third_party_imports': 0,
                'local_imports': 0,
                'circular_deps': 0
            },
            'recommendations': []
        }
        
        # Find all dependencies for this file
        for node in self.dependency_nodes.values():
            if node.file_path == file_path:
                dep_info = {
                    'module': node.module_name,
                    'type': node.dependency_type.value,
                    'import_count': node.import_count,
                    'is_circular': node.is_circular
                }
                file_deps['dependencies'].append(dep_info)
                
                # Update statistics
                file_deps['statistics']['total_imports'] += node.import_count
                
                if node.dependency_type == DependencyType.STANDARD_LIBRARY:
                    file_deps['statistics']['standard_lib_imports'] += node.import_count
                elif node.dependency_type == DependencyType.THIRD_PARTY:
                    file_deps['statistics']['third_party_imports'] += node.import_count
                else:
                    file_deps['statistics']['local_imports'] += node.import_count
                    
                if node.is_circular:
                    file_deps['statistics']['circular_deps'] += 1
                    
        # Generate file-specific recommendations
        if file_deps['statistics']['total_imports'] > 15:
            file_deps['recommendations'].append("Consider reducing the number of imports")
            
        if file_deps['statistics']['circular_deps'] > 0:
            file_deps['recommendations'].append("Resolve circular dependencies")
            
        third_party_ratio = (file_deps['statistics']['third_party_imports'] / 
                            file_deps['statistics']['total_imports'] 
                            if file_deps['statistics']['total_imports'] > 0 else 0)
                            
        if third_party_ratio > 0.7:
            file_deps['recommendations'].append("Consider using more standard library modules")
            
        return file_deps
        
    def get_module_usage(self, module_name: str) -> Dict[str, Any]:
        """Get usage information for a specific module"""
        module_usage = {
            'module_name': module_name,
            'used_by': [],
            'total_usage': 0,
            'usage_by_file': {},
            'usage_type': 'unknown'
        }
        
        # Find all usage of this module
        for edge in self.dependency_edges:
            if edge.target == module_name:
                module_usage['used_by'].append(edge.source)
                module_usage['total_usage'] += edge.import_count
                
                if edge.source not in module_usage['usage_by_file']:
                    module_usage['usage_by_file'][edge.source] = {
                        'import_count': 0,
                        'dependency_type': edge.dependency_type.value
                    }
                    
                module_usage['usage_by_file'][edge.source]['import_count'] += edge.import_count
                
                # Determine usage type
                if edge.dependency_type == DependencyType.STANDARD_LIBRARY:
                    module_usage['usage_type'] = 'standard_library'
                elif edge.dependency_type == DependencyType.THIRD_PARTY:
                    module_usage['usage_type'] = 'third_party'
                else:
                    module_usage['usage_type'] = 'local'
                    
        return module_usage
        
    def export_dependency_graph(self, output_path: str, format: str = 'json'):
        """Export the dependency graph to a file"""
        if format.lower() == 'json':
            data = {
                'nodes': {k: v.__dict__ for k, v in self.dependency_nodes.items()},
                'edges': [e.__dict__ for e in self.dependency_edges],
                'clusters': {k: v.__dict__ for k, v in self.clusters.items()},
                'statistics': self._calculate_dependency_statistics()
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif format.lower() == 'graphml':
            nx.write_graphml(self.dependency_graph, output_path)
            
        elif format.lower() == 'gexf':
            nx.write_gexf(self.dependency_graph, output_path)
            
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
    def visualize_dependency_graph(self, output_path: str, layout: str = 'spring'):
        """Generate a visualization of the dependency graph"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            plt.figure(figsize=(15, 10))
            
            # Choose layout
            if layout == 'spring':
                pos = nx.spring_layout(self.dependency_graph, k=2, iterations=50)
            elif layout == 'circular':
                pos = nx.circular_layout(self.dependency_graph)
            elif layout == 'shell':
                pos = nx.shell_layout(self.dependency_graph)
            else:
                pos = nx.spring_layout(self.dependency_graph)
                
            # Draw nodes by dependency type
            node_colors = []
            for node in self.dependency_graph.nodes():
                if node in self.dependency_nodes:
                    dep_type = self.dependency_nodes[node].dependency_type
                    if dep_type == DependencyType.STANDARD_LIBRARY:
                        node_colors.append('lightblue')
                    elif dep_type == DependencyType.THIRD_PARTY:
                        node_colors.append('lightgreen')
                    else:
                        node_colors.append('lightcoral')
                else:
                    node_colors.append('gray')
                    
            # Draw the graph
            nx.draw(self.dependency_graph, pos, node_color=node_colors, 
                   node_size=100, with_labels=False, alpha=0.7)
            
            # Add legend
            legend_elements = [
                patches.Patch(color='lightblue', label='Standard Library'),
                patches.Patch(color='lightgreen', label='Third Party'),
                patches.Patch(color='lightcoral', label='Local'),
                patches.Patch(color='gray', label='Unknown')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            
            plt.title("Dependency Graph")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except ImportError:
            if self.verbose:
                dump("Matplotlib not available for visualization")
        except Exception as e:
            if self.verbose:
                dump(f"Error generating visualization: {e}")
