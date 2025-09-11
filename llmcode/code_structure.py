"""
Code Structure Understanding Module for LLMCode

This module provides advanced code structure analysis capabilities including:
- Hierarchical code structure analysis
- Architectural pattern detection
- Code complexity metrics
- Module dependency analysis
- Code organization insights
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import networkx as nx
from enum import Enum

from llmcode.static_analysis import StaticAnalyzer, CodeEntity, ImportDependency
from llmcode.dump import dump


class ArchitecturalPattern(Enum):
    """Common architectural patterns"""
    MVC = "Model-View-Controller"
    MVVM = "Model-View-ViewModel"
    LAYERED = "Layered Architecture"
    MICROSERVICES = "Microservices"
    MONOLITH = "Monolithic"
    PLUGIN = "Plugin Architecture"
    EVENT_DRIVEN = "Event-Driven"
    REPOSITORY = "Repository Pattern"
    SERVICE_LOCATOR = "Service Locator"
    DEPENDENCY_INJECTION = "Dependency Injection"
    UNKNOWN = "Unknown"


@dataclass
class CodeModule:
    """Represents a code module/package"""
    name: str
    path: str
    type: str  # 'package', 'module', 'submodule'
    entities: List[CodeEntity] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    complexity_score: float = 0.0
    cohesion_score: float = 0.0
    coupling_score: float = 0.0


@dataclass
class ArchitecturalLayer:
    """Represents an architectural layer"""
    name: str
    modules: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    allowed_dependencies: Set[str] = field(default_factory=set)
    complexity_score: float = 0.0


@dataclass
class ComplexityMetrics:
    """Code complexity metrics"""
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0
    maintainability_index: float = 0.0
    halstead_volume: float = 0.0
    lines_of_code: int = 0
    comment_ratio: float = 0.0
    function_count: int = 0
    class_count: int = 0
    average_function_size: float = 0.0
    maximum_nesting_level: int = 0


class CodeStructureAnalyzer:
    """Advanced code structure analysis and understanding"""
    
    def __init__(self, root_path: str, static_analyzer: Optional[StaticAnalyzer] = None, verbose: bool = False):
        self.root_path = Path(root_path)
        self.verbose = verbose
        self.static_analyzer = static_analyzer or StaticAnalyzer(root_path, verbose)
        
        self.modules: Dict[str, CodeModule] = {}
        self.architectural_layers: List[ArchitecturalLayer] = []
        self.detected_patterns: List[ArchitecturalPattern] = []
        self.complexity_metrics: Dict[str, ComplexityMetrics] = {}
        self.structure_graph = nx.DiGraph()
        
        # Common architectural patterns indicators
        self.pattern_indicators = {
            ArchitecturalPattern.MVC: {
                'model_indicators': ['model', 'entity', 'domain', 'models'],
                'view_indicators': ['view', 'template', 'ui', 'gui', 'views'],
                'controller_indicators': ['controller', 'handler', 'routes', 'controllers']
            },
            ArchitecturalPattern.LAYERED: {
                'layer_indicators': ['data', 'service', 'business', 'presentation', 'api']
            },
            ArchitecturalPattern.MICROSERVICES: {
                'service_indicators': ['service', 'api', 'gateway', 'registry']
            },
            ArchitecturalPattern.REPOSITORY: {
                'repo_indicators': ['repository', 'repo', 'dao', 'data']
            }
        }
        
    def analyze_code_structure(self) -> Dict[str, Any]:
        """Analyze the overall code structure"""
        results = {
            'modules': {},
            'architectural_pattern': ArchitecturalPattern.UNKNOWN,
            'complexity_metrics': {},
            'structure_graph': None,
            'insights': [],
            'recommendations': []
        }
        
        # First, run static analysis if not already done
        if not self.static_analyzer.entities:
            self.static_analyzer.analyze_project()
            
        # Analyze modules
        self._analyze_modules()
        results['modules'] = self.modules
        
        # Detect architectural patterns
        self._detect_architectural_patterns()
        results['architectural_pattern'] = self.detected_patterns[0] if self.detected_patterns else ArchitecturalPattern.UNKNOWN
        
        # Calculate complexity metrics
        self._calculate_complexity_metrics()
        results['complexity_metrics'] = self.complexity_metrics
        
        # Build structure graph
        self._build_structure_graph()
        results['structure_graph'] = self.structure_graph
        
        # Generate insights and recommendations
        results['insights'] = self._generate_insights()
        results['recommendations'] = self._generate_recommendations()
        
        return results
        
    def _analyze_modules(self):
        """Analyze code modules and their relationships"""
        # Group entities by module/package
        module_entities = defaultdict(list)
        
        for entity in self.static_analyzer.entities.values():
            module_path = self._get_module_path(entity.file_path)
            module_entities[module_path].append(entity)
            
        # Create CodeModule objects
        for module_path, entities in module_entities.items():
            module = CodeModule(
                name=Path(module_path).name,
                path=module_path,
                type=self._determine_module_type(module_path),
                entities=entities
            )
            
            # Calculate module dependencies
            module.dependencies = self._calculate_module_dependencies(module_path, entities)
            module.complexity_score = self._calculate_module_complexity(entities)
            module.cohesion_score = self._calculate_module_cohesion(entities)
            module.coupling_score = self._calculate_module_coupling(module_path, entities)
            
            self.modules[module_path] = module
            
        # Calculate dependents
        for module_path, module in self.modules.items():
            for dep_path in module.dependencies:
                if dep_path in self.modules:
                    self.modules[dep_path].dependents.add(module_path)
                    
    def _get_module_path(self, file_path: str) -> str:
        """Get the module path for a file"""
        file_path = Path(file_path)
        relative_path = file_path.relative_to(self.root_path)
        
        # For Python, consider directories with __init__.py as modules
        if file_path.suffix == '.py':
            if relative_path.name == '__init__.py':
                return str(relative_path.parent)
            else:
                return str(relative_path.parent)
        else:
            # For other languages, use the directory structure
            return str(relative_path.parent)
            
    def _determine_module_type(self, module_path: str) -> str:
        """Determine the type of module"""
        path = Path(module_path)
        
        # Check if it's a Python package
        if (self.root_path / path / '__init__.py').exists():
            return 'package'
        elif path.name == path.parent.name:
            return 'submodule'
        else:
            return 'module'
            
    def _calculate_module_dependencies(self, module_path: str, entities: List[CodeEntity]) -> Set[str]:
        """Calculate dependencies for a module"""
        dependencies = set()
        
        for entity in entities:
            for dep in entity.dependencies:
                # Find which module this dependency belongs to
                for other_module_path, other_module in self.modules.items():
                    if other_module_path == module_path:
                        continue
                        
                    for other_entity in other_module.entities:
                        if other_entity.name == dep:
                            dependencies.add(other_module_path)
                            break
                            
        return dependencies
        
    def _calculate_module_complexity(self, entities: List[CodeEntity]) -> float:
        """Calculate complexity score for a module"""
        if not entities:
            return 0.0
            
        complexity_factors = {
            'function_count': len([e for e in entities if e.type == 'function']),
            'class_count': len([e for e in entities if e.type == 'class']),
            'method_count': len([e for e in entities if e.type == 'method']),
            'import_count': len([e for e in entities if e.type == 'import']),
            'total_dependencies': sum(len(e.dependencies) for e in entities)
        }
        
        # Weighted complexity calculation
        complexity = (
            complexity_factors['function_count'] * 1.0 +
            complexity_factors['class_count'] * 2.0 +
            complexity_factors['method_count'] * 1.5 +
            complexity_factors['import_count'] * 0.5 +
            complexity_factors['total_dependencies'] * 0.3
        )
        
        return complexity
        
    def _calculate_module_cohesion(self, entities: List[CodeEntity]) -> float:
        """Calculate cohesion score for a module"""
        if len(entities) <= 1:
            return 1.0
            
        # Simple cohesion calculation based on shared dependencies
        shared_deps = 0
        total_pairs = 0
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                total_pairs += 1
                common_deps = entity1.dependencies.intersection(entity2.dependencies)
                if common_deps:
                    shared_deps += 1
                    
        return shared_deps / total_pairs if total_pairs > 0 else 0.0
        
    def _calculate_module_coupling(self, module_path: str, entities: List[CodeEntity]) -> float:
        """Calculate coupling score for a module"""
        if not entities:
            return 0.0
            
        # Count external dependencies
        external_deps = 0
        total_deps = 0
        
        for entity in entities:
            for dep in entity.dependencies:
                total_deps += 1
                is_external = True
                
                # Check if dependency is internal to this module
                for other_entity in entities:
                    if other_entity.name == dep:
                        is_external = False
                        break
                        
                if is_external:
                    external_deps += 1
                    
        return external_deps / total_deps if total_deps > 0 else 0.0
        
    def _detect_architectural_patterns(self):
        """Detect architectural patterns based on code structure"""
        detected = []
        
        # Check for MVC pattern
        if self._check_mvc_pattern():
            detected.append(ArchitecturalPattern.MVC)
            
        # Check for Layered architecture
        if self._check_layered_pattern():
            detected.append(ArchitecturalPattern.LAYERED)
            
        # Check for Microservices pattern
        if self._check_microservices_pattern():
            detected.append(ArchitecturalPattern.MICROSERVICES)
            
        # Check for Repository pattern
        if self._check_repository_pattern():
            detected.append(ArchitecturalPattern.REPOSITORY)
            
        self.detected_patterns = detected
        
    def _check_mvc_pattern(self) -> bool:
        """Check if the codebase follows MVC pattern"""
        indicators = self.pattern_indicators[ArchitecturalPattern.MVC]
        
        model_found = any(indicator in module_name.lower() 
                         for module_name in self.modules.keys() 
                         for indicator in indicators['model_indicators'])
                         
        view_found = any(indicator in module_name.lower() 
                        for module_name in self.modules.keys() 
                        for indicator in indicators['view_indicators'])
                        
        controller_found = any(indicator in module_name.lower() 
                              for module_name in self.modules.keys() 
                              for indicator in indicators['controller_indicators'])
                              
        return model_found and view_found and controller_found
        
    def _check_layered_pattern(self) -> bool:
        """Check if the codebase follows layered architecture"""
        indicators = self.pattern_indicators[ArchitecturalPattern.LAYERED]
        
        layers_found = 0
        for layer_type in indicators['layer_indicators']:
            if any(layer_type in module_name.lower() for module_name in self.modules.keys()):
                layers_found += 1
                
        return layers_found >= 2
        
    def _check_microservices_pattern(self) -> bool:
        """Check if the codebase follows microservices pattern"""
        indicators = self.pattern_indicators[ArchitecturalPattern.MICROSERVICES]
        
        service_count = 0
        for indicator in indicators['service_indicators']:
            service_count += sum(1 for module_name in self.modules.keys() 
                               if indicator in module_name.lower())
                               
        return service_count >= 3
        
    def _check_repository_pattern(self) -> bool:
        """Check if the codebase follows repository pattern"""
        indicators = self.pattern_indicators[ArchitecturalPattern.REPOSITORY]
        
        repo_found = any(indicator in module_name.lower() 
                        for module_name in self.modules.keys() 
                        for indicator in indicators['repo_indicators'])
                        
        return repo_found
        
    def _calculate_complexity_metrics(self):
        """Calculate complexity metrics for the codebase"""
        for module_path, module in self.modules.items():
            metrics = ComplexityMetrics()
            
            # Calculate basic metrics
            metrics.function_count = len([e for e in module.entities if e.type == 'function'])
            metrics.class_count = len([e for e in module.entities if e.type == 'class'])
            
            # Calculate lines of code (approximation)
            total_lines = 0
            comment_lines = 0
            
            for entity in module.entities:
                if entity.end_line_number:
                    entity_lines = entity.end_line_number - entity.line_number + 1
                    total_lines += entity_lines
                    
            metrics.lines_of_code = total_lines
            
            # Calculate comment ratio (simplified)
            metrics.comment_ratio = 0.1  # Placeholder - would need actual code analysis
            
            # Calculate average function size
            if metrics.function_count > 0:
                metrics.average_function_size = total_lines / metrics.function_count
            else:
                metrics.average_function_size = 0
                
            # Calculate cyclomatic complexity (simplified)
            metrics.cyclomatic_complexity = self._calculate_cyclomatic_complexity(module.entities)
            
            # Calculate maintainability index (simplified)
            metrics.maintainability_index = self._calculate_maintainability_index(metrics)
            
            self.complexity_metrics[module_path] = metrics
            
    def _calculate_cyclomatic_complexity(self, entities: List[CodeEntity]) -> int:
        """Calculate cyclomatic complexity (simplified)"""
        complexity = 1  # Base complexity
        
        for entity in entities:
            if entity.type in ['function', 'method']:
                # Add complexity based on number of dependencies
                complexity += len(entity.dependencies) * 0.5
                
        return int(complexity)
        
    def _calculate_maintainability_index(self, metrics: ComplexityMetrics) -> float:
        """Calculate maintainability index (simplified)"""
        # Simplified maintainability index calculation
        if metrics.lines_of_code == 0:
            return 100.0
            
        loc_factor = max(0, 100 - (metrics.lines_of_code / 1000) * 10)
        complexity_factor = max(0, 100 - metrics.cyclomatic_complexity * 5)
        comment_factor = 100 + metrics.comment_ratio * 50
        
        return (loc_factor + complexity_factor + comment_factor) / 3
        
    def _build_structure_graph(self):
        """Build a graph representing the code structure"""
        self.structure_graph.clear()
        
        # Add nodes for modules
        for module_path, module in self.modules.items():
            self.structure_graph.add_node(module_path, **module.__dict__)
            
        # Add edges for dependencies
        for module_path, module in self.modules.items():
            for dep_path in module.dependencies:
                if dep_path in self.modules:
                    self.structure_graph.add_edge(module_path, dep_path)
                    
    def _generate_insights(self) -> List[str]:
        """Generate insights about the code structure"""
        insights = []
        
        # Pattern detection insights
        if self.detected_patterns:
            insights.append(f"Detected architectural pattern: {self.detected_patterns[0].value}")
        else:
            insights.append("No clear architectural pattern detected")
            
        # Module count insights
        total_modules = len(self.modules)
        insights.append(f"Codebase contains {total_modules} modules")
        
        # Complexity insights
        if self.complexity_metrics:
            avg_complexity = sum(m.cyclomatic_complexity for m in self.complexity_metrics.values()) / len(self.complexity_metrics)
            insights.append(f"Average cyclomatic complexity: {avg_complexity:.1f}")
            
            avg_maintainability = sum(m.maintainability_index for m in self.complexity_metrics.values()) / len(self.complexity_metrics)
            insights.append(f"Average maintainability index: {avg_maintainability:.1f}")
            
        # Dependency insights
        total_dependencies = sum(len(module.dependencies) for module in self.modules.values())
        insights.append(f"Total inter-module dependencies: {total_dependencies}")
        
        return insights
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for code structure improvements"""
        recommendations = []
        
        # Check for high coupling
        high_coupling_modules = [module for module in self.modules.values() if module.coupling_score > 0.7]
        if high_coupling_modules:
            recommendations.append(f"Consider reducing coupling in modules: {[m.name for m in high_coupling_modules]}")
            
        # Check for low cohesion
        low_cohesion_modules = [module for module in self.modules.values() if module.cohesion_score < 0.3]
        if low_cohesion_modules:
            recommendations.append(f"Consider improving cohesion in modules: {[m.name for m in low_cohesion_modules]}")
            
        # Check for high complexity
        high_complexity_modules = [module for module in self.modules.values() if module.complexity_score > 50]
        if high_complexity_modules:
            recommendations.append(f"Consider refactoring high-complexity modules: {[m.name for m in high_complexity_modules]}")
            
        # Architectural recommendations
        if not self.detected_patterns:
            recommendations.append("Consider adopting a clear architectural pattern for better code organization")
            
        return recommendations
        
    def get_module_structure(self, module_path: str) -> Dict[str, Any]:
        """Get detailed structure information for a specific module"""
        if module_path not in self.modules:
            return {}
            
        module = self.modules[module_path]
        
        return {
            'name': module.name,
            'path': module.path,
            'type': module.type,
            'entities': [{'name': e.name, 'type': e.type, 'line': e.line_number} for e in module.entities],
            'dependencies': list(module.dependencies),
            'dependents': list(module.dependents),
            'complexity_score': module.complexity_score,
            'cohesion_score': module.cohesion_score,
            'coupling_score': module.coupling_score,
            'complexity_metrics': self.complexity_metrics.get(module_path).__dict__ if module_path in self.complexity_metrics else {}
        }
        
    def get_architectural_overview(self) -> Dict[str, Any]:
        """Get architectural overview of the codebase"""
        return {
            'detected_patterns': [pattern.value for pattern in self.detected_patterns],
            'total_modules': len(self.modules),
            'total_dependencies': sum(len(module.dependencies) for module in self.modules.values()),
            'average_complexity': sum(m.complexity_score for m in self.modules.values()) / len(self.modules) if self.modules else 0,
            'average_cohesion': sum(m.cohesion_score for m in self.modules.values()) / len(self.modules) if self.modules else 0,
            'average_coupling': sum(m.coupling_score for m in self.modules.values()) / len(self.modules) if self.modules else 0,
            'modules_by_type': self._group_modules_by_type(),
            'complexity_distribution': self._get_complexity_distribution()
        }
        
    def _group_modules_by_type(self) -> Dict[str, List[str]]:
        """Group modules by their type"""
        grouped = defaultdict(list)
        for module_path, module in self.modules.items():
            grouped[module.type].append(module.name)
        return dict(grouped)
        
    def _get_complexity_distribution(self) -> Dict[str, int]:
        """Get distribution of modules by complexity level"""
        distribution = {'low': 0, 'medium': 0, 'high': 0}
        
        for module in self.modules.values():
            if module.complexity_score < 20:
                distribution['low'] += 1
            elif module.complexity_score < 50:
                distribution['medium'] += 1
            else:
                distribution['high'] += 1
                
        return distribution
