"""
Function/Class Relationship Mapper for LLMCode

This module provides advanced relationship mapping capabilities including:
- Function and class relationship analysis
- Inheritance hierarchy mapping
- Call graph generation
- Data flow analysis
- Relationship visualization support
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import networkx as nx
from enum import Enum

from llmcode.static_analysis import StaticAnalyzer, CodeEntity
from llmcode.code_structure import CodeStructureAnalyzer
from llmcode.dump import dump


class RelationshipType(Enum):
    """Types of relationships between code entities"""
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    CALLS = "calls"
    USES = "uses"
    CREATES = "creates"
    REFERENCES = "references"
    CONTAINS = "contains"
    DEPENDS_ON = "depends_on"
    IMPORTS = "imports"


@dataclass
class Relationship:
    """Represents a relationship between two code entities"""
    source: str  # Entity key
    target: str  # Entity key
    relationship_type: RelationshipType
    strength: float = 1.0  # Strength of the relationship (0.0 to 1.0)
    context: Optional[str] = None  # Additional context about the relationship
    line_number: Optional[int] = None


@dataclass
class EntityRelationships:
    """Relationships for a specific entity"""
    entity_key: str
    entity: CodeEntity
    incoming_relationships: List[Relationship] = field(default_factory=list)
    outgoing_relationships: List[Relationship] = field(default_factory=list)
    relationship_summary: Dict[str, int] = field(default_factory=dict)


class RelationshipMapper:
    """Advanced function/class relationship mapping"""
    
    def __init__(self, root_path: str, static_analyzer: Optional[StaticAnalyzer] = None, 
                 structure_analyzer: Optional[CodeStructureAnalyzer] = None, verbose: bool = False):
        self.root_path = Path(root_path)
        self.verbose = verbose
        self.static_analyzer = static_analyzer or StaticAnalyzer(root_path, verbose)
        self.structure_analyzer = structure_analyzer or CodeStructureAnalyzer(root_path, static_analyzer, verbose)
        
        self.relationships: List[Relationship] = []
        self.entity_relationships: Dict[str, EntityRelationships] = {}
        self.call_graph = nx.DiGraph()
        self.inheritance_graph = nx.DiGraph()
        self.dependency_graph = nx.DiGraph()
        
        # Initialize analyzers if needed
        if not self.static_analyzer.entities:
            self.static_analyzer.analyze_project()
        if not self.structure_analyzer.modules:
            self.structure_analyzer.analyze_code_structure()
            
    def map_relationships(self) -> Dict[str, Any]:
        """Map all relationships between code entities"""
        results = {
            'total_relationships': 0,
            'relationship_types': defaultdict(int),
            'call_graph': None,
            'inheritance_graph': None,
            'dependency_graph': None,
            'entity_relationships': {},
            'relationship_insights': [],
            'critical_relationships': []
        }
        
        # Map different types of relationships
        self._map_inheritance_relationships()
        self._map_call_relationships()
        self._map_usage_relationships()
        self._map_creation_relationships()
        self._map_reference_relationships()
        self._map_containment_relationships()
        
        # Build relationship graphs
        self._build_call_graph()
        self._build_inheritance_graph()
        self._build_dependency_graph()
        
        # Generate entity relationship summaries
        self._generate_entity_relationship_summaries()
        
        # Analyze relationships
        self._analyze_relationships()
        
        # Prepare results
        results['total_relationships'] = len(self.relationships)
        results['relationship_types'] = dict(Counter(r.relationship_type for r in self.relationships))
        results['call_graph'] = self.call_graph
        results['inheritance_graph'] = self.inheritance_graph
        results['dependency_graph'] = self.dependency_graph
        results['entity_relationships'] = {k: v.__dict__ for k, v in self.entity_relationships.items()}
        results['relationship_insights'] = self._generate_relationship_insights()
        results['critical_relationships'] = self._identify_critical_relationships()
        
        return results
        
    def _map_inheritance_relationships(self):
        """Map inheritance relationships between classes"""
        # For Python files, analyze AST for inheritance
        python_files = [f for f in self.root_path.rglob('*.py') if f.is_file()]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content, filename=str(file_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        # Map inheritance relationships
                        for base in node.bases:
                            source_key = f"{file_path}:{node.name}"
                            
                            if isinstance(base, ast.Name):
                                target_key = self._find_entity_by_name(base.id, str(file_path))
                            elif isinstance(base, ast.Attribute):
                                target_key = self._find_entity_by_name(self._get_attribute_name(base), str(file_path))
                            else:
                                continue
                                
                            if target_key:
                                relationship = Relationship(
                                    source=source_key,
                                    target=target_key,
                                    relationship_type=RelationshipType.INHERITS,
                                    context=f"Class {node.name} inherits from {base}",
                                    line_number=node.lineno
                                )
                                self.relationships.append(relationship)
                                
            except Exception as e:
                if self.verbose:
                    dump(f"Error analyzing inheritance in {file_path}: {e}")
                    
    def _map_call_relationships(self):
        """Map function/method call relationships"""
        python_files = [f for f in self.root_path.rglob('*.py') if f.is_file()]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content, filename=str(file_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Find function calls within this function
                        for call_node in ast.walk(node):
                            if isinstance(call_node, ast.Call):
                                source_key = f"{file_path}:{node.name}"
                                
                                if isinstance(call_node.func, ast.Name):
                                    target_key = self._find_entity_by_name(call_node.func.id, str(file_path))
                                elif isinstance(call_node.func, ast.Attribute):
                                    target_key = self._find_entity_by_name(self._get_attribute_name(call_node.func), str(file_path))
                                else:
                                    continue
                                    
                                if target_key and target_key != source_key:
                                    relationship = Relationship(
                                        source=source_key,
                                        target=target_key,
                                        relationship_type=RelationshipType.CALLS,
                                        context=f"Function {node.name} calls {call_node.func}",
                                        line_number=call_node.lineno
                                    )
                                    self.relationships.append(relationship)
                                    
            except Exception as e:
                if self.verbose:
                    dump(f"Error analyzing calls in {file_path}: {e}")
                    
    def _map_usage_relationships(self):
        """Map usage relationships between entities"""
        for entity_key, entity in self.static_analyzer.entities.items():
            for dep in entity.dependencies:
                target_key = self._find_entity_by_name(dep, entity.file_path)
                
                if target_key and target_key != entity_key:
                    relationship = Relationship(
                        source=entity_key,
                        target=target_key,
                        relationship_type=RelationshipType.USES,
                        context=f"{entity.name} uses {dep}",
                        line_number=entity.line_number
                    )
                    self.relationships.append(relationship)
                    
    def _map_creation_relationships(self):
        """Map object creation relationships"""
        python_files = [f for f in self.root_path.rglob('*.py') if f.is_file()]
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content, filename=str(file_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        # Look for class instantiation
                        if isinstance(node.func, ast.Name):
                            # Check if this is a class name
                            target_key = self._find_entity_by_name(node.func.id, str(file_path), entity_type='class')
                            
                            if target_key:
                                # Find the containing function
                                containing_func = self._find_containing_function(node, tree)
                                if containing_func:
                                    source_key = f"{file_path}:{containing_func}"
                                    
                                    relationship = Relationship(
                                        source=source_key,
                                        target=target_key,
                                        relationship_type=RelationshipType.CREATES,
                                        context=f"Function {containing_func} creates instance of {node.func.id}",
                                        line_number=node.lineno
                                    )
                                    self.relationships.append(relationship)
                                    
            except Exception as e:
                if self.verbose:
                    dump(f"Error analyzing creation in {file_path}: {e}")
                    
    def _map_reference_relationships(self):
        """Map reference relationships"""
        # This maps import relationships and other references
        for entity_key, entity in self.static_analyzer.entities.items():
            if entity.type == 'import':
                for dep in entity.dependencies:
                    target_key = self._find_entity_by_name(dep, entity.file_path)
                    
                    if target_key and target_key != entity_key:
                        relationship = Relationship(
                            source=entity_key,
                            target=target_key,
                            relationship_type=RelationshipType.REFERENCES,
                            context=f"Import {entity.name} references {dep}",
                            line_number=entity.line_number
                        )
                        self.relationships.append(relationship)
                        
    def _map_containment_relationships(self):
        """Map containment relationships (classes containing methods, etc.)"""
        for entity_key, entity in self.static_analyzer.entities.items():
            if entity.parent:
                parent_key = self._find_entity_by_name(entity.parent, entity.file_path)
                
                if parent_key:
                    relationship = Relationship(
                        source=parent_key,
                        target=entity_key,
                        relationship_type=RelationshipType.CONTAINS,
                        context=f"{entity.parent} contains {entity.name}",
                        line_number=entity.line_number
                    )
                    self.relationships.append(relationship)
                    
    def _find_entity_by_name(self, name: str, file_path: str, entity_type: Optional[str] = None) -> Optional[str]:
        """Find an entity by name and file path"""
        for entity_key, entity in self.static_analyzer.entities.items():
            if (entity.name == name and 
                entity.file_path == file_path and 
                (entity_type is None or entity.type == entity_type)):
                return entity_key
        return None
        
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get the full attribute name from an AST node"""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        else:
            return node.attr
            
    def _find_containing_function(self, node: ast.AST, tree: ast.AST) -> Optional[str]:
        """Find the containing function for a given AST node"""
        for parent in ast.walk(tree):
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if self._node_is_contained_in(node, parent):
                    return parent.name
        return None
        
    def _node_is_contained_in(self, node: ast.AST, parent: ast.AST) -> bool:
        """Check if a node is contained within a parent node"""
        # Simple check - in a real implementation, you'd need to traverse the AST properly
        return hasattr(parent, 'body') and node in getattr(parent, 'body', [])
        
    def _build_call_graph(self):
        """Build the call graph"""
        self.call_graph.clear()
        
        # Add nodes
        for entity_key, entity in self.static_analyzer.entities.items():
            if entity.type in ['function', 'method']:
                self.call_graph.add_node(entity_key, **entity.__dict__)
                
        # Add edges for call relationships
        for relationship in self.relationships:
            if relationship.relationship_type == RelationshipType.CALLS:
                if (relationship.source in self.call_graph.nodes and 
                    relationship.target in self.call_graph.nodes):
                    self.call_graph.add_edge(relationship.source, relationship.target)
                    
    def _build_inheritance_graph(self):
        """Build the inheritance graph"""
        self.inheritance_graph.clear()
        
        # Add nodes
        for entity_key, entity in self.static_analyzer.entities.items():
            if entity.type == 'class':
                self.inheritance_graph.add_node(entity_key, **entity.__dict__)
                
        # Add edges for inheritance relationships
        for relationship in self.relationships:
            if relationship.relationship_type == RelationshipType.INHERITS:
                if (relationship.source in self.inheritance_graph.nodes and 
                    relationship.target in self.inheritance_graph.nodes):
                    self.inheritance_graph.add_edge(relationship.source, relationship.target)
                    
    def _build_dependency_graph(self):
        """Build the dependency graph"""
        self.dependency_graph.clear()
        
        # Add nodes
        for entity_key, entity in self.static_analyzer.entities.items():
            self.dependency_graph.add_node(entity_key, **entity.__dict__)
            
        # Add edges for all dependency relationships
        for relationship in self.relationships:
            if (relationship.source in self.dependency_graph.nodes and 
                relationship.target in self.dependency_graph.nodes):
                self.dependency_graph.add_edge(
                    relationship.source, 
                    relationship.target,
                    relationship_type=relationship.relationship_type.value,
                    strength=relationship.strength
                )
                
    def _generate_entity_relationship_summaries(self):
        """Generate relationship summaries for each entity"""
        # Initialize entity relationships
        for entity_key, entity in self.static_analyzer.entities.items():
            self.entity_relationships[entity_key] = EntityRelationships(
                entity_key=entity_key,
                entity=entity
            )
            
        # Populate relationships
        for relationship in self.relationships:
            if relationship.source in self.entity_relationships:
                self.entity_relationships[relationship.source].outgoing_relationships.append(relationship)
                
            if relationship.target in self.entity_relationships:
                self.entity_relationships[relationship.target].incoming_relationships.append(relationship)
                
        # Generate summaries
        for entity_key, entity_rel in self.entity_relationships.items():
            summary = defaultdict(int)
            
            for rel in entity_rel.outgoing_relationships:
                summary[f"outgoing_{rel.relationship_type.value}"] += 1
                
            for rel in entity_rel.incoming_relationships:
                summary[f"incoming_{rel.relationship_type.value}"] += 1
                
            entity_rel.relationship_summary = dict(summary)
            
    def _analyze_relationships(self):
        """Analyze relationships and calculate strengths"""
        # Calculate relationship strengths based on frequency and context
        relationship_counts = Counter()
        
        for relationship in self.relationships:
            key = (relationship.source, relationship.target, relationship.relationship_type)
            relationship_counts[key] += 1
            
        # Update relationship strengths
        for relationship in self.relationships:
            key = (relationship.source, relationship.target, relationship.relationship_type)
            count = relationship_counts[key]
            
            # Strength based on frequency (with diminishing returns)
            relationship.strength = min(1.0, count * 0.3)
            
            # Adjust strength based on relationship type
            if relationship.relationship_type == RelationshipType.INHERITS:
                relationship.strength *= 1.5
            elif relationship.relationship_type == RelationshipType.CALLS:
                relationship.strength *= 1.2
                
            relationship.strength = min(1.0, relationship.strength)
            
    def _generate_relationship_insights(self) -> List[str]:
        """Generate insights about the relationships"""
        insights = []
        
        # Total relationships
        insights.append(f"Total relationships mapped: {len(self.relationships)}")
        
        # Relationship type distribution
        type_counts = Counter(r.relationship_type for r in self.relationships)
        most_common_type = type_counts.most_common(1)[0] if type_counts else None
        if most_common_type:
            insights.append(f"Most common relationship type: {most_common_type[0].value} ({most_common_type[1]} occurrences)")
            
        # Graph analysis
        if self.call_graph.number_of_nodes() > 0:
            avg_degree = sum(dict(self.call_graph.degree()).values()) / self.call_graph.number_of_nodes()
            insights.append(f"Average call graph degree: {avg_degree:.2f}")
            
        # Inheritance depth
        if self.inheritance_graph.number_of_nodes() > 0:
            try:
                longest_path = nx.dag_longest_path(self.inheritance_graph)
                insights.append(f"Maximum inheritance depth: {len(longest_path) - 1}")
            except nx.NetworkXError:
                insights.append("Inheritance graph contains cycles")
                
        # Highly connected entities
        if self.dependency_graph.number_of_nodes() > 0:
            degrees = dict(self.dependency_graph.degree())
            max_degree = max(degrees.values()) if degrees else 0
            highly_connected = [k for k, v in degrees.items() if v == max_degree]
            if highly_connected:
                entity_names = [self.static_analyzer.entities[k].name for k in highly_connected[:3]]
                insights.append(f"Most connected entities: {', '.join(entity_names)}")
                
        return insights
        
    def _identify_critical_relationships(self) -> List[Dict[str, Any]]:
        """Identify critical relationships that might need attention"""
        critical_relationships = []
        
        # Find relationships with high strength
        high_strength_relationships = [r for r in self.relationships if r.strength > 0.8]
        
        for relationship in high_strength_relationships[:10]:  # Top 10
            source_entity = self.static_analyzer.entities.get(relationship.source)
            target_entity = self.static_analyzer.entities.get(relationship.target)
            
            if source_entity and target_entity:
                critical_relationships.append({
                    'source': source_entity.name,
                    'target': target_entity.name,
                    'type': relationship.relationship_type.value,
                    'strength': relationship.strength,
                    'context': relationship.context
                })
                
        return critical_relationships
        
    def get_entity_relationships(self, entity_name: str, file_path: str) -> Dict[str, Any]:
        """Get detailed relationships for a specific entity"""
        entity_key = self._find_entity_by_name(entity_name, file_path)
        
        if not entity_key or entity_key not in self.entity_relationships:
            return {}
            
        entity_rel = self.entity_relationships[entity_key]
        
        return {
            'entity': entity_rel.entity.__dict__,
            'incoming_relationships': [
                {
                    'source': self.static_analyzer.entities.get(r.source).__dict__,
                    'type': r.relationship_type.value,
                    'strength': r.strength,
                    'context': r.context
                }
                for r in entity_rel.incoming_relationships
            ],
            'outgoing_relationships': [
                {
                    'target': self.static_analyzer.entities.get(r.target).__dict__,
                    'type': r.relationship_type.value,
                    'strength': r.strength,
                    'context': r.context
                }
                for r in entity_rel.outgoing_relationships
            ],
            'relationship_summary': entity_rel.relationship_summary
        }
        
    def get_relationship_path(self, source_entity: str, target_entity: str, 
                             relationship_type: Optional[RelationshipType] = None) -> List[str]:
        """Find a path between two entities based on relationships"""
        source_key = self._find_entity_by_name(source_entity, self.root_path)
        target_key = self._find_entity_by_name(target_entity, self.root_path)
        
        if not source_key or not target_key:
            return []
            
        # Choose the appropriate graph
        if relationship_type == RelationshipType.CALLS:
            graph = self.call_graph
        elif relationship_type == RelationshipType.INHERITS:
            graph = self.inheritance_graph
        else:
            graph = self.dependency_graph
            
        try:
            path = nx.shortest_path(graph, source_key, target_key)
            return [self.static_analyzer.entities[k].name for k in path]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
            
    def get_relationship_statistics(self) -> Dict[str, Any]:
        """Get comprehensive relationship statistics"""
        stats = {
            'total_relationships': len(self.relationships),
            'relationship_types': dict(Counter(r.relationship_type.value for r in self.relationships)),
            'graph_statistics': {
                'call_graph': {
                    'nodes': self.call_graph.number_of_nodes(),
                    'edges': self.call_graph.number_of_edges(),
                    'density': nx.density(self.call_graph)
                },
                'inheritance_graph': {
                    'nodes': self.inheritance_graph.number_of_nodes(),
                    'edges': self.inheritance_graph.number_of_edges(),
                    'density': nx.density(self.inheritance_graph)
                },
                'dependency_graph': {
                    'nodes': self.dependency_graph.number_of_nodes(),
                    'edges': self.dependency_graph.number_of_edges(),
                    'density': nx.density(self.dependency_graph)
                }
            }
        }
        
        # Add centrality measures
        if self.dependency_graph.number_of_nodes() > 0:
            centrality = nx.degree_centrality(self.dependency_graph)
            stats['most_central_entities'] = [
                {
                    'entity': self.static_analyzer.entities[k].name,
                    'centrality': v
                }
                for k, v in sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
            
        return stats
