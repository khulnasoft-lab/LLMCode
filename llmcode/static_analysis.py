"""
Enhanced Static Analysis Module for LLMCode

This module provides advanced static analysis capabilities including:
- AST parsing for multiple programming languages
- Dependency analysis and graph generation
- Code structure understanding
- Function/class relationship mapping
"""

import ast
import os
import importlib.util
import networkx as nx
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict, Counter
import re

from llmcode.dump import dump


@dataclass
class CodeEntity:
    """Represents a code entity (function, class, variable, etc.)"""
    name: str
    type: str  # 'function', 'class', 'method', 'variable', 'import'
    file_path: str
    line_number: int
    end_line_number: Optional[int] = None
    parent: Optional[str] = None
    dependencies: Set[str] = None
    dependents: Set[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = set()
        if self.dependents is None:
            self.dependents = set()


@dataclass
class ImportDependency:
    """Represents an import dependency"""
    module: str
    alias: Optional[str] = None
    from_module: Optional[str] = None
    file_path: str
    line_number: int
    is_relative: bool = False
    is_standard_library: bool = False
    is_third_party: bool = False
    is_local: bool = False


class StaticAnalyzer:
    """Main static analysis class for code understanding and dependency analysis"""
    
    def __init__(self, root_path: str, verbose: bool = False):
        self.root_path = Path(root_path)
        self.verbose = verbose
        self.entities: Dict[str, CodeEntity] = {}
        self.imports: List[ImportDependency] = []
        self.dependency_graph = nx.DiGraph()
        self.file_entities: Dict[str, List[CodeEntity]] = defaultdict(list)
        
        # Standard library modules (basic list, can be expanded)
        self.stdlib_modules = {
            'os', 'sys', 'json', 're', 'math', 'datetime', 'collections',
            'itertools', 'functools', 'operator', 'pathlib', 'typing',
            'dataclasses', 'enum', 'abc', 'contextlib', 'warnings',
            'logging', 'unittest', 'argparse', 'configparser', 'csv',
            'sqlite3', 'pickle', 'shelve', 'tempfile', 'shutil', 'glob',
            'random', 'statistics', 'decimal', 'fractions', 'array',
            'struct', 'socket', 'ssl', 'hashlib', 'hmac', 'secrets',
            'threading', 'multiprocessing', 'concurrent', 'asyncio',
            'subprocess', 'signal', 'queue', 'time', 'calendar', 'zoneinfo'
        }
        
    def analyze_file(self, file_path: Union[str, Path]) -> List[CodeEntity]:
        """Analyze a single file and extract code entities and dependencies"""
        file_path = Path(file_path)
        if not file_path.exists():
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            if self.verbose:
                dump(f"Error reading file {file_path}: {e}")
            return []
            
        file_ext = file_path.suffix.lower()
        entities = []
        
        if file_ext == '.py':
            entities = self._analyze_python_file(file_path, content)
        elif file_ext in ['.js', '.ts', '.jsx', '.tsx']:
            entities = self._analyze_javascript_file(file_path, content)
        elif file_ext in ['.java']:
            entities = self._analyze_java_file(file_path, content)
        elif file_ext in ['.cpp', '.cc', '.cxx', '.c++', '.h', '.hpp']:
            entities = self._analyze_cpp_file(file_path, content)
        else:
            # For unsupported files, return basic file info
            entities = [CodeEntity(
                name=file_path.name,
                type='file',
                file_path=str(file_path),
                line_number=1
            )]
            
        self.file_entities[str(file_path)] = entities
        for entity in entities:
            self.entities[f"{entity.file_path}:{entity.name}"] = entity
            
        return entities
        
    def _analyze_python_file(self, file_path: Path, content: str) -> List[CodeEntity]:
        """Analyze Python file using AST"""
        entities = []
        
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            if self.verbose:
                dump(f"Syntax error in {file_path}: {e}")
            return entities
            
        # Extract imports first
        import_entities = self._extract_python_imports(file_path, tree)
        entities.extend(import_entities)
        
        # Extract classes and functions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_entity = self._extract_python_class(file_path, node)
                entities.append(class_entity)
                
            elif isinstance(node, ast.FunctionDef):
                func_entity = self._extract_python_function(file_path, node)
                entities.append(func_entity)
                
            elif isinstance(node, ast.AsyncFunctionDef):
                async_func_entity = self._extract_python_function(file_path, node, is_async=True)
                entities.append(async_func_entity)
                
        return entities
        
    def _extract_python_imports(self, file_path: Path, tree: ast.AST) -> List[CodeEntity]:
        """Extract import statements from Python AST"""
        entities = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_dep = ImportDependency(
                        module=alias.name,
                        alias=alias.asname,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        is_standard_library=alias.name in self.stdlib_modules,
                        is_third_party=alias.name not in self.stdlib_modules and not alias.name.startswith('.'),
                        is_local=alias.name.startswith('.')
                    )
                    self.imports.append(import_dep)
                    
                    entity = CodeEntity(
                        name=alias.asname or alias.name,
                        type='import',
                        file_path=str(file_path),
                        line_number=node.lineno,
                        dependencies={alias.name}
                    )
                    entities.append(entity)
                    
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    full_name = f"{module}.{alias.name}" if module else alias.name
                    import_dep = ImportDependency(
                        module=full_name,
                        alias=alias.asname,
                        from_module=module,
                        file_path=str(file_path),
                        line_number=node.lineno,
                        is_relative=node.level > 0,
                        is_standard_library=module in self.stdlib_modules,
                        is_third_party=module not in self.stdlib_modules and module,
                        is_local=node.level > 0
                    )
                    self.imports.append(import_dep)
                    
                    entity = CodeEntity(
                        name=alias.asname or alias.name,
                        type='import',
                        file_path=str(file_path),
                        line_number=node.lineno,
                        dependencies={full_name}
                    )
                    entities.append(entity)
                    
        return entities
        
    def _extract_python_class(self, file_path: Path, node: ast.ClassDef) -> CodeEntity:
        """Extract class information from Python AST"""
        # Extract base classes
        dependencies = set()
        for base in node.bases:
            if isinstance(base, ast.Name):
                dependencies.add(base.id)
            elif isinstance(base, ast.Attribute):
                dependencies.add(self._get_attribute_name(base))
                
        entity = CodeEntity(
            name=node.name,
            type='class',
            file_path=str(file_path),
            line_number=node.lineno,
            end_line_number=getattr(node, 'end_lineno', None),
            dependencies=dependencies
        )
        
        # Extract methods and nested classes
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_entity = self._extract_python_function(file_path, item, parent=node.name)
                entity.dependencies.add(method_entity.name)
            elif isinstance(item, ast.AsyncFunctionDef):
                method_entity = self._extract_python_function(file_path, item, parent=node.name, is_async=True)
                entity.dependencies.add(method_entity.name)
            elif isinstance(item, ast.ClassDef):
                nested_class = self._extract_python_class(file_path, item)
                nested_class.parent = node.name
                entity.dependencies.add(nested_class.name)
                
        return entity
        
    def _extract_python_function(self, file_path: Path, node: ast.FunctionDef, 
                                parent: Optional[str] = None, is_async: bool = False) -> CodeEntity:
        """Extract function information from Python AST"""
        dependencies = set()
        
        # Extract function arguments
        for arg in node.args.args:
            dependencies.add(arg.arg)
            
        # Extract function calls and references in the body
        for body_node in ast.walk(node):
            if isinstance(body_node, ast.Name):
                dependencies.add(body_node.id)
            elif isinstance(body_node, ast.Attribute):
                dependencies.add(self._get_attribute_name(body_node))
            elif isinstance(body_node, ast.Call):
                if isinstance(body_node.func, ast.Name):
                    dependencies.add(body_node.func.id)
                elif isinstance(body_node.func, ast.Attribute):
                    dependencies.add(self._get_attribute_name(body_node.func))
                    
        entity = CodeEntity(
            name=node.name,
            type='method' if parent else 'function',
            file_path=str(file_path),
            line_number=node.lineno,
            end_line_number=getattr(node, 'end_lineno', None),
            parent=parent,
            dependencies=dependencies
        )
        
        return entity
        
    def _get_attribute_name(self, node: ast.Attribute) -> str:
        """Get the full attribute name from an AST node"""
        if isinstance(node.value, ast.Name):
            return f"{node.value.id}.{node.attr}"
        elif isinstance(node.value, ast.Attribute):
            return f"{self._get_attribute_name(node.value)}.{node.attr}"
        else:
            return node.attr
            
    def _analyze_javascript_file(self, file_path: Path, content: str) -> List[CodeEntity]:
        """Analyze JavaScript/TypeScript file (basic implementation)"""
        entities = []
        
        # Basic regex-based extraction for JavaScript
        # This is a simplified version - in production, use a proper JS parser
        
        # Function declarations
        func_pattern = r'(?:async\s+)?function\s+(\w+)\s*\('
        for match in re.finditer(func_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            entities.append(CodeEntity(
                name=match.group(1),
                type='function',
                file_path=str(file_path),
                line_number=line_num
            ))
            
        # Class declarations
        class_pattern = r'class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            entities.append(CodeEntity(
                name=match.group(1),
                type='class',
                file_path=str(file_path),
                line_number=line_num
            ))
            
        # Import statements
        import_pattern = r'import\s+(?:.*\s+from\s+)?[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(import_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            import_dep = ImportDependency(
                module=match.group(1),
                file_path=str(file_path),
                line_number=line_num,
                is_third_party=not match.group(1).startswith('.')
            )
            self.imports.append(import_dep)
            
            entities.append(CodeEntity(
                name=match.group(1),
                type='import',
                file_path=str(file_path),
                line_number=line_num,
                dependencies={match.group(1)}
            ))
            
        return entities
        
    def _analyze_java_file(self, file_path: Path, content: str) -> List[CodeEntity]:
        """Analyze Java file (basic implementation)"""
        entities = []
        
        # Basic regex-based extraction for Java
        # This is a simplified version - in production, use a proper Java parser
        
        # Class declarations
        class_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:abstract\s+|final\s+)?class\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            entities.append(CodeEntity(
                name=match.group(1),
                type='class',
                file_path=str(file_path),
                line_number=line_num
            ))
            
        # Method declarations
        method_pattern = r'(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:\w+(?:<[^>]+>)?\s+)+(\w+)\s*\('
        for match in re.finditer(method_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            entities.append(CodeEntity(
                name=match.group(1),
                type='method',
                file_path=str(file_path),
                line_number=line_num
            ))
            
        # Import statements
        import_pattern = r'import\s+(?:static\s+)?([^;]+);'
        for match in re.finditer(import_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            import_dep = ImportDependency(
                module=match.group(1),
                file_path=str(file_path),
                line_number=line_num,
                is_standard_library=match.group(1).startswith('java.')
            )
            self.imports.append(import_dep)
            
            entities.append(CodeEntity(
                name=match.group(1),
                type='import',
                file_path=str(file_path),
                line_number=line_num,
                dependencies={match.group(1)}
            ))
            
        return entities
        
    def _analyze_cpp_file(self, file_path: Path, content: str) -> List[CodeEntity]:
        """Analyze C++ file (basic implementation)"""
        entities = []
        
        # Basic regex-based extraction for C++
        # This is a simplified version - in production, use a proper C++ parser
        
        # Class declarations
        class_pattern = r'(?:class|struct)\s+(\w+)'
        for match in re.finditer(class_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            entities.append(CodeEntity(
                name=match.group(1),
                type='class',
                file_path=str(file_path),
                line_number=line_num
            ))
            
        # Function declarations
        func_pattern = r'(?:\w+(?:<[^>]+>)?\s+(?:\*\s*)?)+(\w+)\s*\('
        for match in re.finditer(func_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            entities.append(CodeEntity(
                name=match.group(1),
                type='function',
                file_path=str(file_path),
                line_number=line_num
            ))
            
        # Include statements
        include_pattern = r'#include\s*[<"]([^>"]+)[>"]'
        for match in re.finditer(include_pattern, content):
            line_num = content[:match.start()].count('\n') + 1
            import_dep = ImportDependency(
                module=match.group(1),
                file_path=str(file_path),
                line_number=line_num,
                is_standard_library=match.group(1).startswith('<')
            )
            self.imports.append(import_dep)
            
            entities.append(CodeEntity(
                name=match.group(1),
                type='import',
                file_path=str(file_path),
                line_number=line_num,
                dependencies={match.group(1)}
            ))
            
        return entities
        
    def build_dependency_graph(self) -> nx.DiGraph:
        """Build a dependency graph from analyzed entities"""
        self.dependency_graph.clear()
        
        # Add nodes for all entities
        for entity_key, entity in self.entities.items():
            self.dependency_graph.add_node(entity_key, **entity.__dict__)
            
        # Add edges for dependencies
        for entity_key, entity in self.entities.items():
            for dep in entity.dependencies:
                # Find the dependency entity
                dep_entity_key = None
                for other_key, other_entity in self.entities.items():
                    if other_entity.name == dep and other_entity.file_path == entity.file_path:
                        dep_entity_key = other_key
                        break
                        
                if dep_entity_key:
                    self.dependency_graph.add_edge(entity_key, dep_entity_key)
                    
        return self.dependency_graph
        
    def get_file_dependencies(self, file_path: str) -> Dict[str, Set[str]]:
        """Get dependencies for a specific file"""
        dependencies = defaultdict(set)
        
        for entity in self.file_entities.get(file_path, []):
            for dep in entity.dependencies:
                dependencies[entity.name].add(dep)
                
        return dict(dependencies)
        
    def get_entity_relationships(self, entity_name: str) -> Dict[str, Set[str]]:
        """Get relationships for a specific entity"""
        relationships = {
            'dependencies': set(),
            'dependents': set(),
            'related_files': set()
        }
        
        # Find the entity
        entity_key = None
        entity_obj = None
        for key, entity in self.entities.items():
            if entity.name == entity_name:
                entity_key = key
                entity_obj = entity
                break
                
        if not entity_obj:
            return relationships
            
        # Get dependencies
        relationships['dependencies'] = entity_obj.dependencies
        
        # Get dependents
        for other_key, other_entity in self.entities.items():
            if entity_name in other_entity.dependencies:
                relationships['dependents'].add(other_entity.name)
                
        # Get related files
        for other_key, other_entity in self.entities.items():
            if (entity_name in other_entity.dependencies or 
                other_entity.name in entity_obj.dependencies):
                relationships['related_files'].add(other_entity.file_path)
                
        return relationships
        
    def analyze_project(self, file_patterns: List[str] = None) -> Dict[str, Any]:
        """Analyze the entire project"""
        if file_patterns is None:
            file_patterns = ['**/*.py', '**/*.js', '**/*.ts', '**/*.jsx', '**/*.tsx', 
                           '**/*.java', '**/*.cpp', '**/*.cc', '**/*.cxx', '**/*.c++', 
                           '**/*.h', '**/*.hpp']
                           
        results = {
            'total_files': 0,
            'total_entities': 0,
            'total_imports': 0,
            'files_analyzed': [],
            'entities_by_type': defaultdict(int),
            'imports_by_type': defaultdict(int),
            'dependency_graph': None
        }
        
        # Find and analyze files
        for pattern in file_patterns:
            for file_path in self.root_path.rglob(pattern):
                if file_path.is_file():
                    entities = self.analyze_file(file_path)
                    results['total_files'] += 1
                    results['total_entities'] += len(entities)
                    results['files_analyzed'].append(str(file_path))
                    
                    for entity in entities:
                        results['entities_by_type'][entity.type] += 1
                        
        results['total_imports'] = len(self.imports)
        for imp in self.imports:
            if imp.is_standard_library:
                results['imports_by_type']['standard_library'] += 1
            elif imp.is_third_party:
                results['imports_by_type']['third_party'] += 1
            elif imp.is_local:
                results['imports_by_type']['local'] += 1
                
        # Build dependency graph
        results['dependency_graph'] = self.build_dependency_graph()
        
        return results
