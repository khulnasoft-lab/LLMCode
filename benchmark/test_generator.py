#!/usr/bin/env python3
"""
Enhanced Test Generation Framework for Llmcode

This module extends the existing benchmark framework with AI-powered test generation,
coverage analysis, and template-based test creation capabilities.
"""

import ast
import json
import os
import re
import sys
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
import subprocess
import threading
import queue
import concurrent.futures
from collections import defaultdict, Counter

import astor
import coverage
import pytest
import yaml
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

# Import llmcode modules conditionally
try:
    from llmcode import models, sendchat
    from llmcode.io import InputOutput
    from llmcode.coders import Coder, base_coder
    LLMCODE_AVAILABLE = True
except ImportError:
    LLMCODE_AVAILABLE = False
    # Create dummy classes for when llmcode is not available
    class models:
        @staticmethod
        def get_model_info():
            return None
    
    def sendchat(*args, **kwargs):
        return None
    
    class InputOutput:
        def __init__(self, *args, **kwargs):
            pass
    
    class Coder:
        def __init__(self, *args, **kwargs):
            pass
    
    class base_coder:
        pass


class TestType(Enum):
    """Types of tests that can be generated"""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    REGRESSION = "regression"


class TestFramework(Enum):
    """Supported testing frameworks"""
    PYTEST = "pytest"
    UNITEEST = "unittest"
    NOSE = "nose"
    CUSTOM = "custom"


@dataclass
class TestTemplate:
    """Template for generating tests"""
    name: str
    description: str
    framework: TestFramework
    template_code: str
    variables: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    language: str = "python"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestTemplate':
        """Create from dictionary"""
        data['framework'] = TestFramework(data['framework'])
        return cls(**data)


@dataclass
class GeneratedTest:
    """Generated test with metadata"""
    name: str
    code: str
    test_type: TestType
    framework: TestFramework
    file_path: str
    line_number: int
    coverage_target: str
    confidence_score: float
    execution_time: Optional[float] = None
    passed: Optional[bool] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['test_type'] = self.test_type.value
        data['framework'] = self.framework.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GeneratedTest':
        """Create from dictionary"""
        data['test_type'] = TestType(data['test_type'])
        data['framework'] = TestFramework(data['framework'])
        return cls(**data)


@dataclass
class CoverageAnalysis:
    """Coverage analysis results"""
    total_lines: int
    covered_lines: int
    missed_lines: int
    coverage_percentage: float
    uncovered_files: List[str]
    uncovered_functions: List[str]
    uncovered_branches: List[str]
    complexity_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


class TestGenerator(ABC):
    """Abstract base class for test generators"""
    
    @abstractmethod
    def generate_tests(self, code: str, file_path: str, test_type: TestType) -> List[GeneratedTest]:
        """Generate tests for given code"""
        pass
    
    @abstractmethod
    def get_confidence_score(self, test: GeneratedTest) -> float:
        """Get confidence score for generated test"""
        pass


class AITestGenerator(TestGenerator):
    """AI-powered test generator using LLM"""
    
    def __init__(self, model_name: str = "gpt-4", io: Optional[InputOutput] = None):
        """
        Initialize AI test generator
        
        Args:
            model_name: Name of the LLM model to use
            io: InputOutput instance for logging
        """
        self.model_name = model_name
        self.io = io or InputOutput()
        self.console = Console()
        
        # Check if llmcode is available
        if not LLMCODE_AVAILABLE:
            self.io.tool_warning("llmcode modules not available. AI test generation will be disabled.")
            self.enabled = False
        else:
            self.enabled = True
        
        # Test generation prompts
        self.prompts = {
            TestType.UNIT: self._get_unit_test_prompt(),
            TestType.INTEGRATION: self._get_integration_test_prompt(),
            TestType.FUNCTIONAL: self._get_functional_test_prompt(),
            TestType.PERFORMANCE: self._get_performance_test_prompt(),
            TestType.SECURITY: self._get_security_test_prompt(),
            TestType.REGRESSION: self._get_regression_test_prompt()
        }
    
    def _get_unit_test_prompt(self) -> str:
        """Get prompt for unit test generation"""
        return """
You are an expert Python test developer. Generate comprehensive unit tests for the following code.

Requirements:
1. Use pytest framework
2. Test all public methods and functions
3. Include edge cases and error conditions
4. Use meaningful test names
5. Add proper assertions
6. Include setup/teardown if needed
7. Mock external dependencies

Code to test:
{code}

File path: {file_path}

Generate the test code in a single code block:
```python
# Your test code here
```
"""
    
    def _get_integration_test_prompt(self) -> str:
        """Get prompt for integration test generation"""
        return """
You are an expert Python test developer. Generate integration tests for the following code.

Requirements:
1. Use pytest framework
2. Test interactions between components
3. Include database/API integration tests if applicable
4. Test error handling and recovery
5. Use fixtures for setup
6. Include cleanup/teardown

Code to test:
{code}

File path: {file_path}

Generate the test code in a single code block:
```python
# Your test code here
```
"""
    
    def _get_functional_test_prompt(self) -> str:
        """Get prompt for functional test generation"""
        return """
You are an expert Python test developer. Generate functional tests for the following code.

Requirements:
1. Use pytest framework
2. Test user scenarios and workflows
3. Focus on business logic and requirements
4. Include end-to-end testing
5. Test data validation and business rules

Code to test:
{code}

File path: {file_path}

Generate the test code in a single code block:
```python
# Your test code here
```
"""
    
    def _get_performance_test_prompt(self) -> str:
        """Get prompt for performance test generation"""
        return """
You are an expert Python test developer. Generate performance tests for the following code.

Requirements:
1. Use pytest framework with pytest-benchmark
2. Test execution time and resource usage
3. Include load testing scenarios
4. Test memory usage and leaks
5. Include performance assertions

Code to test:
{code}

File path: {file_path}

Generate the test code in a single code block:
```python
# Your test code here
```
"""
    
    def _get_security_test_prompt(self) -> str:
        """Get prompt for security test generation"""
        return """
You are an expert Python test developer. Generate security tests for the following code.

Requirements:
1. Use pytest framework
2. Test input validation and sanitization
3. Include authentication/authorization tests
4. Test for common vulnerabilities (SQL injection, XSS, etc.)
5. Include data encryption tests

Code to test:
{code}

File path: {file_path}

Generate the test code in a single code block:
```python
# Your test code here
```
"""
    
    def _get_regression_test_prompt(self) -> str:
        """Get prompt for regression test generation"""
        return """
You are an expert Python test developer. Generate regression tests for the following code.

Requirements:
1. Use pytest framework
2. Test for known bugs and issues
3. Include smoke tests
4. Test backward compatibility
5. Include configuration testing

Code to test:
{code}

File path: {file_path}

Generate the test code in a single code block:
```python
# Your test code here
```
"""
    
    def generate_tests(self, code: str, file_path: str, test_type: TestType) -> List[GeneratedTest]:
        """Generate tests using AI"""
        # Check if AI generation is enabled
        if not self.enabled:
            if self.io:
                self.io.tool_warning("AI test generation is disabled (llmcode modules not available)")
            return []
        
        try:
            # Get prompt for test type
            prompt = self.prompts[test_type].format(code=code, file_path=file_path)
            
            # Generate test using LLM
            response = sendchat.send_with_retries(
                model_name=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                functions=None,
                verbose=False,
                io=self.io
            )
            
            # Extract test code from response
            test_code = self._extract_test_code(response)
            if not test_code:
                return []
            
            # Parse generated tests
            tests = self._parse_generated_tests(test_code, file_path, test_type)
            
            return tests
            
        except Exception as e:
            if self.io:
                self.io.tool_warning(f"Failed to generate AI tests: {e}")
            return []
    
    def _extract_test_code(self, response: str) -> Optional[str]:
        """Extract test code from LLM response"""
        # Look for code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        # Look for indented code blocks
        code_blocks = re.findall(r'```\n(.*?)\n```', response, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        return None
    
    def _parse_generated_tests(self, test_code: str, file_path: str, test_type: TestType) -> List[GeneratedTest]:
        """Parse generated tests into structured format"""
        tests = []
        
        try:
            # Parse AST to find test functions
            tree = ast.parse(test_code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    # Extract function code
                    func_code = astor.get_source_segment(test_code, node)
                    
                    test = GeneratedTest(
                        name=node.name,
                        code=func_code,
                        test_type=test_type,
                        framework=TestFramework.PYTEST,
                        file_path=file_path,
                        line_number=node.lineno,
                        coverage_target=file_path,
                        confidence_score=self._calculate_confidence_score(func_code)
                    )
                    tests.append(test)
                    
        except Exception as e:
            if self.io:
                self.io.tool_warning(f"Failed to parse generated tests: {e}")
        
        return tests
    
    def _calculate_confidence_score(self, test_code: str) -> float:
        """Calculate confidence score for generated test"""
        score = 0.0
        
        # Check for assertions
        if 'assert' in test_code:
            score += 0.3
        
        # Check for proper test structure
        if 'def test_' in test_code:
            score += 0.2
        
        # Check for error handling
        if 'with pytest.raises' in test_code or 'except' in test_code:
            score += 0.2
        
        # Check for fixtures
        if '@pytest.fixture' in test_code:
            score += 0.1
        
        # Check for mocking
        if 'mock' in test_code or 'patch' in test_code:
            score += 0.1
        
        # Check for comments
        if '#' in test_code:
            score += 0.1
        
        return min(score, 1.0)
    
    def get_confidence_score(self, test: GeneratedTest) -> float:
        """Get confidence score for generated test"""
        return test.confidence_score


class TemplateBasedTestGenerator(TestGenerator):
    """Template-based test generator"""
    
    def __init__(self, template_dir: str = "test_templates"):
        """
        Initialize template-based test generator
        
        Args:
            template_dir: Directory containing test templates
        """
        self.template_dir = Path(template_dir)
        self.templates: Dict[str, TestTemplate] = {}
        self._load_templates()
    
    def _load_templates(self):
        """Load test templates from directory"""
        if not self.template_dir.exists():
            return
        
        for template_file in self.template_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r') as f:
                    template_data = yaml.safe_load(f)
                    template = TestTemplate.from_dict(template_data)
                    self.templates[template.name] = template
            except Exception as e:
                print(f"Failed to load template {template_file}: {e}")
    
    def generate_tests(self, code: str, file_path: str, test_type: TestType) -> List[GeneratedTest]:
        """Generate tests using templates"""
        tests = []
        
        # Find relevant templates
        relevant_templates = [
            template for template in self.templates.values()
            if template.language == "python" and test_type.value in template.tags
        ]
        
        for template in relevant_templates:
            try:
                # Generate test from template
                test_code = self._apply_template(template, code, file_path)
                
                test = GeneratedTest(
                    name=f"test_{template.name}_{len(tests)}",
                    code=test_code,
                    test_type=test_type,
                    framework=template.framework,
                    file_path=file_path,
                    line_number=1,
                    coverage_target=file_path,
                    confidence_score=0.7  # Default confidence for template-based tests
                )
                tests.append(test)
                
            except Exception as e:
                print(f"Failed to generate test from template {template.name}: {e}")
        
        return tests
    
    def _apply_template(self, template: TestTemplate, code: str, file_path: str) -> str:
        """Apply template to generate test code"""
        test_code = template.template_code
        
        # Replace variables
        variables = {
            'code': code,
            'file_path': file_path,
            'file_name': Path(file_path).stem,
            'timestamp': str(int(time.time()))
        }
        
        for var in template.variables:
            if var in variables:
                test_code = test_code.replace(f"{{{var}}}", variables[var])
        
        return test_code
    
    def get_confidence_score(self, test: GeneratedTest) -> float:
        """Get confidence score for generated test"""
        return test.confidence_score


class CoverageAnalyzer:
    """Code coverage analysis tool"""
    
    def __init__(self, project_root: str = "."):
        """
        Initialize coverage analyzer
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.cov = coverage.Coverage()
        self.console = Console()
    
    def analyze_coverage(self, test_files: List[str], source_files: List[str]) -> CoverageAnalysis:
        """Analyze code coverage"""
        try:
            # Start coverage collection
            self.cov.start()
            
            # Run tests
            self._run_tests(test_files)
            
            # Stop coverage collection
            self.cov.stop()
            self.cov.save()
            
            # Generate coverage report
            total_statements = 0
            covered_statements = 0
            missed_statements = 0
            
            uncovered_files = []
            uncovered_functions = []
            uncovered_branches = []
            
            # Analyze each source file
            for source_file in source_files:
                file_analysis = self._analyze_file_coverage(source_file)
                if file_analysis:
                    total_statements += file_analysis['total_statements']
                    covered_statements += file_analysis['covered_statements']
                    missed_statements += file_analysis['missed_statements']
                    
                    if file_analysis['coverage_percentage'] < 100:
                        uncovered_files.append(source_file)
                        uncovered_functions.extend(file_analysis['uncovered_functions'])
                        uncovered_branches.extend(file_analysis['uncovered_branches'])
            
            # Calculate coverage percentage
            coverage_percentage = (covered_statements / total_statements * 100) if total_statements > 0 else 0
            
            # Calculate complexity score
            complexity_score = self._calculate_complexity_score(source_files)
            
            return CoverageAnalysis(
                total_lines=total_statements,
                covered_lines=covered_statements,
                missed_lines=missed_statements,
                coverage_percentage=coverage_percentage,
                uncovered_files=uncovered_files,
                uncovered_functions=uncovered_functions,
                uncovered_branches=uncovered_branches,
                complexity_score=complexity_score
            )
            
        except Exception as e:
            self.console.print(f"[red]Coverage analysis failed: {e}[/red]")
            return CoverageAnalysis(0, 0, 0, 0, [], [], [], 0)
    
    def _run_tests(self, test_files: List[str]):
        """Run tests with coverage collection"""
        try:
            # Run pytest with coverage
            cmd = ['python', '-m', 'pytest'] + test_files + ['--cov=.']
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                self.console.print(f"[yellow]Tests failed: {result.stderr}[/yellow]")
                
        except Exception as e:
            self.console.print(f"[red]Failed to run tests: {e}[/red]")
    
    def _analyze_file_coverage(self, source_file: str) -> Optional[Dict[str, Any]]:
        """Analyze coverage for a specific file"""
        try:
            # Get coverage data for file
            analysis = self.cov.analysis2(source_file)
            
            if not analysis:
                return None
            
            total_statements = len(analysis[1])  # Executable lines
            covered_statements = len(analysis[2])  # Missing lines
            missed_statements = total_statements - covered_statements
            
            coverage_percentage = (covered_statements / total_statements * 100) if total_statements > 0 else 0
            
            # Parse source code to find functions
            uncovered_functions = []
            uncovered_branches = []
            
            try:
                with open(source_file, 'r') as f:
                    source_code = f.read()
                
                tree = ast.parse(source_code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check if function is covered
                        func_start = node.lineno
                        func_end = getattr(node, 'end_lineno', func_start)
                        
                        # Check if any line in function is missed
                        func_missed = any(
                            line in analysis[2]  # Missing lines
                            for line in range(func_start, func_end + 1)
                        )
                        
                        if func_missed:
                            uncovered_functions.append(node.name)
                    
                    elif isinstance(node, ast.If):
                        # Check if branches are covered
                        if_start = node.lineno
                        if_end = getattr(node, 'end_lineno', if_start)
                        
                        if any(line in analysis[2] for line in range(if_start, if_end + 1)):
                            uncovered_branches.append(f"if_statement_{if_start}")
                            
            except Exception:
                pass
            
            return {
                'total_statements': total_statements,
                'covered_statements': covered_statements,
                'missed_statements': missed_statements,
                'coverage_percentage': coverage_percentage,
                'uncovered_functions': uncovered_functions,
                'uncovered_branches': uncovered_branches
            }
            
        except Exception:
            return None
    
    def _calculate_complexity_score(self, source_files: List[str]) -> float:
        """Calculate cyclomatic complexity score"""
        total_complexity = 0
        file_count = 0
        
        for source_file in source_files:
            try:
                with open(source_file, 'r') as f:
                    source_code = f.read()
                
                tree = ast.parse(source_code)
                complexity = self._calculate_function_complexity(tree)
                total_complexity += complexity
                file_count += 1
                
            except Exception:
                continue
        
        return total_complexity / file_count if file_count > 0 else 0
    
    def _calculate_function_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity for AST"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity


class TestGenerationManager:
    """Main test generation manager"""
    
    def __init__(self,
                 project_root: str = ".",
                 output_dir: str = "generated_tests",
                 enable_ai: bool = True,
                 enable_templates: bool = True,
                 model_name: str = "gpt-4"):
        """
        Initialize test generation manager
        
        Args:
            project_root: Root directory of the project
            output_dir: Directory to store generated tests
            enable_ai: Whether to enable AI-powered test generation
            enable_templates: Whether to enable template-based test generation
            model_name: Name of the LLM model to use
        """
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.console = Console()
        
        # Initialize generators
        self.generators = []
        if enable_ai:
            self.generators.append(AITestGenerator(model_name))
        if enable_templates:
            self.generators.append(TemplateBasedTestGenerator())
        
        # Initialize coverage analyzer
        self.coverage_analyzer = CoverageAnalyzer(project_root)
        
        # Statistics
        self.stats = {
            'total_tests_generated': 0,
            'tests_by_type': defaultdict(int),
            'tests_by_framework': defaultdict(int),
            'average_confidence': 0.0,
            'execution_time': 0.0
        }
    
    def generate_tests_for_file(self,
                               file_path: str,
                               test_types: List[TestType] = None,
                               max_tests_per_type: int = 5) -> List[GeneratedTest]:
        """Generate tests for a specific file"""
        if test_types is None:
            test_types = [TestType.UNIT, TestType.INTEGRATION]
        
        try:
            # Read source code
            with open(file_path, 'r') as f:
                source_code = f.read()
            
            all_tests = []
            
            # Generate tests for each type
            for test_type in test_types:
                for generator in self.generators:
                    tests = generator.generate_tests(source_code, file_path, test_type)
                    
                    # Limit number of tests per type
                    tests = tests[:max_tests_per_type]
                    all_tests.extend(tests)
            
            # Update statistics
            self.stats['total_tests_generated'] += len(all_tests)
            for test in all_tests:
                self.stats['tests_by_type'][test.test_type.value] += 1
                self.stats['tests_by_framework'][test.framework.value] += 1
            
            return all_tests
            
        except Exception as e:
            self.console.print(f"[red]Failed to generate tests for {file_path}: {e}[/red]")
            return []
    
    def generate_tests_for_project(self,
                                  source_files: List[str] = None,
                                  test_types: List[TestType] = None,
                                  max_workers: int = 4) -> Dict[str, List[GeneratedTest]]:
        """Generate tests for entire project"""
        if source_files is None:
            source_files = self._discover_source_files()
        
        if test_types is None:
            test_types = [TestType.UNIT, TestType.INTEGRATION]
        
        start_time = time.time()
        results = {}
        
        # Use progress bar
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Generating tests...", total=len(source_files))
            
            # Process files in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_file = {
                    executor.submit(self.generate_tests_for_file, file_path, test_types): file_path
                    for file_path in source_files
                }
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        tests = future.result()
                        if tests:
                            results[file_path] = tests
                    except Exception as e:
                        self.console.print(f"[red]Error processing {file_path}: {e}[/red]")
                    
                    progress.update(task, advance=1)
        
        # Update execution time
        self.stats['execution_time'] = time.time() - start_time
        
        # Calculate average confidence
        all_tests = [test for tests in results.values() for test in tests]
        if all_tests:
            self.stats['average_confidence'] = sum(test.confidence_score for test in all_tests) / len(all_tests)
        
        return results
    
    def save_generated_tests(self, results: Dict[str, List[GeneratedTest]]) -> bool:
        """Save generated tests to files"""
        try:
            for file_path, tests in results.items():
                if not tests:
                    continue
                
                # Create test file path
                source_name = Path(file_path).stem
                test_file_path = self.output_dir / f"test_{source_name}.py"
                
                # Generate test file content
                test_content = self._generate_test_file_content(tests)
                
                # Save test file
                with open(test_file_path, 'w') as f:
                    f.write(test_content)
                
                self.console.print(f"[green]Saved {len(tests)} tests to {test_file_path}[/green]")
            
            return True
            
        except Exception as e:
            self.console.print(f"[red]Failed to save generated tests: {e}[/red]")
            return False
    
    def analyze_coverage(self, test_files: List[str] = None, source_files: List[str] = None) -> CoverageAnalysis:
        """Analyze code coverage"""
        if test_files is None:
            test_files = list(self.output_dir.glob("test_*.py"))
        
        if source_files is None:
            source_files = self._discover_source_files()
        
        return self.coverage_analyzer.analyze_coverage(test_files, source_files)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return dict(self.stats)
    
    def print_statistics(self):
        """Print generation statistics"""
        table = Table(title="Test Generation Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Tests Generated", str(self.stats['total_tests_generated']))
        table.add_row("Execution Time", f"{self.stats['execution_time']:.2f}s")
        table.add_row("Average Confidence", f"{self.stats['average_confidence']:.2%}")
        
        for test_type, count in self.stats['tests_by_type'].items():
            table.add_row(f"  {test_type.title()} Tests", str(count))
        
        for framework, count in self.stats['tests_by_framework'].items():
            table.add_row(f"  {framework.title()} Framework", str(count))
        
        self.console.print(table)
    
    def _discover_source_files(self) -> List[str]:
        """Discover Python source files in project"""
        source_files = []
        
        for pattern in ["**/*.py"]:
            for file_path in self.project_root.rglob(pattern):
                # Skip test files and generated files
                if not any(skip in str(file_path) for skip in ["test_", "generated_tests", "__pycache__"]):
                    source_files.append(str(file_path))
        
        return source_files
    
    def _generate_test_file_content(self, tests: List[GeneratedTest]) -> str:
        """Generate content for test file"""
        content = [
            "# Auto-generated tests",
            "import pytest",
            "import sys",
            "import os",
            "",
            "# Add project root to path",
            "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
            ""
        ]
        
        # Group tests by type
        tests_by_type = defaultdict(list)
        for test in tests:
            tests_by_type[test.test_type].append(test)
        
        # Add tests for each type
        for test_type, type_tests in tests_by_type.items():
            content.append(f"# {test_type.value.title()} Tests")
            content.append("")
            
            for test in type_tests:
                content.append(test.code)
                content.append("")
        
        return "\n".join(content)


def main():
    """Main function for test generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate tests for Python projects")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output-dir", default="generated_tests", help="Output directory for generated tests")
    parser.add_argument("--test-types", nargs="+", default=["unit", "integration"], help="Test types to generate")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of worker threads")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI-powered test generation")
    parser.add_argument("--no-templates", action="store_true", help="Disable template-based test generation")
    parser.add_argument("--model", default="gpt-4", help="LLM model to use")
    parser.add_argument("--coverage", action="store_true", help="Run coverage analysis")
    
    args = parser.parse_args()
    
    # Convert test types to enum
    test_types = []
    for test_type in args.test_types:
        try:
            test_types.append(TestType(test_type))
        except ValueError:
            print(f"Unknown test type: {test_type}")
            sys.exit(1)
    
    # Initialize test generation manager
    manager = TestGenerationManager(
        project_root=args.project_root,
        output_dir=args.output_dir,
        enable_ai=not args.no_ai,
        enable_templates=not args.no_templates,
        model_name=args.model
    )
    
    console = Console()
    console.print("[bold blue]Starting test generation...[/bold blue]")
    
    # Generate tests
    results = manager.generate_tests_for_project(test_types=test_types, max_workers=args.max_workers)
    
    # Save tests
    if results:
        success = manager.save_generated_tests(results)
        if success:
            console.print("[bold green]Test generation completed successfully![/bold green]")
        else:
            console.print("[bold red]Failed to save generated tests[/bold red]")
    else:
        console.print("[yellow]No tests were generated[/yellow]")
    
    # Print statistics
    manager.print_statistics()
    
    # Run coverage analysis if requested
    if args.coverage:
        console.print("[bold blue]Running coverage analysis...[/bold blue]")
        coverage_analysis = manager.analyze_coverage()
        
        coverage_table = Table(title="Coverage Analysis")
        coverage_table.add_column("Metric", style="cyan")
        coverage_table.add_column("Value", style="magenta")
        
        coverage_table.add_row("Total Lines", str(coverage_analysis.total_lines))
        coverage_table.add_row("Covered Lines", str(coverage_analysis.covered_lines))
        coverage_table.add_row("Missed Lines", str(coverage_analysis.missed_lines))
        coverage_table.add_row("Coverage Percentage", f"{coverage_analysis.coverage_percentage:.1f}%")
        coverage_table.add_row("Complexity Score", f"{coverage_analysis.complexity_score:.2f}")
        
        console.print(coverage_table)


if __name__ == "__main__":
    main()
