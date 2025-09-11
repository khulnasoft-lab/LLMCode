"""
AI-Powered Test Generation Benchmarking

This module extends the benchmark framework with AI-powered test generation capabilities,
integrating with the existing benchmark infrastructure.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import concurrent.futures
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn

# Import test generation components
from test_generator import (
    TestType, 
    TestFramework,
    AITestGenerator,
    TemplateBasedTestGenerator,
    TestGenerationManager,
    CoverageAnalyzer,
    GeneratedTest
)

# Import template management
from template_manager import (
    TemplateManager,
    TemplateCategory,
    TemplateLanguage,
    TemplateFramework as TMFramework
)

# Import benchmark utilities
from benchmark import (
    BENCHMARK_DNAME,
    find_latest_benchmark_dir,
    run_unit_tests,
    cleanup_test_output
)

@dataclass
class TestGenerationConfig:
    """Configuration for AI-powered test generation"""
    model_name: str = "gpt-4"
    test_types: List[TestType] = field(default_factory=lambda: [
        TestType.UNIT,
        TestType.INTEGRATION,
        TestType.FUNCTIONAL
    ])
    max_tests_per_type: int = 5
    min_coverage: float = 0.8  # Target minimum coverage
    max_workers: int = 4
    enable_ai: bool = True
    enable_templates: bool = True
    output_dir: str = "generated_tests"
    template_dir: str = "test_templates"

class TestGenerationBenchmark:
    """Benchmark for AI-powered test generation"""
    
    def __init__(self, config: Optional[TestGenerationConfig] = None):
        """Initialize the benchmark with configuration"""
        self.config = config or TestGenerationConfig()
        self.console = Console()
        self.manager = TestGenerationManager(
            project_root=".",
            output_dir=self.config.output_dir,
            enable_ai=self.config.enable_ai,
            enable_templates=self.config.enable_templates,
            model_name=self.config.model_name
        )
        
        # Initialize template manager
        self.template_manager = TemplateManager(self.config.template_dir)
        
        # Results storage
        self.results = {
            'generated_tests': [],
            'coverage': {},
            'performance': {},
            'errors': []
        }
    
    def run_benchmark(self, source_files: List[str] = None):
        """Run the complete test generation benchmark"""
        self.console.print("[bold blue]🚀 Starting AI-Powered Test Generation Benchmark[/]")
        
        # Step 1: Generate tests
        self.console.print("\n[bold]🔧 Generating tests...[/]")
        generation_results = self._generate_tests(source_files)
        
        # Step 2: Analyze coverage
        self.console.print("\n[bold]📊 Analyzing code coverage...[/]")
        coverage_results = self._analyze_coverage(generation_results)
        
        # Step 3: Evaluate performance
        self.console.print("\n[bold]⏱️  Evaluating performance...[/]")
        performance_results = self._evaluate_performance(generation_results, coverage_results)
        
        # Save results
        self._save_results(generation_results, coverage_results, performance_results)
        
        self.console.print("\n[bold green]✅ Benchmark completed successfully![/]")
        return self.results
    
    def _generate_tests(self, source_files: List[str] = None) -> Dict[str, List[GeneratedTest]]:
        """Generate tests using AI and templates"""
        if not source_files:
            source_files = self._discover_source_files()
        
        results = {}
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            "[progress.completed]{task.completed}/{task.total} files",
            "•",
            "[progress.remaining]ETA: {task.remaining}",
            console=self.console
        ) as progress:
            task = progress.add_task("Generating tests...", total=len(source_files))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_file = {
                    executor.submit(
                        self.manager.generate_tests_for_file,
                        file_path=file_path,
                        test_types=self.config.test_types,
                        max_tests_per_type=self.config.max_tests_per_type
                    ): file_path for file_path in source_files
                }
                
                for future in concurrent.futures.as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        file_results = future.result()
                        results[file_path] = file_results
                        self.results['generated_tests'].extend(file_results)
                        
                        # Save intermediate results
                        self._save_test_results(file_path, file_results)
                        
                    except Exception as e:
                        error_msg = f"Error generating tests for {file_path}: {str(e)}"
                        self.console.print(f"[red]❌ {error_msg}[/]")
                        self.results['errors'].append(error_msg)
                    
                    progress.update(task, advance=1)
        
        return results
    
    def _analyze_coverage(self, generation_results: Dict[str, List[GeneratedTest]]) -> Dict[str, Any]:
        """Analyze code coverage for generated tests"""
        test_files = []
        source_files = list(generation_results.keys())
        
        # Collect all generated test files
        for tests in generation_results.values():
            for test in tests:
                if hasattr(test, 'file_path') and test.file_path:
                    test_files.append(test.file_path)
        
        if not test_files:
            self.console.print("[yellow]⚠️ No test files generated for coverage analysis[/]")
            return {}
        
        try:
            # Run coverage analysis
            coverage_analyzer = CoverageAnalyzer(project_root=".")
            coverage_results = coverage_analyzer.analyze_coverage(
                test_files=test_files,
                source_files=source_files
            )
            
            # Store results
            self.results['coverage'] = coverage_results
            return coverage_results
            
        except Exception as e:
            error_msg = f"Error analyzing coverage: {str(e)}"
            self.console.print(f"[red]❌ {error_msg}[/]")
            self.results['errors'].append(error_msg)
            return {}
    
    def _evaluate_performance(self, generation_results: Dict[str, List[GeneratedTest]], 
                            coverage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate performance of test generation"""
        performance = {
            'total_tests_generated': sum(len(tests) for tests in generation_results.values()),
            'total_source_files': len(generation_results),
            'execution_times': {},
            'coverage_metrics': {}
        }
        
        # Calculate average confidence score
        all_tests = [test for tests in generation_results.values() for test in tests]
        if all_tests:
            performance['average_confidence'] = (
                sum(test.confidence_score for test in all_tests) / len(all_tests)
            )
        
        # Add coverage metrics if available
        if coverage_results:
            performance['coverage_metrics'] = {
                'total_lines': coverage_results.get('total_lines', 0),
                'covered_lines': coverage_results.get('covered_lines', 0),
                'coverage_percentage': coverage_results.get('coverage_percentage', 0.0),
                'uncovered_files': len(coverage_results.get('uncovered_files', [])),
                'uncovered_functions': len(coverage_results.get('uncovered_functions', []))
            }
        
        # Store results
        self.results['performance'] = performance
        return performance
    
    def _save_test_results(self, file_path: str, tests: List[GeneratedTest]):
        """Save generated tests to files"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Group tests by test type
        tests_by_type = {}
        for test in tests:
            test_type = test.test_type.value
            if test_type not in tests_by_type:
                tests_by_type[test_type] = []
            tests_by_type[test_type].append(test)
        
        # Save tests by type
        for test_type, type_tests in tests_by_type.items():
            # Create test file name based on source file and test type
            source_path = Path(file_path)
            test_file = output_dir / f"test_{source_path.stem}_{test_type}.py"
            
            # Generate test file content
            test_content = self._generate_test_file_content(type_tests)
            
            # Write to file
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            self.console.print(f"  ✅ Generated {len(type_tests)} {test_type} tests at {test_file}")
    
    def _generate_test_file_content(self, tests: List[GeneratedTest]) -> str:
        """Generate content for a test file"""
        imports = set()
        test_functions = []
        
        for test in tests:
            # Extract imports from test code
            if hasattr(test, 'imports') and test.imports:
                imports.update(test.imports)
            
            # Add test function
            test_functions.append(test.code)
        
        # Generate test file content
        content = """""
# Generated by Llmcode Test Generation Benchmark
# Test file generated on {date}
# Total tests: {test_count}

# Standard library imports
import unittest
import pytest

# Third-party imports
{imports}

# Test cases
{test_functions}

if __name__ == "__main__":
    unittest.main()
""".format(
            date=time.strftime("%Y-%m-%d %H:%M:%S"),
            test_count=len(tests),
            imports='\n'.join(sorted(imports)) if imports else "# No additional imports",
            test_functions='\n\n'.join(test_functions)
        )
        
        return content
    
    def _discover_source_files(self, extensions: List[str] = None) -> List[str]:
        """Discover source files in the project"""
        if extensions is None:
            extensions = ['.py']  # Default to Python files
        
        source_files = []
        for ext in extensions:
            for root, _, files in os.walk('.'):
                for file in files:
                    if file.endswith(ext):
                        source_files.append(os.path.join(root, file))
        
        return source_files
    
    def _save_results(self, generation_results: Dict, coverage_results: Dict, performance_results: Dict):
        """Save benchmark results to a JSON file"""
        results_dir = Path("benchmark_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"test_generation_benchmark_{timestamp}.json"
        
        results = {
            'timestamp': timestamp,
            'config': self.config.__dict__,
            'generation_results': {
                'total_tests': sum(len(tests) for tests in generation_results.values()),
                'test_types': {t.value: 0 for t in self.config.test_types}
            },
            'coverage_results': coverage_results,
            'performance_metrics': performance_results,
            'errors': self.results.get('errors', [])
        }
        
        # Count tests by type
        for tests in generation_results.values():
            for test in tests:
                results['generation_results']['test_types'][test.test_type.value] += 1
        
        # Save to file
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        self.console.print(f"\n[bold]📊 Results saved to: {results_file}[/]")


def main():
    """Main entry point for the benchmark"""
    # Example configuration
    config = TestGenerationConfig(
        model_name="gpt-4",
        test_types=[TestType.UNIT, TestType.INTEGRATION],
        max_tests_per_type=3,
        min_coverage=0.7,
        max_workers=4
    )
    
    # Run benchmark
    benchmark = TestGenerationBenchmark(config)
    results = benchmark.run_benchmark()
    
    # Print summary
    print("\n=== Benchmark Summary ===")
    print(f"Generated {results['performance']['total_tests_generated']} tests")
    print(f"Coverage: {results['performance']['coverage_metrics'].get('coverage_percentage', 0):.1f}%")
    print(f"Average confidence: {results['performance'].get('average_confidence', 0):.2f}")
    
    if results['errors']:
        print("\n=== Errors ===")
        for error in results['errors']:
            print(f"- {error}")


if __name__ == "__main__":
    main()
