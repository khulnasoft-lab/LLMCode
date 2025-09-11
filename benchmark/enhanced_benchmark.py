#!/usr/bin/env python3
"""
Enhanced Benchmark Framework with Test Generation

This script extends the existing benchmark framework with AI-powered test generation,
coverage analysis, and comprehensive quality metrics.
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.test_generator import (
    TestGenerationManager,
    TestType,
    CoverageAnalysis,
    GeneratedTest
)
from benchmark import (
    get_model,
    get_coder,
    show_stats,
    summarize_results,
    get_files,
    check_docker,
    run_cmd,
    check_model_availability,
    check_git_status,
    get_env_var,
)

app = typer.Typer()
console = Console()


@dataclass
class BenchmarkResults:
    """Enhanced benchmark results with test generation metrics"""
    original_results: Dict[str, Any]
    test_generation_results: Dict[str, Any]
    coverage_analysis: CoverageAnalysis
    quality_metrics: Dict[str, Any]
    execution_time: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'original_results': self.original_results,
            'test_generation_results': self.test_generation_results,
            'coverage_analysis': asdict(self.coverage_analysis),
            'quality_metrics': self.quality_metrics,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkResults':
        """Create from dictionary"""
        coverage_data = data['coverage_analysis']
        coverage_analysis = CoverageAnalysis(**coverage_data)
        
        return cls(
            original_results=data['original_results'],
            test_generation_results=data['test_generation_results'],
            coverage_analysis=coverage_analysis,
            quality_metrics=data['quality_metrics'],
            execution_time=data['execution_time'],
            timestamp=data['timestamp']
        )


class EnhancedBenchmarkRunner:
    """Enhanced benchmark runner with test generation"""
    
    def __init__(self,
                 model_name: str,
                 edit_format: str,
                 test_types: List[TestType] = None,
                 enable_test_generation: bool = True,
                 enable_coverage_analysis: bool = True,
                 output_dir: str = "enhanced_benchmark_results"):
        """
        Initialize enhanced benchmark runner
        
        Args:
            model_name: Name of the model to use
            edit_format: Edit format for the coder
            test_types: Types of tests to generate
            enable_test_generation: Whether to enable test generation
            enable_coverage_analysis: Whether to enable coverage analysis
            output_dir: Directory to store results
        """
        self.model_name = model_name
        self.edit_format = edit_format
        self.test_types = test_types or [TestType.UNIT, TestType.INTEGRATION]
        self.enable_test_generation = enable_test_generation
        self.enable_coverage_analysis = enable_coverage_analysis
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize test generation manager
        if self.enable_test_generation:
            self.test_manager = TestGenerationManager(
                output_dir=str(self.output_dir / "generated_tests"),
                enable_ai=True,
                enable_templates=True,
                model_name=model_name
            )
        
        self.console = Console()
        
    def run_benchmark(self,
                     dirname: str,
                     exercises: List[str] = None,
                     max_workers: int = 4) -> BenchmarkResults:
        """
        Run enhanced benchmark
        
        Args:
            dirname: Directory containing benchmark exercises
            exercises: List of exercises to run (None for all)
            max_workers: Maximum number of workers
            
        Returns:
            BenchmarkResults object containing all results
        """
        start_time = time.time()
        
        # Get files to process
        if exercises:
            files = []
            for exercise in exercises:
                exercise_files = get_files(dirname, exercise)
                files.extend(exercise_files)
        else:
            files = get_files(dirname)
        
        if not files:
            raise ValueError(f"No files found in {dirname}")
        
        self.console.print(f"[bold blue]Running enhanced benchmark on {len(files)} files...[/bold blue]")
        
        # Run original benchmark
        original_results = self._run_original_benchmark(dirname, files)
        
        # Run test generation
        test_generation_results = {}
        if self.enable_test_generation:
            test_generation_results = self._run_test_generation(files, max_workers)
        
        # Run coverage analysis
        coverage_analysis = CoverageAnalysis(0, 0, 0, 0, [], [], [], 0)
        if self.enable_coverage_analysis and test_generation_results:
            coverage_analysis = self._run_coverage_analysis()
        
        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            original_results,
            test_generation_results,
            coverage_analysis
        )
        
        execution_time = time.time() - start_time
        
        return BenchmarkResults(
            original_results=original_results,
            test_generation_results=test_generation_results,
            coverage_analysis=coverage_analysis,
            quality_metrics=quality_metrics,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _run_original_benchmark(self, dirname: str, files: List[str]) -> Dict[str, Any]:
        """Run original benchmark on files"""
        self.console.print("[cyan]Running original benchmark...[/cyan]")
        
        # Initialize model and coder
        model = get_model(self.model_name)
        coder = get_coder(model, self.edit_format)
        
        results = {}
        
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            console=self.console
        ) as progress:
            
            task = progress.add_task("Processing files...", total=len(files))
            
            for file_path in files:
                try:
                    # Process file with original benchmark logic
                    result = self._process_file_with_coder(coder, file_path, dirname)
                    results[file_path] = result
                except Exception as e:
                    self.console.print(f"[red]Error processing {file_path}: {e}[/red]")
                    results[file_path] = {"error": str(e)}
                
                progress.update(task, advance=1)
        
        return results
    
    def _process_file_with_coder(self, coder, file_path: str, dirname: str) -> Dict[str, Any]:
        """Process a single file with the coder"""
        # This is a simplified version of the original benchmark logic
        # In practice, you would integrate with the existing benchmark.py logic
        
        try:
            # Read the file
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Process with coder (simplified)
            # In the actual implementation, this would involve the full benchmark workflow
            result = {
                "file_path": file_path,
                "content_length": len(content),
                "processed": True,
                "timestamp": time.time()
            }
            
            return result
            
        except Exception as e:
            return {
                "file_path": file_path,
                "error": str(e),
                "processed": False,
                "timestamp": time.time()
            }
    
    def _run_test_generation(self, files: List[str], max_workers: int) -> Dict[str, Any]:
        """Run test generation on files"""
        self.console.print("[cyan]Running test generation...[/cyan]")
        
        # Generate tests for all files
        test_results = self.test_manager.generate_tests_for_project(
            source_files=files,
            test_types=self.test_types,
            max_workers=max_workers
        )
        
        # Save generated tests
        self.test_manager.save_generated_tests(test_results)
        
        # Get statistics
        stats = self.test_manager.get_statistics()
        
        return {
            "test_results": {k: [t.to_dict() for t in v] for k, v in test_results.items()},
            "statistics": stats,
            "files_processed": len(files),
            "files_with_tests": len([f for f, tests in test_results.items() if tests])
        }
    
    def _run_coverage_analysis(self) -> CoverageAnalysis:
        """Run coverage analysis"""
        self.console.print("[cyan]Running coverage analysis...[/cyan]")
        
        # Get test files and source files
        test_files = list(self.test_manager.output_dir.glob("test_*.py"))
        source_files = self.test_manager._discover_source_files()
        
        # Run coverage analysis
        coverage_analysis = self.test_manager.analyze_coverage(
            test_files=[str(f) for f in test_files],
            source_files=source_files
        )
        
        return coverage_analysis
    
    def _calculate_quality_metrics(self,
                                 original_results: Dict[str, Any],
                                 test_generation_results: Dict[str, Any],
                                 coverage_analysis: CoverageAnalysis) -> Dict[str, Any]:
        """Calculate quality metrics"""
        metrics = {
            "test_generation_success_rate": 0.0,
            "average_tests_per_file": 0.0,
            "coverage_score": 0.0,
            "quality_index": 0.0
        }
        
        # Calculate test generation success rate
        if test_generation_results:
            files_with_tests = test_generation_results.get("files_with_tests", 0)
            files_processed = test_generation_results.get("files_processed", 1)
            metrics["test_generation_success_rate"] = files_with_tests / files_processed
            
            # Calculate average tests per file
            total_tests = sum(len(tests) for tests in test_generation_results.get("test_results", {}).values())
            metrics["average_tests_per_file"] = total_tests / files_processed if files_processed > 0 else 0
        
        # Calculate coverage score
        if coverage_analysis.total_lines > 0:
            metrics["coverage_score"] = coverage_analysis.coverage_percentage / 100
        
        # Calculate overall quality index
        metrics["quality_index"] = (
            metrics["test_generation_success_rate"] * 0.4 +
            min(metrics["average_tests_per_file"] / 10, 1.0) * 0.3 +
            metrics["coverage_score"] * 0.3
        )
        
        return metrics
    
    def save_results(self, results: BenchmarkResults, filename: str = None) -> str:
        """Save benchmark results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"enhanced_benchmark_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        
        self.console.print(f"[green]Results saved to {filepath}[/green]")
        return str(filepath)
    
    def print_results(self, results: BenchmarkResults):
        """Print benchmark results"""
        # Print summary table
        summary_table = Table(title="Enhanced Benchmark Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")
        
        summary_table.add_row("Total Files Processed", str(len(results.original_results)))
        summary_table.add_row("Files with Tests", str(results.test_generation_results.get("files_with_tests", 0)))
        summary_table.add_row("Total Tests Generated", str(results.test_generation_results.get("statistics", {}).get("total_tests_generated", 0)))
        summary_table.add_row("Coverage Percentage", f"{results.coverage_analysis.coverage_percentage:.1f}%")
        summary_table.add_row("Quality Index", f"{results.quality_metrics['quality_index']:.2%}")
        summary_table.add_row("Execution Time", f"{results.execution_time:.2f}s")
        
        self.console.print(summary_table)
        
        # Print test generation statistics
        if results.test_generation_results:
            stats = results.test_generation_results.get("statistics", {})
            stats_table = Table(title="Test Generation Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="magenta")
            
            stats_table.add_row("Total Tests Generated", str(stats.get("total_tests_generated", 0)))
            stats_table.add_row("Average Confidence", f"{stats.get('average_confidence', 0):.2%}")
            stats_table.add_row("Generation Time", f"{stats.get('execution_time', 0):.2f}s")
            
            for test_type, count in stats.get("tests_by_type", {}).items():
                stats_table.add_row(f"  {test_type.title()} Tests", str(count))
            
            self.console.print(stats_table)
        
        # Print coverage analysis
        coverage_table = Table(title="Coverage Analysis")
        coverage_table.add_column("Metric", style="cyan")
        coverage_table.add_column("Value", style="magenta")
        
        coverage_table.add_row("Total Lines", str(results.coverage_analysis.total_lines))
        coverage_table.add_row("Covered Lines", str(results.coverage_analysis.covered_lines))
        coverage_table.add_row("Missed Lines", str(results.coverage_analysis.missed_lines))
        coverage_table.add_row("Coverage Percentage", f"{results.coverage_analysis.coverage_percentage:.1f}%")
        coverage_table.add_row("Complexity Score", f"{results.coverage_analysis.complexity_score:.2f}")
        coverage_table.add_row("Uncovered Files", str(len(results.coverage_analysis.uncovered_files)))
        coverage_table.add_row("Uncovered Functions", str(len(results.coverage_analysis.uncovered_functions)))
        
        self.console.print(coverage_table)
        
        # Print quality metrics
        quality_table = Table(title="Quality Metrics")
        quality_table.add_column("Metric", style="cyan")
        quality_table.add_column("Value", style="magenta")
        
        quality_table.add_row("Test Generation Success Rate", f"{results.quality_metrics['test_generation_success_rate']:.2%}")
        quality_table.add_row("Average Tests per File", f"{results.quality_metrics['average_tests_per_file']:.1f}")
        quality_table.add_row("Coverage Score", f"{results.quality_metrics['coverage_score']:.2%}")
        quality_table.add_row("Quality Index", f"{results.quality_metrics['quality_index']:.2%}")
        
        self.console.print(quality_table)


@app.command()
def run_enhanced_benchmark(
    dirname: str = typer.Argument(..., help="Directory containing benchmark exercises"),
    model: str = typer.Option("gpt-4", "--model", "-m", help="Model to use"),
    edit_format: str = typer.Option("diff", "--edit-format", "-e", help="Edit format"),
    test_types: str = typer.Option("unit,integration", "--test-types", "-t", help="Comma-separated test types"),
    max_workers: int = typer.Option(4, "--max-workers", "-w", help="Maximum number of workers"),
    no_test_generation: bool = typer.Option(False, "--no-test-generation", help="Disable test generation"),
    no_coverage_analysis: bool = typer.Option(False, "--no-coverage-analysis", help="Disable coverage analysis"),
    output_dir: str = typer.Option("enhanced_benchmark_results", "--output-dir", "-o", help="Output directory"),
    save_results: bool = typer.Option(True, "--save-results", help="Save results to file"),
    exercises: str = typer.Option(None, "--exercises", "-x", help="Comma-separated list of exercises to run"),
):
    """Run enhanced benchmark with test generation"""
    
    # Parse test types
    test_type_list = []
    for test_type in test_types.split(','):
        try:
            test_type_list.append(TestType(test_type.strip()))
        except ValueError:
            console.print(f"[red]Unknown test type: {test_type}[/red]")
            raise typer.Exit(1)
    
    # Parse exercises
    exercise_list = None
    if exercises:
        exercise_list = [ex.strip() for ex in exercises.split(',')]
    
    # Check prerequisites
    try:
        check_docker()
        check_model_availability(model)
        check_git_status()
    except Exception as e:
        console.print(f"[red]Prerequisite check failed: {e}[/red]")
        raise typer.Exit(1)
    
    # Initialize enhanced benchmark runner
    runner = EnhancedBenchmarkRunner(
        model_name=model,
        edit_format=edit_format,
        test_types=test_type_list,
        enable_test_generation=not no_test_generation,
        enable_coverage_analysis=not no_coverage_analysis,
        output_dir=output_dir
    )
    
    try:
        # Run benchmark
        results = runner.run_benchmark(
            dirname=dirname,
            exercises=exercise_list,
            max_workers=max_workers
        )
        
        # Print results
        runner.print_results(results)
        
        # Save results
        if save_results:
            results_file = runner.save_results(results)
            console.print(f"[green]Results saved to: {results_file}[/green]")
        
    except Exception as e:
        console.print(f"[red]Benchmark failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def compare_results(
    result_files: List[str] = typer.Argument(..., help="Result files to compare"),
    output_format: str = typer.Option("table", "--output-format", "-f", help="Output format (table, json, csv)"),
):
    """Compare multiple benchmark results"""
    
    results = []
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                results.append(BenchmarkResults.from_dict(data))
        except Exception as e:
            console.print(f"[red]Failed to load {result_file}: {e}[/red]")
            continue
    
    if not results:
        console.print("[red]No valid result files found[/red]")
        raise typer.Exit(1)
    
    if output_format == "table":
        # Print comparison table
        comparison_table = Table(title="Benchmark Results Comparison")
        comparison_table.add_column("Metric", style="cyan")
        
        for i, result in enumerate(results):
            comparison_table.add_column(f"Run {i+1}", style="magenta")
        
        # Add rows
        metrics = [
            ("Total Files", lambda r: len(r.original_results)),
            ("Files with Tests", lambda r: r.test_generation_results.get("files_with_tests", 0)),
            ("Total Tests", lambda r: r.test_generation_results.get("statistics", {}).get("total_tests_generated", 0)),
            ("Coverage %", lambda r: f"{r.coverage_analysis.coverage_percentage:.1f}%"),
            ("Quality Index", lambda r: f"{r.quality_metrics['quality_index']:.2%}"),
            ("Execution Time", lambda r: f"{r.execution_time:.2f}s"),
        ]
        
        for metric_name, metric_func in metrics:
            row = [metric_name]
            for result in results:
                row.append(str(metric_func(result)))
            comparison_table.add_row(*row)
        
        console.print(comparison_table)
    
    elif output_format == "json":
        # Output as JSON
        comparison_data = []
        for i, result in enumerate(results):
            comparison_data.append({
                "run": i + 1,
                "timestamp": result.timestamp,
                "total_files": len(result.original_results),
                "files_with_tests": result.test_generation_results.get("files_with_tests", 0),
                "total_tests": result.test_generation_results.get("statistics", {}).get("total_tests_generated", 0),
                "coverage_percentage": result.coverage_analysis.coverage_percentage,
                "quality_index": result.quality_metrics["quality_index"],
                "execution_time": result.execution_time
            })
        
        console.print(json.dumps(comparison_data, indent=2))
    
    elif output_format == "csv":
        # Output as CSV
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(["Run", "Timestamp", "Total Files", "Files with Tests", "Total Tests", 
                        "Coverage %", "Quality Index", "Execution Time"])
        
        # Write data
        for i, result in enumerate(results):
            writer.writerow([
                i + 1,
                result.timestamp,
                len(result.original_results),
                result.test_generation_results.get("files_with_tests", 0),
                result.test_generation_results.get("statistics", {}).get("total_tests_generated", 0),
                f"{result.coverage_analysis.coverage_percentage:.1f}%",
                f"{result.quality_metrics['quality_index']:.2%}",
                f"{result.execution_time:.2f}s"
            ])
        
        console.print(output.getvalue())
    
    else:
        console.print(f"[red]Unknown output format: {output_format}[/red]")
        raise typer.Exit(1)


@app.command()
def list_test_templates():
    """List available test templates"""
    template_dir = Path("test_templates")
    
    if not template_dir.exists():
        console.print("[yellow]No test templates found[/yellow]")
        return
    
    templates_table = Table(title="Available Test Templates")
    templates_table.add_column("Name", style="cyan")
    templates_table.add_column("Description", style="magenta")
    templates_table.add_column("Framework", style="green")
    templates_table.add_column("Language", style="blue")
    templates_table.add_column("Tags", style="yellow")
    
    for template_file in template_dir.glob("*.yaml"):
        try:
            with open(template_file, 'r') as f:
                import yaml
                template_data = yaml.safe_load(f)
                
                templates_table.add_row(
                    template_data.get("name", "Unknown"),
                    template_data.get("description", "No description"),
                    template_data.get("framework", "Unknown"),
                    template_data.get("language", "Unknown"),
                    ", ".join(template_data.get("tags", []))
                )
        except Exception as e:
            console.print(f"[red]Failed to load {template_file}: {e}[/red]")
    
    console.print(templates_table)


if __name__ == "__main__":
    app()
