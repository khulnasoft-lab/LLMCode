#!/usr/bin/env python3
"""
Advanced Coverage Analysis Tool

This module provides comprehensive code coverage analysis with advanced features
including branch coverage, mutation testing, and coverage visualization.
"""

import ast
import json
import os
import sys
import time
import subprocess
import threading
import multiprocessing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from collections import defaultdict, Counter
import concurrent.futures

import coverage
import pytest
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.tree import Tree
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


@dataclass
class CoverageMetrics:
    """Detailed coverage metrics"""
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    statement_coverage: float
    complexity_score: float
    mutation_score: float
    risk_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class FileCoverage:
    """Coverage information for a single file"""
    file_path: str
    total_lines: int
    covered_lines: int
    missed_lines: List[int]
    total_branches: int
    covered_branches: int
    missed_branches: List[Tuple[int, str]]
    total_functions: int
    covered_functions: int
    missed_functions: List[str]
    complexity: int
    risk_level: str
    
    def get_line_coverage_percentage(self) -> float:
        """Get line coverage percentage"""
        return (self.covered_lines / self.total_lines * 100) if self.total_lines > 0 else 0
    
    def get_branch_coverage_percentage(self) -> float:
        """Get branch coverage percentage"""
        return (self.covered_branches / self.total_branches * 100) if self.total_branches > 0 else 0
    
    def get_function_coverage_percentage(self) -> float:
        """Get function coverage percentage"""
        return (self.covered_functions / self.total_functions * 100) if self.total_functions > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)


@dataclass
class CoverageReport:
    """Comprehensive coverage report"""
    overall_metrics: CoverageMetrics
    file_coverage: Dict[str, FileCoverage]
    uncovered_files: List[str]
    high_risk_files: List[str]
    recommendations: List[str]
    timestamp: str
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'overall_metrics': self.overall_metrics.to_dict(),
            'file_coverage': {k: v.to_dict() for k, v in self.file_coverage.items()},
            'uncovered_files': self.uncovered_files,
            'high_risk_files': self.high_risk_files,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp,
            'execution_time': self.execution_time
        }


class MutationTester:
    """Mutation testing for coverage analysis"""
    
    def __init__(self, project_root: str = "."):
        """
        Initialize mutation tester
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = Path(project_root)
        self.console = Console()
        self.mutation_operators = [
            self._mutate_arithmetic_operators,
            self._mutate_logical_operators,
            self._mutate_comparison_operators,
            self._mutate_constants,
            self._mutate_return_values
        ]
    
    def run_mutation_testing(self, test_files: List[str], source_files: List[str]) -> float:
        """
        Run mutation testing on source files
        
        Args:
            test_files: List of test files
            source_files: List of source files to mutate
            
        Returns:
            Mutation score (percentage of mutations caught)
        """
        total_mutations = 0
        killed_mutations = 0
        
        self.console.print("[cyan]Running mutation testing...[/cyan]")
        
        for source_file in source_files:
            try:
                file_mutations, file_killed = self._test_file_mutations(
                    source_file, test_files
                )
                total_mutations += file_mutations
                killed_mutations += file_killed
                
            except Exception as e:
                self.console.print(f"[yellow]Failed to test mutations for {source_file}: {e}[/yellow]")
        
        mutation_score = (killed_mutations / total_mutations * 100) if total_mutations > 0 else 0
        
        self.console.print(f"[green]Mutation testing completed: {mutation_score:.1f}% ({killed_mutations}/{total_mutations})[/green]")
        
        return mutation_score
    
    def _test_file_mutations(self, source_file: str, test_files: List[str]) -> Tuple[int, int]:
        """Test mutations for a single file"""
        total_mutations = 0
        killed_mutations = 0
        
        try:
            # Read original file
            with open(source_file, 'r') as f:
                original_content = f.read()
            
            # Parse AST
            tree = ast.parse(original_content)
            
            # Generate mutations
            mutations = self._generate_mutations(tree, source_file)
            
            # Test each mutation
            for mutation in mutations:
                total_mutations += 1
                
                # Apply mutation
                mutated_content = self._apply_mutation(original_content, mutation)
                
                # Test if mutation is caught
                if self._is_mutation_killed(source_file, mutated_content, test_files):
                    killed_mutations += 1
                    
        except Exception as e:
            self.console.print(f"[yellow]Error in mutation testing for {source_file}: {e}[/yellow]")
        
        return total_mutations, killed_mutations
    
    def _generate_mutations(self, tree: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        """Generate mutations for AST"""
        mutations = []
        
        for node in ast.walk(tree):
            for operator in self.mutation_operators:
                try:
                    node_mutations = operator(node, file_path)
                    mutations.extend(node_mutations)
                except Exception:
                    continue
        
        return mutations
    
    def _mutate_arithmetic_operators(self, node: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        """Mutate arithmetic operators"""
        mutations = []
        
        if isinstance(node, ast.BinOp):
            operator_map = {
                ast.Add: ast.Sub,
                ast.Sub: ast.Add,
                ast.Mult: ast.Div,
                ast.Div: ast.Mult,
                ast.FloorDiv: ast.Mult,
                ast.Mod: ast.Mult
            }
            
            if type(node.op) in operator_map:
                mutations.append({
                    'type': 'arithmetic_operator',
                    'line': node.lineno,
                    'col_offset': node.col_offset,
                    'original': type(node.op).__name__,
                    'mutated': operator_map[type(node.op)].__name__
                })
        
        return mutations
    
    def _mutate_logical_operators(self, node: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        """Mutate logical operators"""
        mutations = []
        
        if isinstance(node, ast.BoolOp):
            operator_map = {
                ast.And: ast.Or,
                ast.Or: ast.And
            }
            
            if type(node.op) in operator_map:
                mutations.append({
                    'type': 'logical_operator',
                    'line': node.lineno,
                    'col_offset': node.col_offset,
                    'original': type(node.op).__name__,
                    'mutated': operator_map[type(node.op)].__name__
                })
        
        return mutations
    
    def _mutate_comparison_operators(self, node: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        """Mutate comparison operators"""
        mutations = []
        
        if isinstance(node, ast.Compare):
            operator_map = {
                ast.Eq: ast.NotEq,
                ast.NotEq: ast.Eq,
                ast.Lt: ast.Gt,
                ast.Gt: ast.Lt,
                ast.LtE: ast.GtE,
                ast.GtE: ast.LtE
            }
            
            for op in node.ops:
                if type(op) in operator_map:
                    mutations.append({
                        'type': 'comparison_operator',
                        'line': node.lineno,
                        'col_offset': node.col_offset,
                        'original': type(op).__name__,
                        'mutated': operator_map[type(op)].__name__
                    })
        
        return mutations
    
    def _mutate_constants(self, node: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        """Mutate constants"""
        mutations = []
        
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                mutations.append({
                    'type': 'constant',
                    'line': node.lineno,
                    'col_offset': node.col_offset,
                    'original': node.value,
                    'mutated': node.value + 1
                })
            elif isinstance(node.value, bool):
                mutations.append({
                    'type': 'constant',
                    'line': node.lineno,
                    'col_offset': node.col_offset,
                    'original': node.value,
                    'mutated': not node.value
                })
        
        return mutations
    
    def _mutate_return_values(self, node: ast.AST, file_path: str) -> List[Dict[str, Any]]:
        """Mutate return values"""
        mutations = []
        
        if isinstance(node, ast.Return):
            if node.value is not None:
                mutations.append({
                    'type': 'return_value',
                    'line': node.lineno,
                    'col_offset': node.col_offset,
                    'original': 'return',
                    'mutated': 'return None'
                })
        
        return mutations
    
    def _apply_mutation(self, content: str, mutation: Dict[str, Any]) -> str:
        """Apply mutation to content"""
        lines = content.split('\n')
        line_num = mutation['line'] - 1  # Convert to 0-based indexing
        
        if line_num < len(lines):
            line = lines[line_num]
            
            # Simple mutation application (can be enhanced)
            if mutation['type'] == 'arithmetic_operator':
                line = line.replace(mutation['original'], mutation['mutated'])
            elif mutation['type'] == 'logical_operator':
                line = line.replace(mutation['original'], mutation['mutated'])
            elif mutation['type'] == 'comparison_operator':
                line = line.replace(mutation['original'], mutation['mutated'])
            elif mutation['type'] == 'constant':
                line = line.replace(str(mutation['original']), str(mutation['mutated']))
            elif mutation['type'] == 'return_value':
                line = line.replace('return', 'return None')
            
            lines[line_num] = line
        
        return '\n'.join(lines)
    
    def _is_mutation_killed(self, source_file: str, mutated_content: str, test_files: List[str]) -> bool:
        """Test if a mutation is killed by tests"""
        try:
            # Write mutated file
            with open(source_file, 'w') as f:
                f.write(mutated_content)
            
            # Run tests
            result = subprocess.run(
                ['python', '-m', 'pytest'] + test_files + ['--tb=short'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Restore original file
            with open(source_file, 'r') as f:
                current_content = f.read()
            
            if current_content != mutated_content:
                # File was modified, restore it
                with open(source_file, 'w') as f:
                    f.write(current_content)
            
            # Mutation is killed if tests fail
            return result.returncode != 0
            
        except Exception:
            return False


class CoverageVisualizer:
    """Coverage visualization tools"""
    
    def __init__(self, output_dir: str = "coverage_reports"):
        """
        Initialize coverage visualizer
        
        Args:
            output_dir: Directory to save visualization reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.console = Console()
    
    def generate_coverage_report(self, coverage_report: CoverageReport) -> str:
        """
        Generate comprehensive coverage report with visualizations
        
        Args:
            coverage_report: Coverage report to visualize
            
        Returns:
            Path to generated report
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"coverage_report_{timestamp}.html"
        
        # Generate visualizations
        self._generate_coverage_charts(coverage_report)
        self._generate_file_coverage_heatmap(coverage_report)
        self._generate_risk_analysis_chart(coverage_report)
        
        # Generate HTML report
        html_content = self._generate_html_report(coverage_report)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        self.console.print(f"[green]Coverage report generated: {report_path}[/green]")
        return str(report_path)
    
    def _generate_coverage_charts(self, coverage_report: CoverageReport):
        """Generate coverage charts"""
        metrics = coverage_report.overall_metrics
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Code Coverage Analysis', fontsize=16, fontweight='bold')
        
        # Coverage types pie chart
        coverage_types = ['Line Coverage', 'Branch Coverage', 'Function Coverage']
        coverage_values = [
            metrics.line_coverage,
            metrics.branch_coverage,
            metrics.function_coverage
        ]
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        axes[0, 0].pie(coverage_values, labels=coverage_types, autopct='%1.1f%%', colors=colors)
        axes[0, 0].set_title('Coverage Distribution')
        
        # Metrics bar chart
        metric_names = ['Line', 'Branch', 'Function', 'Statement']
        metric_values = [
            metrics.line_coverage,
            metrics.branch_coverage,
            metrics.function_coverage,
            metrics.statement_coverage
        ]
        
        bars = axes[0, 1].bar(metric_names, metric_values, color=colors)
        axes[0, 1].set_title('Coverage Metrics')
        axes[0, 1].set_ylabel('Coverage (%)')
        axes[0, 1].set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.1f}%', ha='center', va='bottom')
        
        # Quality metrics radar chart
        quality_metrics = ['Coverage', 'Complexity', 'Mutation', 'Risk']
        quality_values = [
            metrics.line_coverage,
            max(0, 100 - metrics.complexity_score * 10),  # Invert complexity
            metrics.mutation_score,
            max(0, 100 - metrics.risk_score * 10)  # Invert risk
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(quality_metrics), endpoint=False).tolist()
        quality_values += quality_values[:1]  # Complete the circle
        angles += angles[:1]
        
        axes[1, 0].plot(angles, quality_values, 'o-', linewidth=2, color='#3498db')
        axes[1, 0].fill(angles, quality_values, alpha=0.25, color='#3498db')
        axes[1, 0].set_xticks(angles[:-1])
        axes[1, 0].set_xticklabels(quality_metrics)
        axes[1, 0].set_title('Quality Metrics')
        axes[1, 0].set_ylim(0, 100)
        
        # Trend analysis (if historical data available)
        axes[1, 1].text(0.5, 0.5, 'Trend Analysis\n(Coming Soon)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        axes[1, 1].set_title('Coverage Trends')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"coverage_charts_{time.strftime('%Y%m%d_%H%M%S')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_file_coverage_heatmap(self, coverage_report: CoverageReport):
        """Generate file coverage heatmap"""
        # Prepare data
        file_data = []
        file_names = []
        
        for file_path, file_cov in coverage_report.file_coverage.items():
            file_names.append(Path(file_path).name)
            file_data.append([
                file_cov.get_line_coverage_percentage(),
                file_cov.get_branch_coverage_percentage(),
                file_cov.get_function_coverage_percentage(),
                file_cov.complexity
            ])
        
        if not file_data:
            return
        
        # Create DataFrame
        df = pd.DataFrame(file_data, 
                         index=file_names,
                         columns=['Line Coverage', 'Branch Coverage', 'Function Coverage', 'Complexity'])
        
        # Create heatmap
        plt.figure(figsize=(12, max(8, len(file_names) * 0.5)))
        
        # Normalize complexity for better visualization
        df_normalized = df.copy()
        df_normalized['Complexity'] = df['Complexity'] / df['Complexity'].max() * 100
        
        # Create heatmap
        sns.heatmap(df_normalized, annot=True, fmt='.1f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Percentage'})
        
        plt.title('File Coverage Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Metrics')
        plt.ylabel('Files')
        plt.tight_layout()
        
        plt.savefig(self.output_dir / f"coverage_heatmap_{time.strftime('%Y%m%d_%H%M%S')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_risk_analysis_chart(self, coverage_report: CoverageReport):
        """Generate risk analysis chart"""
        # Group files by risk level
        risk_counts = defaultdict(int)
        for file_cov in coverage_report.file_coverage.values():
            risk_counts[file_cov.risk_level] += 1
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        risk_labels = list(risk_counts.keys())
        risk_values = list(risk_counts.values())
        
        colors = {'High': '#e74c3c', 'Medium': '#f39c12', 'Low': '#2ecc71'}
        pie_colors = [colors.get(label, '#95a5a6') for label in risk_labels]
        
        wedges, texts, autotexts = ax.pie(risk_values, labels=risk_labels, autopct='%1.1f%%',
                                         colors=pie_colors, startangle=90)
        
        ax.set_title('Risk Analysis by File', fontsize=14, fontweight='bold')
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"risk_analysis_{time.strftime('%Y%m%d_%H%M%S')}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, coverage_report: CoverageReport) -> str:
        """Generate HTML report"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Coverage Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    border-left: 4px solid #007bff;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #007bff;
                }}
                .metric-label {{
                    font-size: 0.9em;
                    color: #666;
                    margin-top: 5px;
                }}
                .file-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                .file-table th, .file-table td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                .file-table th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                }}
                .risk-high {{ background-color: #ffebee; }}
                .risk-medium {{ background-color: #fff3e0; }}
                .risk-low {{ background-color: #e8f5e8; }}
                .recommendations {{
                    background-color: #e3f2fd;
                    padding: 15px;
                    border-radius: 8px;
                    margin: 20px 0;
                }}
                .chart {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .chart img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Code Coverage Analysis Report</h1>
                <p>Generated on: {coverage_report.timestamp}</p>
                <p>Analysis completed in: {coverage_report.execution_time:.2f}s</p>
                
                <h2>Overall Metrics</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{coverage_report.overall_metrics.line_coverage:.1f}%</div>
                        <div class="metric-label">Line Coverage</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{coverage_report.overall_metrics.branch_coverage:.1f}%</div>
                        <div class="metric-label">Branch Coverage</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{coverage_report.overall_metrics.function_coverage:.1f}%</div>
                        <div class="metric-label">Function Coverage</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{coverage_report.overall_metrics.mutation_score:.1f}%</div>
                        <div class="metric-label">Mutation Score</div>
                    </div>
                </div>
                
                <h2>Visualizations</h2>
                <div class="chart">
                    <img src="coverage_charts_{timestamp}.png" alt="Coverage Charts">
                </div>
                <div class="chart">
                    <img src="coverage_heatmap_{timestamp}.png" alt="Coverage Heatmap">
                </div>
                <div class="chart">
                    <img src="risk_analysis_{timestamp}.png" alt="Risk Analysis">
                </div>
                
                <h2>File Coverage Details</h2>
                <table class="file-table">
                    <thead>
                        <tr>
                            <th>File</th>
                            <th>Line Coverage</th>
                            <th>Branch Coverage</th>
                            <th>Function Coverage</th>
                            <th>Complexity</th>
                            <th>Risk Level</th>
                        </tr>
                    </thead>
                    <tbody>
        """
        
        for file_path, file_cov in coverage_report.file_coverage.items():
            risk_class = f"risk-{file_cov.risk_level.lower()}"
            html_content += f"""
                        <tr class="{risk_class}">
                            <td>{Path(file_path).name}</td>
                            <td>{file_cov.get_line_coverage_percentage():.1f}%</td>
                            <td>{file_cov.get_branch_coverage_percentage():.1f}%</td>
                            <td>{file_cov.get_function_coverage_percentage():.1f}%</td>
                            <td>{file_cov.complexity}</td>
                            <td>{file_cov.risk_level}</td>
                        </tr>
            """
        
        html_content += """
                    </tbody>
                </table>
                
                <h2>Recommendations</h2>
                <div class="recommendations">
                    <ul>
        """
        
        for recommendation in coverage_report.recommendations:
            html_content += f"                        <li>{recommendation}</li>\n"
        
        html_content += """
                    </ul>
                </div>
                
                <h2>High Risk Files</h2>
                <ul>
        """
        
        for file_path in coverage_report.high_risk_files:
            html_content += f"                    <li>{Path(file_path).name}</li>\n"
        
        html_content += """
                </ul>
                
                <h2>Uncovered Files</h2>
                <ul>
        """
        
        for file_path in coverage_report.uncovered_files:
            html_content += f"                    <li>{Path(file_path).name}</li>\n"
        
        html_content += """
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html_content


class AdvancedCoverageAnalyzer:
    """Advanced coverage analyzer with comprehensive analysis capabilities"""
    
    def __init__(self, project_root: str = ".", output_dir: str = "coverage_reports"):
        """
        Initialize advanced coverage analyzer
        
        Args:
            project_root: Root directory of the project
            output_dir: Directory to save reports
        """
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.console = Console()
        self.cov = coverage.Coverage(branch=True)
        self.mutation_tester = MutationTester(project_root)
        self.visualizer = CoverageVisualizer(output_dir)
    
    def analyze_coverage(self, test_files: List[str], source_files: List[str]) -> CoverageReport:
        """
        Perform comprehensive coverage analysis
        
        Args:
            test_files: List of test files
            source_files: List of source files to analyze
            
        Returns:
            Comprehensive coverage report
        """
        start_time = time.time()
        
        self.console.print("[bold blue]Starting comprehensive coverage analysis...[/bold blue]")
        
        # Run basic coverage analysis
        file_coverage = self._analyze_file_coverage(test_files, source_files)
        
        # Run mutation testing
        mutation_score = self.mutation_tester.run_mutation_testing(test_files, source_files)
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(file_coverage, mutation_score)
        
        # Identify high-risk files
        high_risk_files = self._identify_high_risk_files(file_coverage)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(file_coverage, overall_metrics)
        
        # Create coverage report
        coverage_report = CoverageReport(
            overall_metrics=overall_metrics,
            file_coverage=file_coverage,
            uncovered_files=[f for f, cov in file_coverage.items() if cov.get_line_coverage_percentage() == 0],
            high_risk_files=high_risk_files,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat(),
            execution_time=time.time() - start_time
        )
        
        # Generate visualizations
        self.visualizer.generate_coverage_report(coverage_report)
        
        return coverage_report
    
    def _analyze_file_coverage(self, test_files: List[str], source_files: List[str]) -> Dict[str, FileCoverage]:
        """Analyze coverage for each file"""
        self.console.print("[cyan]Analyzing file coverage...[/cyan]")
        
        # Start coverage collection
        self.cov.start()
        
        # Run tests
        self._run_tests(test_files)
        
        # Stop coverage collection
        self.cov.stop()
        self.cov.save()
        
        file_coverage = {}
        
        for source_file in source_files:
            try:
                file_cov = self._analyze_single_file_coverage(source_file)
                if file_cov:
                    file_coverage[source_file] = file_cov
            except Exception as e:
                self.console.print(f"[yellow]Failed to analyze coverage for {source_file}: {e}[/yellow]")
        
        return file_coverage
    
    def _run_tests(self, test_files: List[str]):
        """Run tests with coverage collection"""
        try:
            cmd = ['python', '-m', 'pytest'] + test_files + ['--cov=.']
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode != 0:
                self.console.print(f"[yellow]Tests failed: {result.stderr}[/yellow]")
                
        except Exception as e:
            self.console.print(f"[red]Failed to run tests: {e}[/red]")
    
    def _analyze_single_file_coverage(self, source_file: str) -> Optional[FileCoverage]:
        """Analyze coverage for a single file"""
        try:
            # Get coverage data
            analysis = self.cov.analysis2(source_file)
            if not analysis:
                return None
            
            # Parse source code for detailed analysis
            with open(source_file, 'r') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            # Extract coverage information
            executable_lines = set(analysis[1])  # Executable lines
            missing_lines = set(analysis[2])     # Missing lines
            covered_lines = executable_lines - missing_lines
            
            # Analyze functions
            functions = self._analyze_functions(tree, covered_lines, missing_lines)
            
            # Analyze branches
            branches = self._analyze_branches(tree, covered_lines, missing_lines)
            
            # Calculate complexity
            complexity = self._calculate_complexity(tree)
            
            # Determine risk level
            risk_level = self._determine_risk_level(
                len(covered_lines) / len(executable_lines) if executable_lines else 0,
                complexity,
                len(functions['missed'])
            )
            
            return FileCoverage(
                file_path=source_file,
                total_lines=len(executable_lines),
                covered_lines=len(covered_lines),
                missed_lines=sorted(missing_lines),
                total_branches=branches['total'],
                covered_branches=branches['covered'],
                missed_branches=branches['missed'],
                total_functions=functions['total'],
                covered_functions=functions['covered'],
                missed_functions=functions['missed'],
                complexity=complexity,
                risk_level=risk_level
            )
            
        except Exception as e:
            self.console.print(f"[yellow]Error analyzing {source_file}: {e}[/yellow]")
            return None
    
    def _analyze_functions(self, tree: ast.AST, covered_lines: Set[int], missing_lines: Set[int]) -> Dict[str, Any]:
        """Analyze function coverage"""
        functions = {
            'total': 0,
            'covered': 0,
            'missed': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions['total'] += 1
                
                # Check if function is covered
                func_start = node.lineno
                func_end = getattr(node, 'end_lineno', func_start)
                
                # Check if any line in function is covered
                func_covered = any(line in covered_lines for line in range(func_start, func_end + 1))
                
                if func_covered:
                    functions['covered'] += 1
                else:
                    functions['missed'].append(node.name)
        
        return functions
    
    def _analyze_branches(self, tree: ast.AST, covered_lines: Set[int], missing_lines: Set[int]) -> Dict[str, Any]:
        """Analyze branch coverage"""
        branches = {
            'total': 0,
            'covered': 0,
            'missed': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                branches['total'] += 1
                
                # Check if branch is covered
                branch_start = node.lineno
                branch_end = getattr(node, 'end_lineno', branch_start)
                
                branch_covered = any(line in covered_lines for line in range(branch_start, branch_end + 1))
                
                if branch_covered:
                    branches['covered'] += 1
                else:
                    branches['missed'].append((branch_start, type(node).__name__))
        
        return branches
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
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
    
    def _determine_risk_level(self, coverage: float, complexity: int, missed_functions: int) -> str:
        """Determine risk level based on coverage and complexity"""
        if coverage < 50 or complexity > 20 or missed_functions > 5:
            return "High"
        elif coverage < 80 or complexity > 10 or missed_functions > 2:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_overall_metrics(self, file_coverage: Dict[str, FileCoverage], mutation_score: float) -> CoverageMetrics:
        """Calculate overall coverage metrics"""
        if not file_coverage:
            return CoverageMetrics(0, 0, 0, 0, 0, 0, 0)
        
        total_lines = sum(f.total_lines for f in file_coverage.values())
        covered_lines = sum(f.covered_lines for f in file_coverage.values())
        total_branches = sum(f.total_branches for f in file_coverage.values())
        covered_branches = sum(f.covered_branches for f in file_coverage.values())
        total_functions = sum(f.total_functions for f in file_coverage.values())
        covered_functions = sum(f.covered_functions for f in file_coverage.values())
        
        line_coverage = (covered_lines / total_lines * 100) if total_lines > 0 else 0
        branch_coverage = (covered_branches / total_branches * 100) if total_branches > 0 else 0
        function_coverage = (covered_functions / total_functions * 100) if total_functions > 0 else 0
        statement_coverage = line_coverage  # Simplified
        
        avg_complexity = sum(f.complexity for f in file_coverage.values()) / len(file_coverage)
        
        # Calculate risk score
        high_risk_files = sum(1 for f in file_coverage.values() if f.risk_level == "High")
        risk_score = (high_risk_files / len(file_coverage) * 100) if file_coverage else 0
        
        return CoverageMetrics(
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            function_coverage=function_coverage,
            statement_coverage=statement_coverage,
            complexity_score=avg_complexity,
            mutation_score=mutation_score,
            risk_score=risk_score
        )
    
    def _identify_high_risk_files(self, file_coverage: Dict[str, FileCoverage]) -> List[str]:
        """Identify high-risk files"""
        return [
            file_path for file_path, cov in file_coverage.items()
            if cov.risk_level == "High"
        ]
    
    def _generate_recommendations(self, file_coverage: Dict[str, FileCoverage], metrics: CoverageMetrics) -> List[str]:
        """Generate recommendations based on coverage analysis"""
        recommendations = []
        
        # Overall coverage recommendations
        if metrics.line_coverage < 80:
            recommendations.append("Overall line coverage is below 80%. Focus on increasing test coverage for critical paths.")
        
        if metrics.branch_coverage < 70:
            recommendations.append("Branch coverage is low. Add tests for conditional branches and edge cases.")
        
        if metrics.function_coverage < 90:
            recommendations.append("Function coverage can be improved. Ensure all functions have corresponding tests.")
        
        # Mutation testing recommendations
        if metrics.mutation_score < 60:
            recommendations.append("Mutation score is low. Improve test quality to catch more code defects.")
        
        # Complexity recommendations
        if metrics.complexity_score > 15:
            recommendations.append("Average cyclomatic complexity is high. Consider refactoring complex functions.")
        
        # Risk-based recommendations
        high_risk_files = [f for f in file_coverage.values() if f.risk_level == "High"]
        if high_risk_files:
            recommendations.append(f"Found {len(high_risk_files)} high-risk files. Prioritize testing for these files.")
        
        # File-specific recommendations
        for file_path, cov in file_coverage.items():
            if cov.get_line_coverage_percentage() == 0:
                recommendations.append(f"File {Path(file_path).name} has no coverage. Add tests for this file.")
            elif cov.get_line_coverage_percentage() < 50:
                recommendations.append(f"File {Path(file_path).name} has low coverage ({cov.get_line_coverage_percentage():.1f}%).")
        
        return recommendations


def main():
    """Main function for coverage analysis"""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Advanced coverage analysis tool")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--test-dir", default="tests", help="Directory containing test files")
    parser.add_argument("--source-dir", default=".", help="Directory containing source files")
    parser.add_argument("--output-dir", default="coverage_reports", help="Output directory for reports")
    parser.add_argument("--mutation-testing", action="store_true", help="Enable mutation testing")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of workers")
    
    args = parser.parse_args()
    
    # Discover test and source files
    test_files = list(Path(args.test_dir).rglob("test_*.py"))
    source_files = list(Path(args.source_dir).rglob("*.py"))
    
    # Filter out test files from source files
    source_files = [str(f) for f in source_files if not any(skip in str(f) for skip in ["test_", "tests", "__pycache__"])]
    test_files = [str(f) for f in test_files]
    
    if not test_files:
        print("No test files found.")
        return
    
    if not source_files:
        print("No source files found.")
        return
    
    console = Console()
    console.print(f"[bold blue]Found {len(test_files)} test files and {len(source_files)} source files[/bold blue]")
    
    # Initialize analyzer
    analyzer = AdvancedCoverageAnalyzer(args.project_root, args.output_dir)
    
    # Run analysis
    try:
        coverage_report = analyzer.analyze_coverage(test_files, source_files)
        
        # Print summary
        console.print("[bold green]Coverage analysis completed![/bold green]")
        
        summary_table = Table(title="Coverage Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="magenta")
        
        summary_table.add_row("Line Coverage", f"{coverage_report.overall_metrics.line_coverage:.1f}%")
        summary_table.add_row("Branch Coverage", f"{coverage_report.overall_metrics.branch_coverage:.1f}%")
        summary_table.add_row("Function Coverage", f"{coverage_report.overall_metrics.function_coverage:.1f}%")
        summary_table.add_row("Mutation Score", f"{coverage_report.overall_metrics.mutation_score:.1f}%")
        summary_table.add_row("Risk Score", f"{coverage_report.overall_metrics.risk_score:.1f}%")
        
        console.print(summary_table)
        
        # Print recommendations
        if coverage_report.recommendations:
            console.print("\n[bold yellow]Recommendations:[/bold yellow]")
            for rec in coverage_report.recommendations:
                console.print(f"  • {rec}")
        
    except Exception as e:
        console.print(f"[red]Coverage analysis failed: {e}[/red]")


if __name__ == "__main__":
    main()
