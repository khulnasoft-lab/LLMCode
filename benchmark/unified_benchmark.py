#!/usr/bin/env python3
"""
Unified Benchmarking System for Llmcode

This module integrates all components of the enhanced benchmarking framework:
- AI-powered test generation
- Advanced coverage analysis
- Test template management
- Property-based testing
- Contract testing
- Mutation testing
- Load testing

The system provides a comprehensive benchmarking solution with automated
test generation, quality analysis, and performance evaluation.
"""

import os
import sys
import argparse
import logging
import json
import yaml
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import coverage
import psutil
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.text import Text
from jinja2 import Template, Environment, FileSystemLoader

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from test_generator import TestGenerationManager
from coverage_analyzer import AdvancedCoverageAnalyzer, CoverageConfig
from template_manager import TemplateManager, TemplateConfig
from enhanced_benchmark import EnhancedBenchmarkRunner


@dataclass
class UnifiedBenchmarkConfig:
    """Configuration for the unified benchmarking system"""
    # Project configuration
    project_root: str = "."
    source_dirs: List[str] = None
    test_dirs: List[str] = None
    
    # Test generation configuration
    enable_ai_generation: bool = True
    enable_template_generation: bool = True
    ai_model: str = "gpt-3.5-turbo"
    template_categories: List[str] = None
    
    # Coverage analysis configuration
    enable_coverage_analysis: bool = True
    enable_mutation_testing: bool = True
    coverage_threshold: float = 80.0
    mutation_threshold: float = 70.0
    
    # Performance testing configuration
    enable_load_testing: bool = True
    load_test_users: int = 10
    load_test_duration: int = 60
    
    # Output configuration
    output_dir: str = "benchmark_results"
    report_format: str = "html"
    verbose: bool = False
    
    # Parallel processing
    max_workers: int = 4
    enable_parallel: bool = True
    
    def __post_init__(self):
        """Initialize default values"""
        if self.source_dirs is None:
            self.source_dirs = ["src", "lib", "llmcode"]
        if self.test_dirs is None:
            self.test_dirs = ["tests", "test"]
        if self.template_categories is None:
            self.template_categories = ["unit", "integration", "security", "performance"]


class UnifiedBenchmarkRunner:
    """Unified benchmarking system runner"""
    
    def __init__(self, config: UnifiedBenchmarkConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.results = {}
        
        # Initialize components
        self.test_generator = None
        self.coverage_analyzer = None
        self.template_manager = None
        self.benchmark_runner = None
        
        self._initialize_components()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger("unified_benchmark")
        logger.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        os.makedirs(self.config.output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(self.config.output_dir, "unified_benchmark.log")
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_components(self):
        """Initialize all benchmarking components"""
        try:
            # Initialize template manager
            template_config = TemplateConfig(
                template_dir="benchmark/test_templates",
                cache_dir=os.path.join(self.config.output_dir, "template_cache")
            )
            self.template_manager = TemplateManager(template_config)
            self.logger.info("Template manager initialized")
            
            # Initialize test generator
            test_gen_config = TestGenerationConfig(
                project_root=self.config.project_root,
                source_dirs=self.config.source_dirs,
                test_dirs=self.config.test_dirs,
                output_dir=os.path.join(self.config.output_dir, "generated_tests"),
                ai_model=self.config.ai_model,
                max_workers=self.config.max_workers
            )
            self.test_generator = TestGenerationManager(test_gen_config)
            self.logger.info("Test generator initialized")
            
            # Initialize coverage analyzer
            coverage_config = CoverageConfig(
                project_root=self.config.project_root,
                source_dirs=self.config.source_dirs,
                test_dirs=self.config.test_dirs,
                output_dir=os.path.join(self.config.output_dir, "coverage"),
                enable_mutation=self.config.enable_mutation_testing
            )
            self.coverage_analyzer = AdvancedCoverageAnalyzer(coverage_config)
            self.logger.info("Coverage analyzer initialized")
            
            # Initialize benchmark runner
            benchmark_config = BenchmarkConfig(
                project_root=self.config.project_root,
                output_dir=self.config.output_dir,
                max_workers=self.config.max_workers
            )
            self.benchmark_runner = EnhancedBenchmarkRunner(benchmark_config)
            self.logger.info("Benchmark runner initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmarking pipeline"""
        self.logger.info("Starting unified benchmarking pipeline")
        start_time = time.time()
        
        try:
            # Phase 1: Project Analysis
            self.logger.info("Phase 1: Project Analysis")
            project_analysis = self._analyze_project()
            self.results['project_analysis'] = project_analysis
            
            # Phase 2: Test Generation
            self.logger.info("Phase 2: Test Generation")
            test_generation_results = self._run_test_generation()
            self.results['test_generation'] = test_generation_results
            
            # Phase 3: Coverage Analysis
            self.logger.info("Phase 3: Coverage Analysis")
            coverage_results = self._run_coverage_analysis()
            self.results['coverage_analysis'] = coverage_results
            
            # Phase 4: Performance Testing
            self.logger.info("Phase 4: Performance Testing")
            performance_results = self._run_performance_testing()
            self.results['performance_testing'] = performance_results
            
            # Phase 5: Quality Assessment
            self.logger.info("Phase 5: Quality Assessment")
            quality_assessment = self._assess_quality()
            self.results['quality_assessment'] = quality_assessment
            
            # Phase 6: Report Generation
            self.logger.info("Phase 6: Report Generation")
            self._generate_reports()
            
            # Calculate total execution time
            total_time = time.time() - start_time
            self.results['execution_time'] = total_time
            self.logger.info(f"Unified benchmarking completed in {total_time:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            self.logger.error(f"Error during unified benchmarking: {e}")
            raise
    
    def _analyze_project(self) -> Dict[str, Any]:
        """Analyze project structure and characteristics"""
        self.logger.info("Analyzing project structure")
        
        analysis = {
            'project_root': self.config.project_root,
            'source_files': [],
            'test_files': [],
            'languages': set(),
            'frameworks': set(),
            'complexity_metrics': {}
        }
        
        # Analyze source files
        for source_dir in self.config.source_dirs:
            source_path = Path(self.config.project_root) / source_dir
            if source_path.exists():
                for file_path in source_path.rglob("*.py"):
                    analysis['source_files'].append(str(file_path))
                    analysis['languages'].add('python')
        
        # Analyze test files
        for test_dir in self.config.test_dirs:
            test_path = Path(self.config.project_root) / test_dir
            if test_path.exists():
                for file_path in test_path.rglob("*.py"):
                    if file_path.name.startswith("test_") or file_path.name.endswith("_test.py"):
                        analysis['test_files'].append(str(file_path))
        
        # Detect frameworks
        self._detect_frameworks(analysis)
        
        # Calculate complexity metrics
        analysis['complexity_metrics'] = self._calculate_complexity_metrics(analysis['source_files'])
        
        self.logger.info(f"Found {len(analysis['source_files'])} source files and {len(analysis['test_files'])} test files")
        return analysis
    
    def _detect_frameworks(self, analysis: Dict[str, Any]):
        """Detect frameworks used in the project"""
        # Check for common framework indicators
        framework_indicators = {
            'django': ['django', 'DJANGO_SETTINGS_MODULE'],
            'flask': ['flask', 'Flask'],
            'fastapi': ['fastapi', 'FastAPI'],
            'pytest': ['pytest', 'conftest.py'],
            'unittest': ['unittest', 'TestCase'],
            'sqlalchemy': ['sqlalchemy', 'SQLAlchemy'],
            'requests': ['requests', 'Request']
        }
        
        for source_file in analysis['source_files']:
            try:
                with open(source_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for framework, indicators in framework_indicators.items():
                        if any(indicator in content for indicator in indicators):
                            analysis['frameworks'].add(framework)
            except Exception:
                continue
    
    def _calculate_complexity_metrics(self, source_files: List[str]) -> Dict[str, Any]:
        """Calculate complexity metrics for source files"""
        metrics = {
            'total_files': len(source_files),
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0,
            'average_complexity': 0.0,
            'complexity_distribution': {'low': 0, 'medium': 0, 'high': 0}
        }
        
        # Simple complexity analysis (could be enhanced with AST parsing)
        for file_path in source_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    metrics['total_lines'] += len(lines)
                    
                    # Count functions and classes (simplified)
                    for line in lines:
                        line = line.strip()
                        if line.startswith('def '):
                            metrics['total_functions'] += 1
                        elif line.startswith('class '):
                            metrics['total_classes'] += 1
                            
            except Exception:
                continue
        
        # Calculate average complexity (simplified)
        if metrics['total_functions'] > 0:
            metrics['average_complexity'] = metrics['total_lines'] / metrics['total_functions']
        
        # Distribute complexity (simplified heuristic)
        if metrics['average_complexity'] < 20:
            metrics['complexity_distribution']['low'] = 100
        elif metrics['average_complexity'] < 50:
            metrics['complexity_distribution']['medium'] = 100
        else:
            metrics['complexity_distribution']['high'] = 100
        
        return metrics
    
    def _run_test_generation(self) -> Dict[str, Any]:
        """Run test generation phase"""
        self.logger.info("Running test generation")
        
        results = {
            'ai_generated_tests': {},
            'template_generated_tests': {},
            'total_generated': 0,
            'generation_time': 0.0
        }
        
        start_time = time.time()
        
        # AI-powered test generation
        if self.config.enable_ai_generation:
            self.logger.info("Running AI-powered test generation")
            try:
                ai_results = self.test_generator.generate_tests_for_project(
                    use_ai=True,
                    use_templates=False
                )
                results['ai_generated_tests'] = ai_results
                results['total_generated'] += len(ai_results.get('generated_tests', []))
            except Exception as e:
                self.logger.error(f"AI test generation failed: {e}")
                results['ai_generated_tests'] = {'error': str(e)}
        
        # Template-based test generation
        if self.config.enable_template_generation:
            self.logger.info("Running template-based test generation")
            try:
                template_results = self.test_generator.generate_tests_for_project(
                    use_ai=False,
                    use_templates=True,
                    template_categories=self.config.template_categories
                )
                results['template_generated_tests'] = template_results
                results['total_generated'] += len(template_results.get('generated_tests', []))
            except Exception as e:
                self.logger.error(f"Template test generation failed: {e}")
                results['template_generated_tests'] = {'error': str(e)}
        
        results['generation_time'] = time.time() - start_time
        self.logger.info(f"Generated {results['total_generated']} tests in {results['generation_time']:.2f} seconds")
        
        return results
    
    def _run_coverage_analysis(self) -> Dict[str, Any]:
        """Run coverage analysis phase"""
        self.logger.info("Running coverage analysis")
        
        results = {
            'coverage_metrics': {},
            'mutation_results': {},
            'coverage_time': 0.0,
            'recommendations': []
        }
        
        start_time = time.time()
        
        try:
            # Run coverage analysis
            coverage_report = self.coverage_analyzer.run_coverage_analysis()
            results['coverage_metrics'] = coverage_report
            
            # Run mutation testing if enabled
            if self.config.enable_mutation_testing:
                self.logger.info("Running mutation testing")
                mutation_results = self.coverage_analyzer.run_mutation_testing()
                results['mutation_results'] = mutation_results
            
            # Generate recommendations
            results['recommendations'] = self.coverage_analyzer.generate_recommendations(coverage_report)
            
        except Exception as e:
            self.logger.error(f"Coverage analysis failed: {e}")
            results['coverage_metrics'] = {'error': str(e)}
        
        results['coverage_time'] = time.time() - start_time
        self.logger.info(f"Coverage analysis completed in {results['coverage_time']:.2f} seconds")
        
        return results
    
    def _run_performance_testing(self) -> Dict[str, Any]:
        """Run performance testing phase"""
        self.logger.info("Running performance testing")
        
        results = {
            'load_test_results': {},
            'performance_metrics': {},
            'performance_time': 0.0
        }
        
        start_time = time.time()
        
        if self.config.enable_load_testing:
            try:
                # Run enhanced benchmark
                benchmark_results = self.benchmark_runner.run_benchmark()
                results['load_test_results'] = benchmark_results
                
                # Extract performance metrics
                results['performance_metrics'] = self._extract_performance_metrics(benchmark_results)
                
            except Exception as e:
                self.logger.error(f"Performance testing failed: {e}")
                results['load_test_results'] = {'error': str(e)}
        else:
            self.logger.info("Performance testing disabled")
        
        results['performance_time'] = time.time() - start_time
        self.logger.info(f"Performance testing completed in {results['performance_time']:.2f} seconds")
        
        return results
    
    def _extract_performance_metrics(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance metrics from benchmark results"""
        metrics = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'average_execution_time': 0.0,
            'max_execution_time': 0.0,
            'min_execution_time': 0.0,
            'throughput': 0.0
        }
        
        # Extract metrics from benchmark results
        if 'test_results' in benchmark_results:
            test_results = benchmark_results['test_results']
            metrics['total_tests'] = len(test_results)
            metrics['passed_tests'] = sum(1 for r in test_results if r.get('passed', False))
            metrics['failed_tests'] = metrics['total_tests'] - metrics['passed_tests']
            
            # Calculate execution time metrics
            execution_times = [r.get('execution_time', 0) for r in test_results]
            if execution_times:
                metrics['average_execution_time'] = sum(execution_times) / len(execution_times)
                metrics['max_execution_time'] = max(execution_times)
                metrics['min_execution_time'] = min(execution_times)
        
        # Calculate throughput
        if metrics['total_tests'] > 0:
            total_time = sum(r.get('execution_time', 0) for r in benchmark_results.get('test_results', []))
            if total_time > 0:
                metrics['throughput'] = metrics['total_tests'] / total_time
        
        return metrics
    
    def _assess_quality(self) -> Dict[str, Any]:
        """Assess overall project quality based on all results"""
        self.logger.info("Assessing project quality")
        
        quality_assessment = {
            'overall_score': 0.0,
            'test_coverage_score': 0.0,
            'code_quality_score': 0.0,
            'performance_score': 0.0,
            'maintainability_score': 0.0,
            'recommendations': [],
            'risk_level': 'low'
        }
        
        try:
            # Calculate test coverage score
            if 'coverage_analysis' in self.results:
                coverage_metrics = self.results['coverage_analysis'].get('coverage_metrics', {})
                total_coverage = coverage_metrics.get('total_coverage', 0)
                quality_assessment['test_coverage_score'] = min(total_coverage, 100)
            
            # Calculate code quality score
            if 'project_analysis' in self.results:
                complexity_metrics = self.results['project_analysis'].get('complexity_metrics', {})
                avg_complexity = complexity_metrics.get('average_complexity', 0)
                # Lower complexity is better
                quality_assessment['code_quality_score'] = max(0, 100 - (avg_complexity / 2))
            
            # Calculate performance score
            if 'performance_testing' in self.results:
                performance_metrics = self.results['performance_testing'].get('performance_metrics', {})
                success_rate = 0
                if performance_metrics.get('total_tests', 0) > 0:
                    success_rate = (performance_metrics.get('passed_tests', 0) / 
                                  performance_metrics['total_tests']) * 100
                quality_assessment['performance_score'] = success_rate
            
            # Calculate maintainability score
            if 'test_generation' in self.results:
                generated_tests = self.results['test_generation'].get('total_generated', 0)
                # More generated tests indicate better maintainability
                quality_assessment['maintainability_score'] = min(generated_tests * 10, 100)
            
            # Calculate overall score
            scores = [
                quality_assessment['test_coverage_score'],
                quality_assessment['code_quality_score'],
                quality_assessment['performance_score'],
                quality_assessment['maintainability_score']
            ]
            quality_assessment['overall_score'] = sum(scores) / len(scores)
            
            # Determine risk level
            if quality_assessment['overall_score'] >= 80:
                quality_assessment['risk_level'] = 'low'
            elif quality_assessment['overall_score'] >= 60:
                quality_assessment['risk_level'] = 'medium'
            else:
                quality_assessment['risk_level'] = 'high'
            
            # Generate recommendations
            quality_assessment['recommendations'] = self._generate_quality_recommendations(quality_assessment)
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed: {e}")
            quality_assessment['error'] = str(e)
        
        self.logger.info(f"Quality assessment completed - Overall score: {quality_assessment['overall_score']:.2f}")
        return quality_assessment
    
    def _generate_quality_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        # Test coverage recommendations
        if assessment['test_coverage_score'] < 80:
            recommendations.append("Improve test coverage by adding more unit and integration tests")
        
        # Code quality recommendations
        if assessment['code_quality_score'] < 70:
            recommendations.append("Refactor complex functions to improve code maintainability")
        
        # Performance recommendations
        if assessment['performance_score'] < 90:
            recommendations.append("Optimize performance by identifying and fixing bottlenecks")
        
        # Maintainability recommendations
        if assessment['maintainability_score'] < 60:
            recommendations.append("Improve code documentation and add more comprehensive tests")
        
        # Risk-based recommendations
        if assessment['risk_level'] == 'high':
            recommendations.append("Address high-risk areas immediately to prevent production issues")
        elif assessment['risk_level'] == 'medium':
            recommendations.append("Monitor medium-risk areas and plan improvements")
        
        return recommendations
    
    def _generate_reports(self):
        """Generate comprehensive reports"""
        self.logger.info("Generating reports")
        
        try:
            # Generate JSON report
            self._generate_json_report()
            
            # Generate HTML report
            self._generate_html_report()
            
            # Generate summary report
            self._generate_summary_report()
            
            self.logger.info("Reports generated successfully")
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise
    
    def _generate_json_report(self):
        """Generate detailed JSON report"""
        report_path = os.path.join(self.config.output_dir, "unified_benchmark_report.json")
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"JSON report generated: {report_path}")
    
    def _generate_html_report(self):
        """Generate comprehensive HTML report"""
        report_path = os.path.join(self.config.output_dir, "unified_benchmark_report.html")
        
        html_content = self._create_html_report()
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report generated: {report_path}")
    
    def _create_html_report(self) -> str:
        """Create HTML report content"""
        # Create HTML structure with embedded CSS and JavaScript
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Unified Benchmark Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 2px solid #e0e0e0;
                }}
                .header h1 {{
                    color: #333;
                    margin-bottom: 10px;
                }}
                .header p {{
                    color: #666;
                    font-size: 16px;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    border-radius: 8px;
                    background-color: #f9f9f9;
                }}
                .section h2 {{
                    color: #2c3e50;
                    margin-top: 0;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .metric {{
                    background-color: white;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }}
                .metric h3 {{
                    margin: 0 0 10px 0;
                    color: #34495e;
                }}
                .metric .value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #3498db;
                }}
                .metric .unit {{
                    font-size: 14px;
                    color: #7f8c8d;
                }}
                .score {{
                    display: inline-block;
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-weight: bold;
                    color: white;
                }}
                .score.high {{ background-color: #27ae60; }}
                .score.medium {{ background-color: #f39c12; }}
                .score.low {{ background-color: #e74c3c; }}
                .recommendations {{
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 8px;
                    padding: 15px;
                }}
                .recommendations h3 {{
                    color: #856404;
                    margin-top: 0;
                }}
                .recommendations ul {{
                    margin: 10px 0;
                    padding-left: 20px;
                }}
                .recommendations li {{
                    margin-bottom: 5px;
                    color: #856404;
                }}
                .tabs {{
                    display: flex;
                    margin-bottom: 20px;
                    border-bottom: 2px solid #e0e0e0;
                }}
                .tab {{
                    padding: 10px 20px;
                    cursor: pointer;
                    background-color: #f8f9fa;
                    border: 1px solid #e0e0e0;
                    border-bottom: none;
                    border-radius: 5px 5px 0 0;
                    margin-right: 5px;
                }}
                .tab.active {{
                    background-color: #3498db;
                    color: white;
                    border-color: #3498db;
                }}
                .tab-content {{
                    display: none;
                }}
                .tab-content.active {{
                    display: block;
                }}
                .progress-bar {{
                    width: 100%;
                    height: 20px;
                    background-color: #e0e0e0;
                    border-radius: 10px;
                    overflow: hidden;
                    margin: 10px 0;
                }}
                .progress-fill {{
                    height: 100%;
                    background-color: #3498db;
                    transition: width 0.3s ease;
                }}
                .chart-container {{
                    width: 100%;
                    height: 300px;
                    margin: 20px 0;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Unified Benchmark Report</h1>
                    <p>Generated on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div class="section">
                    <h2>Executive Summary</h2>
                    <div class="metrics">
                        <div class="metric">
                            <h3>Overall Quality Score</h3>
                            <div class="value">{self.results.get('quality_assessment', {}).get('overall_score', 0):.1f}%</div>
                            <div class="unit">out of 100%</div>
                        </div>
                        <div class="metric">
                            <h3>Risk Level</h3>
                            <div class="value">
                                <span class="score {self.results.get('quality_assessment', {}).get('risk_level', 'low')}">
                                    {self.results.get('quality_assessment', {}).get('risk_level', 'low').upper()}
                                </span>
                            </div>
                        </div>
                        <div class="metric">
                            <h3>Total Tests Generated</h3>
                            <div class="value">{self.results.get('test_generation', {}).get('total_generated', 0)}</div>
                            <div class="unit">tests</div>
                        </div>
                        <div class="metric">
                            <h3>Execution Time</h3>
                            <div class="value">{self.results.get('execution_time', 0):.1f}</div>
                            <div class="unit">seconds</div>
                        </div>
                    </div>
                </div>
                
                <div class="tabs">
                    <div class="tab active" onclick="showTab('project-analysis')">Project Analysis</div>
                    <div class="tab" onclick="showTab('test-generation')">Test Generation</div>
                    <div class="tab" onclick="showTab('coverage-analysis')">Coverage Analysis</div>
                    <div class="tab" onclick="showTab('performance-testing')">Performance Testing</div>
                    <div class="tab" onclick="showTab('quality-assessment')">Quality Assessment</div>
                </div>
                
                <div id="project-analysis" class="tab-content active">
                    <div class="section">
                        <h2>Project Analysis</h2>
                        {self._create_project_analysis_html()}
                    </div>
                </div>
                
                <div id="test-generation" class="tab-content">
                    <div class="section">
                        <h2>Test Generation Results</h2>
                        {self._create_test_generation_html()}
                    </div>
                </div>
                
                <div id="coverage-analysis" class="tab-content">
                    <div class="section">
                        <h2>Coverage Analysis</h2>
                        {self._create_coverage_analysis_html()}
                    </div>
                </div>
                
                <div id="performance-testing" class="tab-content">
                    <div class="section">
                        <h2>Performance Testing</h2>
                        {self._create_performance_testing_html()}
                    </div>
                </div>
                
                <div id="quality-assessment" class="tab-content">
                    <div class="section">
                        <h2>Quality Assessment</h2>
                        {self._create_quality_assessment_html()}
                    </div>
                </div>
                
                <div class="section">
                    <h2>Recommendations</h2>
                    <div class="recommendations">
                        <h3>Quality Improvement Recommendations</h3>
                        <ul>
                            {self._create_recommendations_html()}
                        </ul>
                    </div>
                </div>
            </div>
            
            <script>
                function showTab(tabName) {{
                    // Hide all tab contents
                    const tabContents = document.querySelectorAll('.tab-content');
                    tabContents.forEach(content => content.classList.remove('active'));
                    
                    // Remove active class from all tabs
                    const tabs = document.querySelectorAll('.tab');
                    tabs.forEach(tab => tab.classList.remove('active'));
                    
                    // Show selected tab content
                    document.getElementById(tabName).classList.add('active');
                    
                    // Add active class to clicked tab
                    event.target.classList.add('active');
                }}
            </script>
        </body>
        </html>
        """
        
        return html
    
    def _create_project_analysis_html(self) -> str:
        """Create HTML content for project analysis"""
        project_analysis = self.results.get('project_analysis', {})
        
        html = f"""
        <div class="metrics">
            <div class="metric">
                <h3>Source Files</h3>
                <div class="value">{len(project_analysis.get('source_files', []))}</div>
            </div>
            <div class="metric">
                <h3>Test Files</h3>
                <div class="value">{len(project_analysis.get('test_files', []))}</div>
            </div>
            <div class="metric">
                <h3>Languages</h3>
                <div class="value">{', '.join(project_analysis.get('languages', []))}</div>
            </div>
            <div class="metric">
                <h3>Frameworks</h3>
                <div class="value">{', '.join(project_analysis.get('frameworks', []))}</div>
            </div>
        </div>
        """
        
        return html
    
    def _create_test_generation_html(self) -> str:
        """Create HTML content for test generation"""
        test_generation = self.results.get('test_generation', {})
        
        html = f"""
        <div class="metrics">
            <div class="metric">
                <h3>Total Generated Tests</h3>
                <div class="value">{test_generation.get('total_generated', 0)}</div>
            </div>
            <div class="metric">
                <h3>AI Generated Tests</h3>
                <div class="value">{len(test_generation.get('ai_generated_tests', {}).get('generated_tests', []))}</div>
            </div>
            <div class="metric">
                <h3>Template Generated Tests</h3>
                <div class="value">{len(test_generation.get('template_generated_tests', {}).get('generated_tests', []))}</div>
            </div>
            <div class="metric">
                <h3>Generation Time</h3>
                <div class="value">{test_generation.get('generation_time', 0):.1f}</div>
                <div class="unit">seconds</div>
            </div>
        </div>
        """
        
        return html
    
    def _create_coverage_analysis_html(self) -> str:
        """Create HTML content for coverage analysis"""
        coverage_analysis = self.results.get('coverage_analysis', {})
        coverage_metrics = coverage_analysis.get('coverage_metrics', {})
        
        html = f"""
        <div class="metrics">
            <div class="metric">
                <h3>Total Coverage</h3>
                <div class="value">{coverage_metrics.get('total_coverage', 0):.1f}%</div>
            </div>
            <div class="metric">
                <h3>Line Coverage</h3>
                <div class="value">{coverage_metrics.get('line_coverage', 0):.1f}%</div>
            </div>
            <div class="metric">
                <h3>Branch Coverage</h3>
                <div class="value">{coverage_metrics.get('branch_coverage', 0):.1f}%</div>
            </div>
            <div class="metric">
                <h3>Function Coverage</h3>
                <div class="value">{coverage_metrics.get('function_coverage', 0):.1f}%</div>
            </div>
        </div>
        """
        
        return html
    
    def _create_performance_testing_html(self) -> str:
        """Create HTML content for performance testing"""
        performance_testing = self.results.get('performance_testing', {})
        performance_metrics = performance_testing.get('performance_metrics', {})
        
        html = f"""
        <div class="metrics">
            <div class="metric">
                <h3>Total Tests</h3>
                <div class="value">{performance_metrics.get('total_tests', 0)}</div>
            </div>
            <div class="metric">
                <h3>Passed Tests</h3>
                <div class="value">{performance_metrics.get('passed_tests', 0)}</div>
            </div>
            <div class="metric">
                <h3>Failed Tests</h3>
                <div class="value">{performance_metrics.get('failed_tests', 0)}</div>
            </div>
            <div class="metric">
                <h3>Average Execution Time</h3>
                <div class="value">{performance_metrics.get('average_execution_time', 0):.3f}</div>
                <div class="unit">seconds</div>
            </div>
        </div>
        """
        
        return html
    
    def _create_quality_assessment_html(self) -> str:
        """Create HTML content for quality assessment"""
        quality_assessment = self.results.get('quality_assessment', {})
        
        html = f"""
        <div class="metrics">
            <div class="metric">
                <h3>Overall Score</h3>
                <div class="value">{quality_assessment.get('overall_score', 0):.1f}%</div>
            </div>
            <div class="metric">
                <h3>Test Coverage Score</h3>
                <div class="value">{quality_assessment.get('test_coverage_score', 0):.1f}%</div>
            </div>
            <div class="metric">
                <h3>Code Quality Score</h3>
                <div class="value">{quality_assessment.get('code_quality_score', 0):.1f}%</div>
            </div>
            <div class="metric">
                <h3>Performance Score</h3>
                <div class="value">{quality_assessment.get('performance_score', 0):.1f}%</div>
            </div>
        </div>
        
        <div style="margin-top: 20px;">
            <h3>Quality Scores Breakdown</h3>
            <div style="margin: 10px 0;">
                <label>Test Coverage: {quality_assessment.get('test_coverage_score', 0):.1f}%</label>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {quality_assessment.get('test_coverage_score', 0)}%"></div>
                </div>
            </div>
            <div style="margin: 10px 0;">
                <label>Code Quality: {quality_assessment.get('code_quality_score', 0):.1f}%</label>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {quality_assessment.get('code_quality_score', 0)}%"></div>
                </div>
            </div>
            <div style="margin: 10px 0;">
                <label>Performance: {quality_assessment.get('performance_score', 0):.1f}%</label>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {quality_assessment.get('performance_score', 0)}%"></div>
                </div>
            </div>
            <div style="margin: 10px 0;">
                <label>Maintainability: {quality_assessment.get('maintainability_score', 0):.1f}%</label>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {quality_assessment.get('maintainability_score', 0)}%"></div>
                </div>
            </div>
        </div>
        """
        
        return html
    
    def _create_recommendations_html(self) -> str:
        """Create HTML content for recommendations"""
        quality_assessment = self.results.get('quality_assessment', {})
        recommendations = quality_assessment.get('recommendations', [])
        
        html = ""
        for recommendation in recommendations:
            html += f"<li>{recommendation}</li>"
        
        return html
    
    def _generate_summary_report(self):
        """Generate summary report"""
        summary_path = os.path.join(self.config.output_dir, "benchmark_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("UNIFIED BENCHMARK SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Project Root: {self.config.project_root}\n")
            f.write(f"Execution Time: {self.results.get('execution_time', 0):.2f} seconds\n\n")
            
            # Quality Assessment
            quality_assessment = self.results.get('quality_assessment', {})
            f.write("QUALITY ASSESSMENT\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Score: {quality_assessment.get('overall_score', 0):.1f}%\n")
            f.write(f"Risk Level: {quality_assessment.get('risk_level', 'unknown')}\n\n")
            
            # Test Generation
            test_generation = self.results.get('test_generation', {})
            f.write("TEST GENERATION\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Generated Tests: {test_generation.get('total_generated', 0)}\n")
            f.write(f"AI Generated: {len(test_generation.get('ai_generated_tests', {}).get('generated_tests', []))}\n")
            f.write(f"Template Generated: {len(test_generation.get('template_generated_tests', {}).get('generated_tests', []))}\n\n")
            
            # Coverage Analysis
            coverage_analysis = self.results.get('coverage_analysis', {})
            coverage_metrics = coverage_analysis.get('coverage_metrics', {})
            f.write("COVERAGE ANALYSIS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Coverage: {coverage_metrics.get('total_coverage', 0):.1f}%\n")
            f.write(f"Line Coverage: {coverage_metrics.get('line_coverage', 0):.1f}%\n")
            f.write(f"Branch Coverage: {coverage_metrics.get('branch_coverage', 0):.1f}%\n")
            f.write(f"Function Coverage: {coverage_metrics.get('function_coverage', 0):.1f}%\n\n")
            
            # Performance Testing
            performance_testing = self.results.get('performance_testing', {})
            performance_metrics = performance_testing.get('performance_metrics', {})
            f.write("PERFORMANCE TESTING\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Tests: {performance_metrics.get('total_tests', 0)}\n")
            f.write(f"Passed Tests: {performance_metrics.get('passed_tests', 0)}\n")
            f.write(f"Failed Tests: {performance_metrics.get('failed_tests', 0)}\n")
            f.write(f"Average Execution Time: {performance_metrics.get('average_execution_time', 0):.3f}s\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            for recommendation in quality_assessment.get('recommendations', []):
                f.write(f"- {recommendation}\n")
        
        self.logger.info(f"Summary report generated: {summary_path}")


def main():
    """Main entry point for the unified benchmarking system"""
    parser = argparse.ArgumentParser(description="Unified Benchmarking System for Llmcode")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--source-dirs", nargs="+", default=["src", "lib", "llmcode"], 
                       help="Source directories to analyze")
    parser.add_argument("--test-dirs", nargs="+", default=["tests", "test"], 
                       help="Test directories to analyze")
    parser.add_argument("--output-dir", default="benchmark_results", 
                       help="Output directory for reports")
    parser.add_argument("--no-ai", action="store_true", help="Disable AI-powered test generation")
    parser.add_argument("--no-templates", action="store_true", help="Disable template-based test generation")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage analysis")
    parser.add_argument("--no-mutation", action="store_true", help="Disable mutation testing")
    parser.add_argument("--no-load-testing", action="store_true", help="Disable load testing")
    parser.add_argument("--max-workers", type=int, default=4, help="Maximum number of parallel workers")
    parser.add_argument("--report-format", choices=["html", "json", "both"], default="html", 
                       help="Report format")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Create configuration
    config = UnifiedBenchmarkConfig(
        project_root=args.project_root,
        source_dirs=args.source_dirs,
        test_dirs=args.test_dirs,
        output_dir=args.output_dir,
        enable_ai_generation=not args.no_ai,
        enable_template_generation=not args.no_templates,
        enable_coverage_analysis=not args.no_coverage,
        enable_mutation_testing=not args.no_mutation,
        enable_load_testing=not args.no_load_testing,
        max_workers=args.max_workers,
        report_format=args.report_format,
        verbose=args.verbose
    )
    
    # Create and run unified benchmark
    try:
        runner = UnifiedBenchmarkRunner(config)
        results = runner.run_full_benchmark()
        
        print(f"\nUnified benchmarking completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Overall quality score: {results.get('quality_assessment', {}).get('overall_score', 0):.1f}%")
        print(f"Risk level: {results.get('quality_assessment', {}).get('risk_level', 'unknown')}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
