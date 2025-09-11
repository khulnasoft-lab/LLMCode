#!/usr/bin/env python3
"""
Unified Benchmarking System - Demo Script

This script demonstrates how to use the unified benchmarking system
to analyze a project, generate tests, and produce comprehensive reports.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add benchmark directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from unified_benchmark import UnifiedBenchmarkRunner, UnifiedBenchmarkConfig


def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('unified_benchmark_demo.log')
        ]
    )


def create_demo_config():
    """Create a demo configuration for the unified benchmark"""
    config = UnifiedBenchmarkConfig(
        project_root=".",  # Current directory
        source_dirs=["llmcode"],  # Analyze llmcode directory
        test_dirs=["tests"],  # Test directories
        output_dir="demo_benchmark_results",  # Output directory
        
        # Test generation settings
        enable_ai_generation=False,  # Disable AI for demo (requires API key)
        enable_template_generation=True,  # Enable template-based generation
        template_categories=["unit", "integration"],  # Focus on unit and integration tests
        
        # Coverage analysis settings
        enable_coverage_analysis=True,  # Enable coverage analysis
        enable_mutation_testing=False,  # Disable mutation testing for demo (slow)
        coverage_threshold=70.0,  # Lower threshold for demo
        
        # Performance testing settings
        enable_load_testing=False,  # Disable load testing for demo
        
        # Output settings
        report_format="html",  # Generate HTML reports
        verbose=True,  # Enable verbose logging
        
        # Parallel processing
        max_workers=2,  # Use fewer workers for demo
        enable_parallel=True
    )
    
    return config


def run_quick_demo():
    """Run a quick demonstration of the unified benchmarking system"""
    print("=" * 60)
    print("UNIFIED BENCHMARKING SYSTEM - QUICK DEMO")
    print("=" * 60)
    
    # Setup logging
    setup_logging(verbose=True)
    
    # Create configuration
    config = create_demo_config()
    
    print(f"\nConfiguration:")
    print(f"  Project Root: {config.project_root}")
    print(f"  Source Directories: {config.source_dirs}")
    print(f"  Test Directories: {config.test_dirs}")
    print(f"  Output Directory: {config.output_dir}")
    print(f"  AI Generation: {config.enable_ai_generation}")
    print(f"  Template Generation: {config.enable_template_generation}")
    print(f"  Coverage Analysis: {config.enable_coverage_analysis}")
    print(f"  Load Testing: {config.enable_load_testing}")
    
    try:
        # Create unified benchmark runner
        print(f"\nInitializing unified benchmark runner...")
        runner = UnifiedBenchmarkRunner(config)
        
        # Run full benchmark
        print(f"\nRunning unified benchmark...")
        results = runner.run_full_benchmark()
        
        # Display summary
        print(f"\n" + "=" * 60)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        
        quality_assessment = results.get('quality_assessment', {})
        test_generation = results.get('test_generation', {})
        coverage_analysis = results.get('coverage_analysis', {})
        
        print(f"Overall Quality Score: {quality_assessment.get('overall_score', 0):.1f}%")
        print(f"Risk Level: {quality_assessment.get('risk_level', 'unknown')}")
        print(f"Total Tests Generated: {test_generation.get('total_generated', 0)}")
        print(f"Total Coverage: {coverage_analysis.get('coverage_metrics', {}).get('total_coverage', 0):.1f}%")
        print(f"Execution Time: {results.get('execution_time', 0):.2f} seconds")
        
        print(f"\nReports generated in: {config.output_dir}")
        print(f"  - HTML Report: {config.output_dir}/unified_benchmark_report.html")
        print(f"  - JSON Report: {config.output_dir}/unified_benchmark_report.json")
        print(f"  - Summary Report: {config.output_dir}/benchmark_summary.txt")
        
        # Display recommendations
        recommendations = quality_assessment.get('recommendations', [])
        if recommendations:
            print(f"\nRecommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        return True
        
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return False


def run_component_demo():
    """Run individual component demonstrations"""
    print("=" * 60)
    print("COMPONENT DEMONSTRATIONS")
    print("=" * 60)
    
    setup_logging(verbose=True)
    config = create_demo_config()
    
    try:
        runner = UnifiedBenchmarkRunner(config)
        
        # Demo 1: Project Analysis
        print(f"\n1. Project Analysis Demo")
        print("-" * 30)
        project_analysis = runner._analyze_project()
        print(f"Source files found: {len(project_analysis.get('source_files', []))}")
        print(f"Test files found: {len(project_analysis.get('test_files', []))}")
        print(f"Languages detected: {', '.join(project_analysis.get('languages', []))}")
        print(f"Frameworks detected: {', '.join(project_analysis.get('frameworks', []))}")
        
        # Demo 2: Template Management
        print(f"\n2. Template Management Demo")
        print("-" * 30)
        templates = runner.template_manager.list_templates()
        print(f"Available templates: {len(templates)}")
        for template in templates[:5]:  # Show first 5
            print(f"  - {template['name']} ({template['category']})")
        
        # Demo 3: Test Generation (Templates only)
        print(f"\n3. Test Generation Demo (Templates)")
        print("-" * 30)
        test_results = runner._run_test_generation()
        print(f"Generated tests: {test_results.get('total_generated', 0)}")
        print(f"Template-based tests: {len(test_results.get('template_generated_tests', {}).get('generated_tests', []))}")
        
        # Demo 4: Coverage Analysis
        print(f"\n4. Coverage Analysis Demo")
        print("-" * 30)
        coverage_results = runner._run_coverage_analysis()
        coverage_metrics = coverage_results.get('coverage_metrics', {})
        print(f"Total coverage: {coverage_metrics.get('total_coverage', 0):.1f}%")
        print(f"Line coverage: {coverage_metrics.get('line_coverage', 0):.1f}%")
        print(f"Branch coverage: {coverage_metrics.get('branch_coverage', 0):.1f}%")
        
        # Demo 5: Quality Assessment
        print(f"\n5. Quality Assessment Demo")
        print("-" * 30)
        quality_assessment = runner._assess_quality()
        print(f"Overall score: {quality_assessment.get('overall_score', 0):.1f}%")
        print(f"Risk level: {quality_assessment.get('risk_level', 'unknown')}")
        
        print(f"\nComponent demonstrations completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during component demo: {e}")
        return False


def run_custom_demo(project_path, components):
    """Run custom demo with specified project and components"""
    print("=" * 60)
    print("CUSTOM DEMO")
    print("=" * 60)
    
    setup_logging(verbose=True)
    
    # Create custom configuration
    config = UnifiedBenchmarkConfig(
        project_root=project_path,
        output_dir=f"custom_benchmark_results_{Path(project_path).name}",
        enable_ai_generation=False,
        enable_template_generation='template' in components,
        enable_coverage_analysis='coverage' in components,
        enable_mutation_testing=False,
        enable_load_testing='performance' in components,
        verbose=True
    )
    
    try:
        runner = UnifiedBenchmarkRunner(config)
        
        # Run selected components
        results = {}
        
        if 'analysis' in components:
            print(f"\nRunning project analysis...")
            results['project_analysis'] = runner._analyze_project()
        
        if 'template' in components:
            print(f"\nRunning test generation...")
            results['test_generation'] = runner._run_test_generation()
        
        if 'coverage' in components:
            print(f"\nRunning coverage analysis...")
            results['coverage_analysis'] = runner._run_coverage_analysis()
        
        if 'performance' in components:
            print(f"\nRunning performance testing...")
            results['performance_testing'] = runner._run_performance_testing()
        
        if 'quality' in components:
            print(f"\nRunning quality assessment...")
            # Set results for quality assessment
            runner.results = results
            results['quality_assessment'] = runner._assess_quality()
        
        # Generate reports
        if results:
            runner.results = results
            runner._generate_reports()
        
        print(f"\nCustom demo completed!")
        print(f"Results saved to: {config.output_dir}")
        
        return True
        
    except Exception as e:
        print(f"Error during custom demo: {e}")
        return False


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Unified Benchmarking System Demo")
    parser.add_argument("--mode", choices=['quick', 'components', 'custom'], 
                       default='quick', help="Demo mode")
    parser.add_argument("--project", help="Project path for custom demo")
    parser.add_argument("--components", nargs='+', 
                       choices=['analysis', 'template', 'coverage', 'performance', 'quality'],
                       default=['analysis', 'template', 'coverage', 'quality'],
                       help="Components to run for custom demo")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        success = run_quick_demo()
    elif args.mode == 'components':
        success = run_component_demo()
    elif args.mode == 'custom':
        if not args.project:
            print("Error: --project is required for custom demo")
            return 1
        success = run_custom_demo(args.project, args.components)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
