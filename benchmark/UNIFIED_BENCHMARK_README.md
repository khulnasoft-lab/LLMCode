# Unified Benchmarking System for Llmcode

A comprehensive benchmarking framework that integrates AI-powered test generation, advanced coverage analysis, template-based testing, and performance evaluation to provide automated quality assessment and testing capabilities for Python projects.

## Overview

The Unified Benchmarking System extends the existing Llmcode benchmark framework with advanced features including:

- **AI-Powered Test Generation**: Intelligent test case generation using machine learning models
- **Template-Based Testing**: Extensive library of test templates for common patterns
- **Advanced Coverage Analysis**: Detailed code coverage metrics with mutation testing
- **Performance Testing**: Load testing and performance evaluation
- **Quality Assessment**: Automated quality scoring and risk assessment
- **Comprehensive Reporting**: HTML, JSON, and text-based reports

## Features

### 1. Test Generation

#### AI-Powered Generation
- Uses advanced language models to generate test cases
- Analyzes code structure and patterns
- Generates unit tests, integration tests, and edge cases
- Supports multiple programming languages

#### Template-Based Generation
- Extensive library of pre-built test templates
- Supports various testing categories:
  - Unit testing (functions, classes)
  - Integration testing (API endpoints)
  - Security testing (SQL injection, XSS, etc.)
  - Performance testing (benchmarks, load testing)
  - Property-based testing (Hypothesis)
  - Contract testing (Pact)
  - Mutation testing (Mutmut)
  - Load testing (Locust)

### 2. Coverage Analysis

#### Advanced Metrics
- Line coverage analysis
- Branch coverage analysis
- Function coverage analysis
- Cyclomatic complexity calculation
- Risk level assessment per file

#### Mutation Testing
- AST-based mutation generation
- Test effectiveness evaluation
- Mutation score calculation
- Surviving mutant analysis

#### Visualization
- Coverage charts and graphs
- Heat maps for coverage distribution
- Risk analysis visualization
- Interactive HTML reports

### 3. Performance Testing

#### Load Testing
- Concurrent user simulation
- Request rate testing
- Response time analysis
- Throughput measurement

#### Performance Metrics
- Average response time
- 95th percentile response time
- Success rate analysis
- Error rate tracking
- Resource utilization monitoring

### 4. Quality Assessment

#### Multi-Dimensional Scoring
- Test coverage score
- Code quality score
- Performance score
- Maintainability score
- Overall quality score

#### Risk Assessment
- Risk level classification (Low, Medium, High)
- Risk factor analysis
- Critical issue identification
- Trend analysis

#### Recommendations
- Automated improvement suggestions
- Best practice recommendations
- Priority-based action items
- Customized guidance

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Dependencies

```bash
# Core dependencies
pip install pytest pyyaml rich jinja2

# Coverage analysis
pip install coverage

# Mutation testing
pip install mutmut

# Performance testing
pip install locust psutil

# Property-based testing
pip install hypothesis

# Contract testing
pip install pact-python

# Visualization
pip install matplotlib seaborn

# AI generation (optional)
pip install openai anthropic
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd llmcode/benchmark
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python unified_benchmark.py --help
```

## Usage

### Quick Start

Run a complete benchmark analysis on your project:

```bash
# Basic usage
python unified_benchmark.py

# With custom settings
python unified_benchmark.py \
    --project-root /path/to/your/project \
    --source-dirs src lib \
    --test-dirs tests \
    --output-dir benchmark_results \
    --verbose
```

### Demo Modes

The system includes several demo modes to help you get started:

```bash
# Quick demo (recommended for first-time users)
python run_unified_benchmark.py --mode quick

# Component demonstration
python run_unified_benchmark.py --mode components

# Custom demo with specific project
python run_unified_benchmark.py --mode custom --project /path/to/project
```

### Configuration Options

#### Project Configuration
- `--project-root`: Root directory of the project to analyze
- `--source-dirs`: List of source code directories
- `--test-dirs`: List of test directories
- `--output-dir`: Output directory for reports

#### Test Generation Options
- `--no-ai`: Disable AI-powered test generation
- `--no-templates`: Disable template-based test generation
- `--max-workers`: Number of parallel workers for generation

#### Analysis Options
- `--no-coverage`: Disable coverage analysis
- `--no-mutation`: Disable mutation testing
- `--no-load-testing`: Disable load testing

#### Output Options
- `--report-format`: Report format (html, json, both)
- `--verbose`: Enable verbose logging

### Advanced Usage

#### Custom Configuration

Create a custom configuration file:

```python
# config.py
from unified_benchmark import UnifiedBenchmarkConfig

config = UnifiedBenchmarkConfig(
    project_root="/path/to/project",
    source_dirs=["src", "lib"],
    test_dirs=["tests", "test"],
    output_dir="custom_results",
    enable_ai_generation=True,
    enable_template_generation=True,
    enable_coverage_analysis=True,
    enable_mutation_testing=True,
    enable_load_testing=True,
    coverage_threshold=85.0,
    mutation_threshold=75.0,
    max_workers=8,
    verbose=True
)
```

#### Programmatic Usage

```python
from unified_benchmark import UnifiedBenchmarkRunner, UnifiedBenchmarkConfig

# Create configuration
config = UnifiedBenchmarkConfig(
    project_root=".",
    output_dir="benchmark_results"
)

# Create runner
runner = UnifiedBenchmarkRunner(config)

# Run full benchmark
results = runner.run_full_benchmark()

# Access results
quality_score = results['quality_assessment']['overall_score']
risk_level = results['quality_assessment']['risk_level']
```

## Test Templates

The system includes a comprehensive library of test templates:

### Unit Testing Templates
- `unit_function.yaml`: Function-level unit tests
- `unit_class.yaml`: Class-level unit tests

### Integration Testing Templates
- `integration_api.yaml`: API endpoint integration tests
- `contract_testing.yaml`: Service contract tests

### Security Testing Templates
- `security.yaml`: Security vulnerability tests
- Includes SQL injection, XSS, authentication tests

### Performance Testing Templates
- `performance_benchmark.yaml`: Performance benchmark tests
- `load_testing.yaml`: Load and stress tests

### Advanced Testing Templates
- `property_based.yaml`: Property-based tests with Hypothesis
- `mutation_testing.yaml`: Mutation testing with Mutmut

### Template Structure

Each template follows a consistent structure:

```yaml
name: template_name
description: Template description
category: template_category
language: python
framework: pytest
template_content: |
  # Test code with Jinja2 variables
  def test_{{ function_name }}():
      # Test implementation
      pass
variables:
  - name: function_name
    type: string
    description: Name of function to test
    default_value: my_function
tags:
  - unit
  - python
  - pytest
dependencies:
  - pytest
  - pytest-cov
complexity: medium
estimated_time: 10m
```

## Reports

The system generates comprehensive reports in multiple formats:

### HTML Reports
- Interactive web-based reports
- Visual charts and graphs
- Tabbed interface for different sections
- Responsive design

### JSON Reports
- Machine-readable format
- Complete result data
- Suitable for CI/CD integration
- API consumption

### Summary Reports
- Text-based summaries
- Key metrics and recommendations
- Executive overview
- Action items

### Report Sections

1. **Executive Summary**
   - Overall quality score
   - Risk level assessment
   - Key metrics overview

2. **Project Analysis**
   - Source code statistics
   - Test coverage analysis
   - Framework detection

3. **Test Generation**
   - Generated tests count
   - AI vs template generation
   - Generation performance

4. **Coverage Analysis**
   - Coverage metrics
   - Mutation testing results
   - Coverage visualization

5. **Performance Testing**
   - Performance metrics
   - Load test results
   - Resource utilization

6. **Quality Assessment**
   - Multi-dimensional scoring
   - Risk assessment
   - Improvement recommendations

## Integration

### CI/CD Integration

#### GitHub Actions

```yaml
name: Unified Benchmark
on: [push, pull_request]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run unified benchmark
      run: |
        python unified_benchmark.py \
          --project-root . \
          --output-dir benchmark_results \
          --report-format json
    
    - name: Upload results
      uses: actions/upload-artifact@v2
      with:
        name: benchmark-results
        path: benchmark_results/
```

#### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    stages {
        stage('Benchmark') {
            steps {
                sh '''
                    python unified_benchmark.py \
                        --project-root . \
                        --output-dir benchmark_results \
                        --report-format both
                '''
            }
        }
        
        stage('Quality Gate') {
            steps {
                script {
                    def results = readJSON file: 'benchmark_results/unified_benchmark_report.json'
                    def qualityScore = results.quality_assessment.overall_score
                    
                    if (qualityScore < 70) {
                        error("Quality score ${qualityScore}% is below threshold")
                    }
                }
            }
        }
    }
}
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: unified-benchmark
        name: Unified Benchmark
        entry: python unified_benchmark.py
        language: python
        args: [--project-root, ., --output-dir, benchmark_results, --report-format, json]
        pass_filenames: false
        always_run: true
```

## Performance Considerations

### System Requirements
- **CPU**: Multi-core processor recommended
- **Memory**: 8GB RAM minimum, 16GB+ recommended
- **Storage**: 1GB+ free space for reports
- **Network**: Required for AI-powered generation

### Optimization Tips

1. **Parallel Processing**
   ```bash
   python unified_benchmark.py --max-workers 8
   ```

2. **Selective Analysis**
   ```bash
   python unified_benchmark.py --no-mutation --no-load-testing
   ```

3. **Incremental Analysis**
   ```bash
   python unified_benchmark.py --source-dirs modified_src
   ```

### Large Projects

For large codebases, consider:

1. **Incremental Analysis**: Analyze only changed files
2. **Sampling**: Use representative file samples
3. **Distributed Processing**: Run across multiple machines
4. **Caching**: Enable result caching for repeated runs

## Troubleshooting

### Common Issues

#### Installation Problems
```bash
# Ensure Python version compatibility
python --version

# Install in virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Memory Issues
```bash
# Reduce worker count
python unified_benchmark.py --max-workers 2

# Disable memory-intensive features
python unified_benchmark.py --no-mutation --no-load-testing
```

#### Permission Issues
```bash
# Ensure write permissions for output directory
chmod 755 benchmark_results

# Run with appropriate user permissions
sudo python unified_benchmark.py  # Not recommended, use proper permissions
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
python unified_benchmark.py --verbose
```

### Log Files

Check log files for detailed error information:
- `benchmark_results/unified_benchmark.log`
- `unified_benchmark_demo.log`

## Contributing

### Development Setup

1. Fork the repository
2. Create a development branch
3. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
4. Run tests:
   ```bash
   pytest tests/
   ```

### Adding New Templates

1. Create template file in `test_templates/`
2. Follow the template structure
3. Add template variables and metadata
4. Test template rendering:
   ```bash
   python -c "from template_manager import TemplateManager; tm = TemplateManager(); print(tm.render_template('template_name.yaml', {'variable': 'value'}))"
   ```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add comprehensive docstrings
- Include unit tests for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review log files for error details
3. Create an issue in the project repository
4. Contact the development team

## Changelog

### Version 1.0.0
- Initial release of unified benchmarking system
- AI-powered test generation
- Template-based testing framework
- Advanced coverage analysis
- Performance testing capabilities
- Comprehensive reporting system
- CI/CD integration support

## Roadmap

### Planned Features
- [ ] Multi-language support (JavaScript, TypeScript, Java)
- [ ] Advanced AI models integration
- [ ] Real-time monitoring dashboard
- [ ] Historical trend analysis
- [ ] Custom rule engine
- [ ] Integration with popular IDEs
- [ ] Cloud-based benchmarking service
- [ ] Advanced anomaly detection
- [ ] Predictive quality analysis

### Performance Improvements
- [ ] Distributed processing
- [ ] Incremental analysis optimization
- [ ] Memory usage optimization
- [ ] Caching improvements
- [ ] Parallel processing enhancements

### User Experience
- [ ] Web-based interface
- [ ] Interactive configuration
- [ ] Real-time progress tracking
- [ ] Custom report templates
- [ ] Integration with project management tools
