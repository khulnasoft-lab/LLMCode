#!/usr/bin/env python3
"""
Test Template Management System

This module provides a comprehensive system for managing test templates,
including validation, categorization, and dynamic template generation.
"""

import os
import json
import yaml
import re
import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Type
from enum import Enum
from collections import defaultdict
import concurrent.futures
import threading
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.tree import Tree
import jinja2


class TemplateCategory(Enum):
    """Template categories"""
    UNIT = "unit"
    INTEGRATION = "integration"
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    REGRESSION = "regression"
    ACCEPTANCE = "acceptance"
    CONTRACT = "contract"
    PROPERTY = "property"
    MUTATION = "mutation"


class TemplateLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CSHARP = "csharp"
    CPP = "cpp"
    GO = "go"
    RUST = "rust"
    RUBY = "ruby"
    PHP = "php"


class TemplateFramework(Enum):
    """Testing frameworks"""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    MOCHA = "mocha"
    JUNIT = "junit"
    TESTNG = "testng"
    NUNIT = "nunit"
    CATCH2 = "catch2"
    GO_TEST = "go_test"
    RUST_TEST = "rust_test"
    RSPEC = "rspec"
    PHPUNIT = "phpunit"


@dataclass
class TemplateVariable:
    """Template variable definition"""
    name: str
    type: str
    description: str
    default_value: Any = None
    required: bool = True
    validation_pattern: Optional[str] = None
    example_values: List[Any] = field(default_factory=list)
    
    def validate(self, value: Any) -> Tuple[bool, Optional[str]]:
        """Validate a value against the variable definition"""
        if self.required and value is None:
            return False, f"Variable '{self.name}' is required"
        
        if value is None:
            return True, None
        
        # Type validation
        expected_type = self.type.lower()
        if expected_type == "string":
            if not isinstance(value, str):
                return False, f"Variable '{self.name}' must be a string"
        elif expected_type == "integer":
            if not isinstance(value, int):
                return False, f"Variable '{self.name}' must be an integer"
        elif expected_type == "float":
            if not isinstance(value, (int, float)):
                return False, f"Variable '{self.name}' must be a number"
        elif expected_type == "boolean":
            if not isinstance(value, bool):
                return False, f"Variable '{self.name}' must be a boolean"
        elif expected_type == "list":
            if not isinstance(value, list):
                return False, f"Variable '{self.name}' must be a list"
        elif expected_type == "dict":
            if not isinstance(value, dict):
                return False, f"Variable '{self.name}' must be a dictionary"
        
        # Pattern validation
        if self.validation_pattern and isinstance(value, str):
            if not re.match(self.validation_pattern, value):
                return False, f"Variable '{self.name}' does not match pattern: {self.validation_pattern}"
        
        return True, None


@dataclass
class TestTemplate:
    """Test template definition"""
    name: str
    description: str
    category: TemplateCategory
    language: TemplateLanguage
    framework: TemplateFramework
    template_content: str
    variables: List[TemplateVariable] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    author: str = "Unknown"
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    complexity: str = "medium"
    estimated_time: str = "5m"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'description': self.description,
            'category': self.category.value,
            'language': self.language.value,
            'framework': self.framework.value,
            'template_content': self.template_content,
            'variables': [asdict(var) for var in self.variables],
            'tags': self.tags,
            'dependencies': self.dependencies,
            'author': self.author,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'complexity': self.complexity,
            'estimated_time': self.estimated_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestTemplate':
        """Create template from dictionary"""
        variables = [TemplateVariable(**var_data) for var_data in data.get('variables', [])]
        
        return cls(
            name=data['name'],
            description=data['description'],
            category=TemplateCategory(data['category']),
            language=TemplateLanguage(data['language']),
            framework=TemplateFramework(data['framework']),
            template_content=data['template_content'],
            variables=variables,
            tags=data.get('tags', []),
            dependencies=data.get('dependencies', []),
            author=data.get('author', 'Unknown'),
            version=data.get('version', '1.0.0'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            complexity=data.get('complexity', 'medium'),
            estimated_time=data.get('estimated_time', '5m')
        )
    
    def validate_variables(self, variables: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate provided variables against template requirements"""
        errors = []
        
        # Check for required variables
        for var in self.variables:
            if var.required and var.name not in variables:
                errors.append(f"Required variable '{var.name}' is missing")
        
        # Validate variable values
        for var_name, var_value in variables.items():
            var_def = next((v for v in self.variables if v.name == var_name), None)
            if var_def:
                is_valid, error = var_def.validate(var_value)
                if not is_valid:
                    errors.append(error)
        
        return len(errors) == 0, errors
    
    def render(self, variables: Dict[str, Any]) -> str:
        """Render template with provided variables"""
        # Validate variables
        is_valid, errors = self.validate_variables(variables)
        if not is_valid:
            raise ValueError(f"Variable validation failed: {', '.join(errors)}")
        
        # Set default values for missing variables
        for var in self.variables:
            if var.name not in variables and var.default_value is not None:
                variables[var.name] = var.default_value
        
        # Render template using Jinja2
        env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        template = env.from_string(self.template_content)
        return template.render(**variables)


class TemplateValidator:
    """Template validation utilities"""
    
    @staticmethod
    def validate_template(template: TestTemplate) -> Tuple[bool, List[str]]:
        """Validate template structure and content"""
        errors = []
        
        # Basic validation
        if not template.name:
            errors.append("Template name is required")
        
        if not template.description:
            errors.append("Template description is required")
        
        if not template.template_content:
            errors.append("Template content is required")
        
        # Validate template syntax
        try:
            env = jinja2.Environment()
            env.parse(template.template_content)
        except jinja2.TemplateSyntaxError as e:
            errors.append(f"Template syntax error: {e}")
        
        # Validate variable references
        referenced_vars = TemplateValidator._extract_variables(template.template_content)
        defined_vars = {var.name for var in template.variables}
        
        # Check for undefined variables
        undefined_vars = referenced_vars - defined_vars
        if undefined_vars:
            errors.append(f"Undefined variables referenced: {', '.join(undefined_vars)}")
        
        # Check for unused variables
        unused_vars = defined_vars - referenced_vars
        if unused_vars:
            errors.append(f"Unused variables defined: {', '.join(unused_vars)}")
        
        # Language-specific validation
        if template.language == TemplateLanguage.PYTHON:
            errors.extend(TemplateValidator._validate_python_template(template))
        elif template.language == TemplateLanguage.JAVASCRIPT:
            errors.extend(TemplateValidator._validate_javascript_template(template))
        
        return len(errors) == 0, errors
    
    @staticmethod
    def _extract_variables(template_content: str) -> Set[str]:
        """Extract variable references from template content"""
        pattern = r'\{\{\s*(\w+)\s*\}\}'
        return set(re.findall(pattern, template_content))
    
    @staticmethod
    def _validate_python_template(template: TestTemplate) -> List[str]:
        """Validate Python-specific template content"""
        errors = []
        
        try:
            # Try to parse as Python code (basic syntax check)
            ast.parse(template.template_content)
        except SyntaxError as e:
            errors.append(f"Python syntax error: {e}")
        
        # Check for pytest-specific patterns
        if template.framework == TemplateFramework.PYTEST:
            if 'def test_' not in template.template_content:
                errors.append("Pytest template should contain test functions starting with 'test_'")
        
        return errors
    
    @staticmethod
    def _validate_javascript_template(template: TestTemplate) -> List[str]:
        """Validate JavaScript-specific template content"""
        errors = []
        
        # Basic JavaScript syntax checks
        if template.framework == TemplateFramework.JEST:
            if 'test(' not in template.template_content and 'describe(' not in template.template_content:
                errors.append("Jest template should contain test() or describe() calls")
        
        return errors


class TemplateGenerator:
    """Dynamic template generation utilities"""
    
    @staticmethod
    def generate_unit_function_template(language: TemplateLanguage = TemplateLanguage.PYTHON) -> TestTemplate:
        """Generate a unit function test template"""
        if language == TemplateLanguage.PYTHON:
            return TestTemplate(
                name="unit_function_python",
                description="Unit test template for individual functions in Python",
                category=TemplateCategory.UNIT,
                language=language,
                framework=TemplateFramework.PYTEST,
                template_content="""import pytest
from {{ module_name }} import {{ function_name }}


def test_{{ function_name }}_basic():
    \"\"\"Test basic functionality of {{ function_name }}\"\"\"
    # Arrange
    {{ arrange_section }}
    
    # Act
    result = {{ function_name }}({{ function_args }})
    
    # Assert
    assert result == {{ expected_result }}


def test_{{ function_name }}_edge_cases():
    \"\"\"Test edge cases for {{ function_name }}\"\"\"
    # Test with empty inputs
    {% if edge_cases %}
    {% for case in edge_cases %}
    assert {{ function_name }}({{ case.input }}) == {{ case.expected }}
    {% endfor %}
    {% endif %}
    
    # Test with None values if applicable
    {% if handles_none %}
    assert {{ function_name }}(None) == {{ none_result }}
    {% endif %}


def test_{{ function_name }}_error_handling():
    \"\"\"Test error handling in {{ function_name }}\"\"\"
    {% if raises_exception %}
    with pytest.raises({{ exception_type }}):
        {{ function_name }}({{ invalid_input }})
    {% endif %}
    
    {% if returns_none %}
    assert {{ function_name }}({{ invalid_input }}) is None
    {% endif %}


def test_{{ function_name }}_performance():
    \"\"\"Test performance of {{ function_name }}\"\"\"
    import time
    
    start_time = time.time()
    for _ in range({{ performance_iterations }}):
        {{ function_name }}({{ performance_input }})
    end_time = time.time()
    
    assert end_time - start_time < {{ performance_threshold }}
""",
                variables=[
                    TemplateVariable("module_name", "string", "Module containing the function", "my_module"),
                    TemplateVariable("function_name", "string", "Name of the function to test", "my_function"),
                    TemplateVariable("function_args", "string", "Arguments to pass to the function", "arg1, arg2"),
                    TemplateVariable("expected_result", "string", "Expected result", "expected_value"),
                    TemplateVariable("arrange_section", "string", "Setup code for the test", "# Setup test data"),
                    TemplateVariable("edge_cases", "list", "List of edge cases to test", []),
                    TemplateVariable("handles_none", "boolean", "Whether function handles None values", False),
                    TemplateVariable("none_result", "string", "Result when input is None", "None"),
                    TemplateVariable("raises_exception", "boolean", "Whether function raises exceptions", True),
                    TemplateVariable("exception_type", "string", "Type of exception raised", "ValueError"),
                    TemplateVariable("invalid_input", "string", "Invalid input for error testing", "invalid_value"),
                    TemplateVariable("returns_none", "boolean", "Whether function returns None for invalid input", False),
                    TemplateVariable("performance_iterations", "integer", "Number of iterations for performance test", 1000),
                    TemplateVariable("performance_input", "string", "Input for performance testing", "test_input"),
                    TemplateVariable("performance_threshold", "float", "Performance threshold in seconds", 1.0)
                ],
                tags=["unit", "function", "python", "pytest"],
                dependencies=["pytest"],
                complexity="simple",
                estimated_time="3m"
            )
        
        # Add more language support as needed
        raise NotImplementedError(f"Unit function template not implemented for {language}")
    
    @staticmethod
    def generate_integration_api_template(language: TemplateLanguage = TemplateLanguage.PYTHON) -> TestTemplate:
        """Generate an integration API test template"""
        if language == TemplateLanguage.PYTHON:
            return TestTemplate(
                name="integration_api_python",
                description="Integration test template for API endpoints in Python",
                category=TemplateCategory.INTEGRATION,
                language=language,
                framework=TemplateFramework.PYTEST,
                template_content="""import pytest
import json
from fastapi.testclient import TestClient
from {{ app_module }} import app


@pytest.fixture
def client():
    \"\"\"Create test client\"\"\"
    return TestClient(app)


@pytest.fixture
def mock_database():
    \"\"\"Mock database fixture\"\"\"
    # Setup mock database
    {% if database_setup %}
    {{ database_setup }}
    {% endif %}
    
    yield
    
    # Cleanup
    {% if database_cleanup %}
    {{ database_cleanup }}
    {% endif %}


def test_{{ endpoint_name }}_get_success(client, mock_database):
    \"\"\"Test successful GET request to {{ endpoint_name }}\"\"\"
    response = client.get("{{ endpoint_path }}")
    
    assert response.status_code == 200
    data = response.json()
    
    {% if expected_fields %}
    # Validate response structure
    for field in {{ expected_fields }}:
        assert field in data
    {% endif %}
    
    {% if expected_values %}
    # Validate specific values
    {% for key, value in expected_values.items() %}
    assert data["{{ key }}"] == {{ value }}
    {% endfor %}
    {% endif %}


def test_{{ endpoint_name }}_post_success(client, mock_database):
    \"\"\"Test successful POST request to {{ endpoint_name }}\"\"\"
    payload = {
        {% for key, value in post_payload.items() %}
        "{{ key }}": {{ value }},
        {% endfor %}
    }
    
    response = client.post("{{ endpoint_path }}", json=payload)
    
    assert response.status_code == {{ post_success_status }}
    data = response.json()
    
    {% if post_response_validation %}
    {{ post_response_validation }}
    {% endif %}


def test_{{ endpoint_name }}_put_success(client, mock_database):
    \"\"\"Test successful PUT request to {{ endpoint_name }}\"\"\"
    payload = {
        {% for key, value in put_payload.items() %}
        "{{ key }}": {{ value }},
        {% endfor %}
    }
    
    response = client.put("{{ endpoint_path }}/{{ resource_id }}", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    {% if put_response_validation %}
    {{ put_response_validation }}
    {% endif %}


def test_{{ endpoint_name }}_delete_success(client, mock_database):
    \"\"\"Test successful DELETE request to {{ endpoint_name }}\"\"\"
    response = client.delete("{{ endpoint_path }}/{{ resource_id }}")
    
    assert response.status_code == {{ delete_success_status }}
    
    # Verify resource is deleted
    get_response = client.get("{{ endpoint_path }}/{{ resource_id }}")
    assert get_response.status_code == 404


def test_{{ endpoint_name }}_error_handling(client, mock_database):
    \"\"\"Test error handling for {{ endpoint_name }}\"\"\"
    {% if error_cases %}
    {% for error_case in error_cases %}
    # Test {{ error_case.description }}
    response = client.{{ error_case.method.lower() }}("{{ error_case.path }}", {{ error_case.params }})
    assert response.status_code == {{ error_case.expected_status }}
    {% endfor %}
    {% endif %}


def test_{{ endpoint_name }}_authentication(client):
    \"\"\"Test authentication requirements for {{ endpoint_name }}\"\"\"
    {% if requires_auth %}
    # Test without authentication
    response = client.get("{{ endpoint_path }}")
    assert response.status_code == 401
    
    # Test with invalid authentication
    response = client.get("{{ endpoint_path }}", headers={"Authorization": "Invalid token"})
    assert response.status_code == 401
    {% else %}
    # Test without authentication (should succeed)
    response = client.get("{{ endpoint_path }}")
    assert response.status_code == 200
    {% endif %}


def test_{{ endpoint_name }}_rate_limiting(client):
    \"\"\"Test rate limiting for {{ endpoint_name }}\"\"\"
    {% if has_rate_limiting %}
    # Make multiple requests quickly
    for i in range({{ rate_limit_requests }}):
        response = client.get("{{ endpoint_path }}")
        if response.status_code == 429:
            break
    
    # Verify rate limiting works
    assert response.status_code == 429
    {% endif %}


def test_{{ endpoint_name }}_validation(client, mock_database):
    \"\"\"Test input validation for {{ endpoint_name }}\"\"\"
    {% if validation_cases %}
    {% for case in validation_cases %}
    # Test {{ case.description }}
    payload = {{ case.payload }}
    response = client.{{ case.method.lower() }}("{{ case.path }}", json=payload)
    assert response.status_code == {{ case.expected_status }}
    {% endfor %}
    {% endif %}
""",
                variables=[
                    TemplateVariable("app_module", "string", "Module containing the FastAPI app", "main"),
                    TemplateVariable("endpoint_name", "string", "Name of the endpoint", "my_endpoint"),
                    TemplateVariable("endpoint_path", "string", "Path of the endpoint", "/api/my-endpoint"),
                    TemplateVariable("expected_fields", "list", "Expected fields in response", ["id", "name", "created_at"]),
                    TemplateVariable("expected_values", "dict", "Expected values in response", {}),
                    TemplateVariable("post_payload", "dict", "Payload for POST requests", {"name": "test", "value": 123}),
                    TemplateVariable("post_success_status", "integer", "Expected status for successful POST", 201),
                    TemplateVariable("post_response_validation", "string", "Validation for POST response", ""),
                    TemplateVariable("put_payload", "dict", "Payload for PUT requests", {"name": "updated", "value": 456}),
                    TemplateVariable("resource_id", "string", "ID of resource for PUT/DELETE", "1"),
                    TemplateVariable("put_response_validation", "string", "Validation for PUT response", ""),
                    TemplateVariable("delete_success_status", "integer", "Expected status for successful DELETE", 204),
                    TemplateVariable("error_cases", "list", "Error cases to test", []),
                    TemplateVariable("requires_auth", "boolean", "Whether endpoint requires authentication", True),
                    TemplateVariable("has_rate_limiting", "boolean", "Whether endpoint has rate limiting", True),
                    TemplateVariable("rate_limit_requests", "integer", "Number of requests for rate limit test", 100),
                    TemplateVariable("validation_cases", "list", "Validation cases to test", []),
                    TemplateVariable("database_setup", "string", "Database setup code", ""),
                    TemplateVariable("database_cleanup", "string", "Database cleanup code", "")
                ],
                tags=["integration", "api", "python", "pytest", "fastapi"],
                dependencies=["pytest", "fastapi", "httpx"],
                complexity="medium",
                estimated_time="10m"
            )
        
        raise NotImplementedError(f"Integration API template not implemented for {language}")
    
    @staticmethod
    def generate_security_template(language: TemplateLanguage = TemplateLanguage.PYTHON) -> TestTemplate:
        """Generate a security test template"""
        if language == TemplateLanguage.PYTHON:
            return TestTemplate(
                name="security_python",
                description="Security test template for Python applications",
                category=TemplateCategory.SECURITY,
                language=language,
                framework=TemplateFramework.PYTEST,
                template_content="""import pytest
import json
from fastapi.testclient import TestClient
from {{ app_module }} import app


@pytest.fixture
def client():
    \"\"\"Create test client\"\"\"
    return TestClient(app)


def test_sql_injection_prevention(client):
    \"\"\"Test SQL injection prevention\"\"\"
    malicious_inputs = [
        "1' OR '1'='1",
        "1; DROP TABLE users; --",
        "1' UNION SELECT username, password FROM users--",
        "' OR '1'='1' --",
        "1' AND 1=1--"
    ]
    
    {% if sql_injection_endpoints %}
    {% for endpoint in sql_injection_endpoints %}
    for malicious_input in malicious_inputs:
        response = client.get("{{ endpoint.path }}", params={"{{ endpoint.param }}": malicious_input})
        assert response.status_code != 500  # Should not cause server error
        assert "error" in response.json().get("message", "").lower()
    {% endfor %}
    {% endif %}


def test_xss_prevention(client):
    \"\"\"Test XSS prevention\"\"\"
    malicious_scripts = [
        "<script>alert('XSS')</script>",
        "javascript:alert('XSS')",
        "<img src='x' onerror='alert(1)'>",
        "<svg onload=alert('XSS')>",
        "'\"><script>alert(document.cookie)</script>"
    ]
    
    {% if xss_endpoints %}
    {% for endpoint in xss_endpoints %}
    for script in malicious_scripts:
        payload = {"{{ endpoint.field }}": script}
        response = client.post("{{ endpoint.path }}", json=payload)
        
        # Response should not contain the malicious script
        assert script not in response.text
        assert response.status_code not in [500, 400]  # Should not cause server error or bad request
    {% endfor %}
    {% endif %}


def test_authentication_bypass(client):
    \"\"\"Test authentication bypass prevention\"\"\"
    {% if protected_endpoints %}
    {% for endpoint in protected_endpoints %}
    # Test without authentication
    response = client.{{ endpoint.method.lower() }}("{{ endpoint.path }}")
    assert response.status_code == 401
    
    # Test with invalid token
    response = client.{{ endpoint.method.lower() }}("{{ endpoint.path }}", 
                                                    headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 401
    
    # Test with expired token
    response = client.{{ endpoint.method.lower() }}("{{ endpoint.path }}", 
                                                    headers={"Authorization": "Bearer expired_token"})
    assert response.status_code == 401
    {% endfor %}
    {% endif %}


def test_authorization_checks(client):
    \"\"\"Test authorization checks\"\"\"
    {% if authorization_endpoints %}
    {% for endpoint in authorization_endpoints %}
    # Test with regular user trying to access admin resource
    response = client.{{ endpoint.method.lower() }}("{{ endpoint.path }}", 
                                                    headers={"Authorization": "Bearer regular_user_token"})
    assert response.status_code == 403
    
    # Test with admin user
    response = client.{{ endpoint.method.lower() }}("{{ endpoint.path }}", 
                                                    headers={"Authorization": "Bearer admin_token"})
    assert response.status_code == {{ endpoint.expected_status }}
    {% endfor %}
    {% endif %}


def test_sensitive_data_exposure(client):
    \"\"\"Test sensitive data exposure prevention\"\"\"
    {% if sensitive_data_endpoints %}
    {% for endpoint in sensitive_data_endpoints %}
    response = client.get("{{ endpoint.path }}")
    data = response.json()
    
    # Check for sensitive data
    sensitive_fields = {{ endpoint.sensitive_fields }}
    for field in sensitive_fields:
        assert field not in data, f"Sensitive field '{field}' exposed in response"
    {% endfor %}
    {% endif %}


def test_file_path_traversal(client):
    \"\"\"Test file path traversal prevention\"\"\"
    malicious_paths = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
    ]
    
    {% if file_endpoints %}
    {% for endpoint in file_endpoints %}
    for malicious_path in malicious_paths:
        response = client.get("{{ endpoint.path }}", params={"file": malicious_path})
        assert response.status_code != 200  # Should not succeed
        assert response.status_code != 500  # Should not cause server error
    {% endfor %}
    {% endif %}


def test_command_injection(client):
    \"\"\"Test command injection prevention\"\"\"
    malicious_commands = [
        "test; rm -rf /",
        "test | cat /etc/passwd",
        "test && ls -la",
        "test || whoami",
        "$(cat /etc/passwd)"
    ]
    
    {% if command_endpoints %}
    {% for endpoint in command_endpoints %}
    for command in malicious_commands:
        payload = {"{{ endpoint.field }}": command}
        response = client.post("{{ endpoint.path }}", json=payload)
        assert response.status_code != 500  # Should not cause server error
    {% endfor %}
    {% endif %}


def test_csrf_protection(client):
    \"\"\"Test CSRF protection\"\"\"
    {% if csrf_endpoints %}
    {% for endpoint in csrf_endpoints %}
    # Test without CSRF token
    response = client.post("{{ endpoint.path }}", json={{"data": "test"}})
    assert response.status_code == 403  # Should be forbidden
    
    # Test with invalid CSRF token
    response = client.post("{{ endpoint.path }}", 
                          json={{"data": "test"}}, 
                          headers={"X-CSRF-Token": "invalid_token"})
    assert response.status_code == 403
    {% endfor %}
    {% endif %}


def test_rate_limiting(client):
    \"\"\"Test rate limiting for security\"\"\"
    {% if rate_limit_endpoints %}
    {% for endpoint in rate_limit_endpoints %}
    # Make multiple requests quickly
    for i in range({{ endpoint.limit }} + 10):
        response = client.get("{{ endpoint.path }}")
        if response.status_code == 429:
            break
    
    # Verify rate limiting works
    assert response.status_code == 429
    {% endfor %}
    {% endif %}


def test_secure_headers(client):
    \"\"\"Test security headers\"\"\"
    response = client.get("/")
    
    # Check for security headers
    headers = response.headers
    
    {% if security_headers %}
    {% for header, expected_value in security_headers.items() %}
    assert "{{ header }}" in headers, "Missing security header: {{ header }}"
    {% if expected_value %}
    assert headers["{{ header }}"] == "{{ expected_value }}", f"Invalid {{ header }} value"
    {% endif %}
    {% endfor %}
    {% endif %}


def test_input_validation(client):
    \"\"\"Test input validation\"\"\"
    {% if validation_endpoints %}
    {% for endpoint in validation_endpoints %}
    # Test with oversized input
    oversized_input = "A" * 10000
    response = client.post("{{ endpoint.path }}", json={{"{{ endpoint.field }}": oversized_input}})
    assert response.status_code == 400
    
    # Test with invalid data types
    invalid_inputs = {{ endpoint.invalid_inputs }}
    for invalid_input in invalid_inputs:
        response = client.post("{{ endpoint.path }}", json={{"{{ endpoint.field }}": invalid_input}})
        assert response.status_code == 400
    {% endfor %}
    {% endif %}


def test_error_message_security(client):
    \"\"\"Test error message security\"\"\"
    {% if error_endpoints %}
    {% for endpoint in error_endpoints %}
    # Test with invalid input that should cause error
    response = client.get("{{ endpoint.path }}", params={{"{{ endpoint.param }}": "invalid"}})
    
    # Error message should not expose sensitive information
    error_response = response.json()
    error_message = error_response.get("message", "")
    
    sensitive_terms = {{ endpoint.sensitive_terms }}
    for term in sensitive_terms:
        assert term.lower() not in error_message.lower(), f"Error message contains sensitive term: {term}"
    {% endfor %}
    {% endif %}
""",
                variables=[
                    TemplateVariable("app_module", "string", "Module containing the app", "main"),
                    TemplateVariable("sql_injection_endpoints", "list", "Endpoints vulnerable to SQL injection", []),
                    TemplateVariable("xss_endpoints", "list", "Endpoints vulnerable to XSS", []),
                    TemplateVariable("protected_endpoints", "list", "Endpoints requiring authentication", []),
                    TemplateVariable("authorization_endpoints", "list", "Endpoints requiring authorization", []),
                    TemplateVariable("sensitive_data_endpoints", "list", "Endpoints that might expose sensitive data", []),
                    TemplateVariable("sensitive_fields", "list", "List of sensitive field names", ["password", "credit_card", "ssn"]),
                    TemplateVariable("file_endpoints", "list", "Endpoints that handle file operations", []),
                    TemplateVariable("command_endpoints", "list", "Endpoints that execute commands", []),
                    TemplateVariable("csrf_endpoints", "list", "Endpoints that should be CSRF protected", []),
                    TemplateVariable("rate_limit_endpoints", "list", "Endpoints with rate limiting", []),
                    TemplateVariable("validation_endpoints", "list", "Endpoints with input validation", []),
                    TemplateVariable("error_endpoints", "list", "Endpoints that might expose sensitive error info", []),
                    TemplateVariable("security_headers", "dict", "Expected security headers", {
                        "X-Content-Type-Options": "nosniff",
                        "X-Frame-Options": "DENY",
                        "X-XSS-Protection": "1; mode=block",
                        "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
                    }),
                    TemplateVariable("sensitive_terms", "list", "Terms that should not appear in error messages", ["password", "database", "table", "column", "sql"])
                ],
                tags=["security", "python", "pytest", "fastapi"],
                dependencies=["pytest", "fastapi", "httpx"],
                complexity="high",
                estimated_time="15m"
            )
        
        raise NotImplementedError(f"Security template not implemented for {language}")


class TemplateManager:
    """Main template management system"""
    
    def __init__(self, template_dir: str = "benchmark/test_templates"):
        """
        Initialize template manager
        
        Args:
            template_dir: Directory containing template files
        """
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(exist_ok=True)
        
        self.templates: Dict[str, TestTemplate] = {}
        self.console = Console()
        
        # Load existing templates
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from directory"""
        self.console.print("[cyan]Loading test templates...[/cyan]")
        
        for template_file in self.template_dir.glob("*.yaml"):
            try:
                with open(template_file, 'r') as f:
                    template_data = yaml.safe_load(f)
                
                template = TestTemplate.from_dict(template_data)
                self.templates[template.name] = template
                
                self.console.print(f"  [green]✓[/green] Loaded template: {template.name}")
                
            except Exception as e:
                self.console.print(f"  [red]✗[/red] Failed to load {template_file}: {e}")
    
    def add_template(self, template: TestTemplate) -> bool:
        """Add a new template"""
        # Validate template
        is_valid, errors = TemplateValidator.validate_template(template)
        if not is_valid:
            self.console.print(f"[red]Template validation failed: {', '.join(errors)}[/red]")
            return False
        
        # Check if template already exists
        if template.name in self.templates:
            self.console.print(f"[yellow]Template '{template.name}' already exists. Use update_template to modify.[/yellow]")
            return False
        
        # Add template
        self.templates[template.name] = template
        
        # Save to file
        return self._save_template(template)
    
    def update_template(self, template_name: str, updated_template: TestTemplate) -> bool:
        """Update an existing template"""
        if template_name not in self.templates:
            self.console.print(f"[red]Template '{template_name}' not found[/red]")
            return False
        
        # Validate updated template
        is_valid, errors = TemplateValidator.validate_template(updated_template)
        if not is_valid:
            self.console.print(f"[red]Template validation failed: {', '.join(errors)}[/red]")
            return False
        
        # Update template
        self.templates[template_name] = updated_template
        updated_template.updated_at = datetime.now().isoformat()
        
        # Save to file
        return self._save_template(updated_template)
    
    def _save_template(self, template: TestTemplate) -> bool:
        """Save template to file"""
        try:
            template_file = self.template_dir / f"{template.name}.yaml"
            
            with open(template_file, 'w') as f:
                yaml.dump(template.to_dict(), f, default_flow_style=False, sort_keys=False)
            
            self.console.print(f"[green]Template '{template.name}' saved successfully[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Failed to save template '{template.name}': {e}[/red]")
            return False
    
    def get_template(self, template_name: str) -> Optional[TestTemplate]:
        """Get a template by name"""
        return self.templates.get(template_name)
    
    def list_templates(self, category: Optional[TemplateCategory] = None, 
                      language: Optional[TemplateLanguage] = None,
                      framework: Optional[TemplateFramework] = None,
                      tags: Optional[List[str]] = None) -> List[TestTemplate]:
        """List templates with optional filtering"""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        
        if language:
            templates = [t for t in templates if t.language == language]
        
        if framework:
            templates = [t for t in templates if t.framework == framework]
        
        if tags:
            templates = [t for t in templates if any(tag in t.tags for tag in tags)]
        
        return templates
    
    def search_templates(self, query: str) -> List[TestTemplate]:
        """Search templates by name or description"""
        query = query.lower()
        return [
            template for template in self.templates.values()
            if query in template.name.lower() or query in template.description.lower()
        ]
    
    def delete_template(self, template_name: str) -> bool:
        """Delete a template"""
        if template_name not in self.templates:
            self.console.print(f"[red]Template '{template_name}' not found[/red]")
            return False
        
        # Remove from memory
        del self.templates[template_name]
        
        # Remove file
        try:
            template_file = self.template_dir / f"{template_name}.yaml"
            template_file.unlink()
            
            self.console.print(f"[green]Template '{template_name}' deleted successfully[/green]")
            return True
            
        except Exception as e:
            self.console.print(f"[red]Failed to delete template file: {e}[/red]")
            return False
    
    def render_template(self, template_name: str, variables: Dict[str, Any]) -> str:
        """Render a template with variables"""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template '{template_name}' not found")
        
        return template.render(variables)
    
    def generate_template(self, category: TemplateCategory, language: TemplateLanguage) -> Optional[TestTemplate]:
        """Generate a template dynamically"""
        if category == TemplateCategory.UNIT:
            return TemplateGenerator.generate_unit_function_template(language)
        elif category == TemplateCategory.INTEGRATION:
            return TemplateGenerator.generate_integration_api_template(language)
        elif category == TemplateCategory.SECURITY:
            return TemplateGenerator.generate_security_template(language)
        else:
            self.console.print(f"[yellow]Template generation not implemented for category: {category}[/yellow]")
            return None
    
    def validate_template_variables(self, template_name: str, variables: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate variables for a template"""
        template = self.get_template(template_name)
        if not template:
            return False, [f"Template '{template_name}' not found"]
        
        return template.validate_variables(variables)
    
    def get_template_statistics(self) -> Dict[str, Any]:
        """Get template statistics"""
        if not self.templates:
            return {"total": 0}
        
        stats = {
            "total": len(self.templates),
            "by_category": defaultdict(int),
            "by_language": defaultdict(int),
            "by_framework": defaultdict(int),
            "by_complexity": defaultdict(int),
            "avg_variables": 0,
            "most_used_tags": Counter()
        }
        
        total_variables = 0
        
        for template in self.templates.values():
            stats["by_category"][template.category.value] += 1
            stats["by_language"][template.language.value] += 1
            stats["by_framework"][template.framework.value] += 1
            stats["by_complexity"][template.complexity] += 1
            
            total_variables += len(template.variables)
            
            for tag in template.tags:
                stats["most_used_tags"][tag] += 1
        
        stats["avg_variables"] = total_variables / len(self.templates)
        stats["most_used_tags"] = dict(stats["most_used_tags"].most_common(10))
        
        return stats
    
    def print_template_list(self, templates: List[TestTemplate]):
        """Print a formatted list of templates"""
        if not templates:
            self.console.print("[yellow]No templates found[/yellow]")
            return
        
        table = Table(title="Test Templates")
        table.add_column("Name", style="cyan")
        table.add_column("Category", style="magenta")
        table.add_column("Language", style="green")
        table.add_column("Framework", style="blue")
        table.add_column("Complexity", style="yellow")
        table.add_column("Variables", style="red")
        
        for template in templates:
            table.add_row(
                template.name,
                template.category.value,
                template.language.value,
                template.framework.value,
                template.complexity,
                str(len(template.variables))
            )
        
        self.console.print(table)
    
    def print_template_details(self, template_name: str):
        """Print detailed information about a template"""
        template = self.get_template(template_name)
        if not template:
            self.console.print(f"[red]Template '{template_name}' not found[/red]")
            return
        
        # Create a rich panel with template details
        details = f"""
[b]Name:[/b] {template.name}
[b]Description:[/b] {template.description}
[b]Category:[/b] {template.category.value}
[b]Language:[/b] {template.language.value}
[b]Framework:[/b] {template.framework.value}
[b]Complexity:[/b] {template.complexity}
[b]Estimated Time:[/b] {template.estimated_time}
[b]Author:[/b] {template.author}
[b]Version:[/b] {template.version}
[b]Created:[/b] {template.created_at}
[b]Updated:[/b] {template.updated_at}

[b]Tags:[/b] {', '.join(template.tags)}
[b]Dependencies:[/b] {', '.join(template.dependencies)}

[b]Variables:[/b]
"""
        
        for var in template.variables:
            details += f"  • {var.name} ({var.type}): {var.description}"
            if var.default_value is not None:
                details += f" [Default: {var.default_value}]"
            if not var.required:
                details += " [Optional]"
            details += "\n"
        
        panel = Panel(details, title=f"Template: {template.name}", border_style="blue")
        self.console.print(panel)


def main():
    """Main function for template management CLI"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Template Management System")
    parser.add_argument("--template-dir", default="benchmark/test_templates", help="Template directory")
    parser.add_argument("--list", action="store_true", help="List all templates")
    parser.add_argument("--search", help="Search templates by name or description")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--language", help="Filter by language")
    parser.add_argument("--framework", help="Filter by framework")
    parser.add_argument("--details", help="Show template details")
    parser.add_argument("--stats", action="store_true", help="Show template statistics")
    parser.add_argument("--generate-category", help="Generate template for category")
    parser.add_argument("--generate-language", help="Generate template for language")
    
    args = parser.parse_args()
    
    # Initialize template manager
    manager = TemplateManager(args.template_dir)
    
    if args.list:
        # Parse filters
        category = TemplateCategory(args.category) if args.category else None
        language = TemplateLanguage(args.language) if args.language else None
        framework = TemplateFramework(args.framework) if args.framework else None
        
        templates = manager.list_templates(category=category, language=language, framework=framework)
        manager.print_template_list(templates)
    
    elif args.search:
        templates = manager.search_templates(args.search)
        manager.print_template_list(templates)
    
    elif args.details:
        manager.print_template_details(args.details)
    
    elif args.stats:
        stats = manager.get_template_statistics()
        
        console = Console()
        console.print("[bold blue]Template Statistics[/bold blue]")
        console.print(f"Total templates: {stats['total']}")
        
        if stats['total'] > 0:
            console.print("\n[bold]By Category:[/bold]")
            for category, count in stats['by_category'].items():
                console.print(f"  {category}: {count}")
            
            console.print("\n[bold]By Language:[/bold]")
            for language, count in stats['by_language'].items():
                console.print(f"  {language}: {count}")
            
            console.print("\n[bold]By Framework:[/bold]")
            for framework, count in stats['by_framework'].items():
                console.print(f"  {framework}: {count}")
            
            console.print("\n[bold]By Complexity:[/bold]")
            for complexity, count in stats['by_complexity'].items():
                console.print(f"  {complexity}: {count}")
            
            console.print(f"\nAverage variables per template: {stats['avg_variables']:.1f}")
            
            console.print("\n[bold]Most Used Tags:[/bold]")
            for tag, count in stats['most_used_tags'].items():
                console.print(f"  {tag}: {count}")
    
    elif args.generate_category and args.generate_language:
        try:
            category = TemplateCategory(args.generate_category)
            language = TemplateLanguage(args.generate_language)
            
            template = manager.generate_template(category, language)
            if template:
                success = manager.add_template(template)
                if success:
                    console.print(f"[green]Generated and added template: {template.name}[/green]")
                else:
                    console.print(f"[red]Failed to add generated template[/red]")
        except ValueError as e:
            console.print(f"[red]Invalid category or language: {e}[/red]")
    
    else:
        console = Console()
        console.print("[yellow]No action specified. Use --help for usage information.[/yellow]")


if __name__ == "__main__":
    main()
