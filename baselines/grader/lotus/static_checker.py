#!/usr/bin/env python3
"""
Static checker for LOTUS pipelines.

Validates LOTUS pipeline Python code for syntax errors, operator usage, and DataFrame consistency.

Usage:
    from static_checker import check_lotus_pipeline
    
    # Check from file
    result = check_lotus_pipeline(pipeline_path="pipeline.py")
    
    # Check from string
    result = check_lotus_pipeline(pipeline_code=code_string)
"""

import ast
import sys
from typing import Dict, Any


# LOTUS semantic operators
LOTUS_OPERATORS = {
    'sem_filter', 'sem_map', 'sem_extract', 'sem_agg', 
    'sem_join', 'sem_topk', 'sem_sim_join', 'sem_search',
    'sem_partition_by', 'sem_index', 'sem_dedup'
}

# Required imports for LOTUS
REQUIRED_IMPORTS = {
    'lotus': ['lotus'],
    'models': ['LM', 'SentenceTransformersRM'],
    'vector_store': ['FaissVS'],
    'pandas': ['pd', 'pandas']
}


class LotusStaticChecker:
    """Static checker for LOTUS pipeline Python code."""
    
    def __init__(self):
        """Initialize the checker."""
        self.errors = []
        self.warnings = []
        self.dataframe_vars = set()  # Track DataFrame variables
        self.imported_modules = set()  # Track imported modules
        self.configured_settings = False  # Track if lotus.settings.configure was called
        
    def check(self, pipeline_code: str = None, pipeline_path: str = None) -> Dict[str, Any]:
        """
        Check LOTUS pipeline for static errors.
        
        Args:
            pipeline_code: Python code string (optional)
            pipeline_path: Path to Python file (optional)
            
        Returns:
            Dict containing is_valid flag and error details
        """
        # Reset state
        self.errors = []
        self.warnings = []
        self.dataframe_vars = set()
        self.imported_modules = set()
        self.configured_settings = False
        
        # Load pipeline code
        if pipeline_code is not None:
            code = pipeline_code
        elif pipeline_path is not None:
            try:
                with open(pipeline_path, 'r', encoding='utf-8') as f:
                    code = f.read()
            except FileNotFoundError:
                return {
                    "is_valid": False,
                    "errors": [{"type": "FILE_ERROR", "message": f"File not found: {pipeline_path}"}],
                    "warnings": []
                }
            except Exception as e:
                return {
                    "is_valid": False,
                    "errors": [{"type": "FILE_ERROR", "message": f"Error reading file: {str(e)}"}],
                    "warnings": []
                }
        else:
            return {
                "is_valid": False,
                "errors": [{"type": "INPUT_ERROR", "message": "Either pipeline_code or pipeline_path must be provided"}],
                "warnings": []
            }
        
        # Run checks
        self._check_python_syntax(code)
        if not self.errors:  # Only continue if syntax is valid
            self._check_imports(code)
            self._check_lotus_configuration(code)
            self._check_operator_usage(code)
            self._check_dataframe_consistency(code)
        
        return {
            "is_valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings
        }
    
    def _check_python_syntax(self, code: str):
        """Check if the Python code has valid syntax."""
        try:
            ast.parse(code)
        except SyntaxError as e:
            self.errors.append({
                "type": "SYNTAX_ERROR",
                "line": e.lineno,
                "message": f"Python syntax error: {e.msg}"
            })
    
    def _check_imports(self, code: str):
        """Check if required LOTUS imports are present."""
        try:
            tree = ast.parse(code)
        except:
            return  # Syntax error already reported
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imported_modules.add(alias.name)
                    if alias.asname:
                        self.imported_modules.add(alias.asname)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.imported_modules.add(node.module)
                    for alias in node.names:
                        self.imported_modules.add(alias.name)
                        if alias.asname:
                            self.imported_modules.add(alias.asname)
        
        # Check for required imports
        has_lotus = any('lotus' in mod for mod in self.imported_modules)
        has_pandas = any(mod in self.imported_modules for mod in ['pandas', 'pd'])
        
        if not has_lotus:
            self.errors.append({
                "type": "IMPORT_ERROR",
                "message": "Missing required import: lotus"
            })
        
        if not has_pandas:
            self.warnings.append({
                "type": "IMPORT_WARNING",
                "message": "Missing recommended import: pandas"
            })
    
    def _check_lotus_configuration(self, code: str):
        """Check if LOTUS settings are properly configured."""
        try:
            tree = ast.parse(code)
        except:
            return
        
        # Look for lotus.settings.configure() call
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Attribute) and
                        hasattr(node.func.value, 'value') and
                        hasattr(node.func.value.value, 'id') and
                        node.func.value.value.id == 'lotus' and
                        node.func.value.attr == 'settings' and
                        node.func.attr == 'configure'):
                        self.configured_settings = True
                        break
        
        if not self.configured_settings:
            self.errors.append({
                "type": "CONFIGURATION_ERROR",
                "message": "LOTUS settings not configured. Call lotus.settings.configure(lm=..., rm=..., vs=...)"
            })
    
    def _check_operator_usage(self, code: str):
        """Check if LOTUS operators are used correctly."""
        try:
            tree = ast.parse(code)
        except:
            return
        
        class OperatorVisitor(ast.NodeVisitor):
            def __init__(self, checker):
                self.checker = checker
                
            def visit_Call(self, node):
                # Check for LOTUS operator calls
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in LOTUS_OPERATORS:
                        # Check if called on a DataFrame-like object
                        if isinstance(node.func.value, ast.Name):
                            var_name = node.func.value.id
                            # Track this as a DataFrame variable
                            self.checker.dataframe_vars.add(var_name)
                            
                        # Validate operator-specific requirements
                        self._validate_operator_args(node.func.attr, node)
                
                self.generic_visit(node)

            def _validate_operator_args(self, op_name: str, node):
                """Validate arguments for specific operators against LOTUS docs."""
                def has_kw(*names):
                    return any(kw.arg in names for kw in node.keywords)

                def has_pos(i: int):
                    return len(node.args) > i

                def present(name: str, pos_index: int | None = None, alt_names: list[str] | None = None):
                    names = {name}
                    if alt_names:
                        names.update(alt_names)
                    return (pos_index is not None and has_pos(pos_index)) or has_kw(*names)

                def require(params: list[tuple[str, int | None, list[str] | None]], msg_hint: str):
                    missing = []
                    for name, pos_index, alt in params:
                        if not present(name, pos_index, alt):
                            missing.append(name if not alt else f"{name} ({'/'.join([name]+alt)})")
                    if missing:
                        self.checker.errors.append({
                            "type": "OPERATOR_ERROR",
                            "line": getattr(node, "lineno", None),
                            "message": f"{op_name} requires {', '.join(missing)}{msg_hint}"
                        })

                # Validate by ops
                if op_name == 'sem_map':
                    # Need: user_instruction
                    require([('user_instruction', 0, None)], " argument")
                elif op_name == 'sem_filter':
                    # Need: user_instruction
                    require([('user_instruction', 0, None)], " argument")
                elif op_name == 'sem_extract':
                    # Need: input_cols, output_cols
                    require([('input_cols', 0, None), ('output_cols', 1, None)], " arguments")
                elif op_name == 'sem_agg':
                    # Need: user_instructions
                    require([('user_instructions', 0, ['user_instruction'])], " argument")
                elif op_name == 'sem_join':
                    # Need: other, join_instruction
                    require([('other', 0, None), ('join_instruction', 1, None)], " arguments")
                elif op_name == 'sem_topk':
                    # Need: user_instruction, K
                    require([('user_instruction', 0, None), ('K', 1, None)], " arguments")
                elif op_name == 'sem_sim_join':
                    # Need: other, left_on, right_on, K
                    require([('other', 0, None), ('left_on', None, None), ('right_on', None, None), ('K', None, None)], " arguments")
                elif op_name == 'sem_search':
                    # Need: col_name, query
                    require([('col_name', 0, None), ('query', 1, None)], " arguments")
                elif op_name == 'sem_partition_by':
                    # Need: partition_fn
                    require([('partition_fn', 0, None)], " argument")
                elif op_name == 'sem_index':
                    # Need: col_name, index_dir
                    require([('col_name', 0, None), ('index_dir', 1, None)], " arguments")
                elif op_name == 'sem_dedup':
                    # Need: col_name, threshold
                    require([('col_name', 0, None), ('threshold', None, None)], " arguments")
                """Validate arguments for specific operators."""
                # Check for required arguments based on operator
                if op_name == 'sem_filter':
                    if len(node.args) < 1 and not any(kw.arg == 'user_instruction' for kw in node.keywords):
                        self.checker.errors.append({
                            "type": "OPERATOR_ERROR",
                            "line": node.lineno,
                            "message": f"{op_name} requires a user_instruction argument"
                        })
                        
                elif op_name == 'sem_map':
                    if len(node.args) < 1 and not any(kw.arg == 'user_instruction' for kw in node.keywords):
                        self.checker.errors.append({
                            "type": "OPERATOR_ERROR",
                            "line": node.lineno,
                            "message": f"{op_name} requires a user_instruction argument"
                        })
                        
                elif op_name == 'sem_join':
                    if len(node.args) < 2:
                        self.checker.errors.append({
                            "type": "OPERATOR_ERROR",
                            "line": node.lineno,
                            "message": f"{op_name} requires at least two arguments (other_df and join_instruction)"
                        })
        
        visitor = OperatorVisitor(self)
        visitor.visit(tree)
    
    def _check_dataframe_consistency(self, code: str):
        """Check DataFrame variable usage and consistency."""
        try:
            tree = ast.parse(code)
        except:
            return
        
        # Track DataFrame assignments and usage
        class DataFrameVisitor(ast.NodeVisitor):
            def __init__(self, checker):
                self.checker = checker
                self.assigned_vars = set()
                
            def visit_Assign(self, node):
                # Track DataFrame assignments
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        # Check if RHS is a DataFrame operation
                        if self._is_dataframe_operation(node.value):
                            self.assigned_vars.add(target.id)
                            self.checker.dataframe_vars.add(target.id)
                            
                self.generic_visit(node)
            
            def _is_dataframe_operation(self, node):
                """Check if a node represents a DataFrame operation."""
                if isinstance(node, ast.Call):
                    # Check for pd.DataFrame()
                    if (isinstance(node.func, ast.Attribute) and
                        node.func.attr == 'DataFrame'):
                        return True
                    # Check for DataFrame methods
                    if (isinstance(node.func, ast.Attribute) and
                        isinstance(node.func.value, ast.Name) and
                        node.func.value.id in self.checker.dataframe_vars):
                        return True
                    # Check for LOTUS operators
                    if (isinstance(node.func, ast.Attribute) and
                        node.func.attr in LOTUS_OPERATORS):
                        return True
                    # Check for pd.read_csv() etc
                    if (isinstance(node.func, ast.Attribute) and
                        hasattr(node.func, 'value') and
                        isinstance(node.func.value, ast.Name) and
                        node.func.value.id == 'pd' and
                        node.func.attr.startswith('read_')):
                        return True
                return False
        
        visitor = DataFrameVisitor(self)
        visitor.visit(tree)
        
        # Check for undefined DataFrame usage
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id.endswith('_df') and node.id not in visitor.assigned_vars:
                    if node.id not in ['pd', 'np']:  # Exclude common module names
                        self.warnings.append({
                            "type": "DATAFRAME_WARNING",
                            "line": node.lineno if hasattr(node, 'lineno') else 0,
                            "message": f"Possible undefined DataFrame: {node.id}"
                        })


def check_lotus_pipeline(pipeline_code: str = None, pipeline_path: str = None) -> Dict[str, Any]:
    """
    Convenience function to check LOTUS pipeline.
    
    Args:
        pipeline_code: Python code string (optional)
        pipeline_path: Path to Python file (optional)
        
    Returns:
        Dict containing validation results
    """
    checker = LotusStaticChecker()
    return checker.check(pipeline_code, pipeline_path)


def main():
    """Command-line interface for the static checker."""
    if len(sys.argv) < 2:
        print("Usage: python3 static_checker.py <pipeline.py>")
        sys.exit(1)
    
    pipeline_path = sys.argv[1]
    result = check_lotus_pipeline(pipeline_path=pipeline_path)
    
    if result["is_valid"]:
        print("✅ Pipeline is valid")
        sys.exit(0)
    else:
        print("❌ Pipeline has errors:")
        for error in result["errors"]:
            print(f"  - {error['type']}: {error['message']}")
            if 'line' in error:
                print(f"    Line: {error['line']}")
        
        if result["warnings"]:
            print("\n⚠️  Warnings:")
            for warning in result["warnings"]:
                print(f"  - {warning['type']}: {warning['message']}")
        
        sys.exit(1)


if __name__ == "__main__":
    main()