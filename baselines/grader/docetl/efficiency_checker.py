#!/usr/bin/env python3
"""
Efficiency checker for DocETL pipelines.
Performs rule-based performance analysis to identify optimization opportunities.

Usage:
    Command line:
        python3 efficiency_checker.py <pipeline.yaml>
    
    Python API:
        from efficiency_checker import check_pipeline_efficiency_file, check_pipeline_efficiency_string
        
        # Check from file path
        result = check_pipeline_efficiency_file("pipeline.yaml")
        
        # Check from YAML string
        yaml_content = "default_model: gpt-4o-mini\n..."
        result = check_pipeline_efficiency_string(yaml_content)
        
    Returns:
        dict: {"warnings": [...], "recommendations": [...]}
"""

import sys
import json
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import yaml
except ImportError:
    yaml = None


class SeverityLevel(Enum):
    """Severity levels for efficiency warnings."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EfficiencyWarning:
    """Represents a performance efficiency warning."""
    rule_id: str
    rule_name: str
    severity: SeverityLevel
    message: str
    step_index: Optional[int] = None
    operation_names: Optional[List[str]] = None
    recommendation: Optional[str] = None
    impact: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "message": self.message,
            "step_index": self.step_index,
            "operation_names": self.operation_names,
            "recommendation": self.recommendation,
            "impact": self.impact
        }


class EfficiencyRule(ABC):
    """Abstract base class for efficiency rules."""
    
    def __init__(self, rule_id: str, name: str, description: str, severity: SeverityLevel):
        self.rule_id = rule_id
        self.name = name
        self.description = description
        self.severity = severity
    
    @abstractmethod
    def check(self, pipeline: Dict[str, Any]) -> List[EfficiencyWarning]:
        """
        Check the pipeline for efficiency violations.
        
        Args:
            pipeline: Parsed pipeline dictionary
            
        Returns:
            List of efficiency warnings
        """
        pass


class PredicatePushdownRule(EfficiencyRule):
    """Rule: Filters should be applied before expensive operations like joins/resolve."""
    
    def __init__(self):
        super().__init__(
            rule_id="E001",
            name="Predicate Pushdown",
            description="Filter operations should be placed before expensive operations",
            severity=SeverityLevel.HIGH
        )
    
    def check(self, pipeline: Dict[str, Any]) -> List[EfficiencyWarning]:
        warnings = []
        
        if 'pipeline' not in pipeline or 'steps' not in pipeline['pipeline']:
            return warnings
        
        expensive_ops = {'resolve', 'equijoin', 'rank', 'order', 'cluster'}
        
        for step_idx, step in enumerate(pipeline['pipeline']['steps']):
            if 'operations' not in step:
                continue
                
            operations = step['operations']
            if len(operations) < 2:
                continue
            
            # Find sequences of expensive_op followed by filter
            for i in range(len(operations) - 1):
                current_op = self._get_operation_type(operations[i], pipeline.get('operations', []))
                next_op = self._get_operation_type(operations[i + 1], pipeline.get('operations', []))
                
                if current_op in expensive_ops and next_op == 'filter':
                    warnings.append(EfficiencyWarning(
                        rule_id=self.rule_id,
                        rule_name=self.name,
                        severity=self.severity,
                        message=f"Filter operation after {current_op} operation - consider moving filter earlier",
                        step_index=step_idx,
                        operation_names=[operations[i], operations[i + 1]] if isinstance(operations[i], str) else None,
                        recommendation=f"Move filter operation before {current_op} to reduce data size early",
                        impact="High - can significantly reduce processing time and cost"
                    ))
        
        return warnings
    
    def _get_operation_type(self, op_ref: Any, operations: List[Dict]) -> Optional[str]:
        """Get the operation type from operation reference."""
        if isinstance(op_ref, str):
            # Find operation definition
            for op in operations:
                if op.get('name') == op_ref:
                    return op.get('type')
        elif isinstance(op_ref, dict):
            # Inline operation
            return list(op_ref.keys())[0] if op_ref else None
        return None


class EagerSamplingRule(EfficiencyRule):
    """Rule: Sampling should be done early to reduce dataset size."""
    
    def __init__(self):
        super().__init__(
            rule_id="E002",
            name="Eager Sampling",
            description="Sampling operations should be placed early in the pipeline",
            severity=SeverityLevel.MEDIUM
        )
    
    def check(self, pipeline: Dict[str, Any]) -> List[EfficiencyWarning]:
        warnings = []
        
        if 'pipeline' not in pipeline or 'steps' not in pipeline['pipeline']:
            return warnings
        
        expensive_ops = {'map', 'filter', 'reduce', 'resolve', 'equijoin', 'rank', 'order', 'cluster', 'extract'}
        
        for step_idx, step in enumerate(pipeline['pipeline']['steps']):
            if 'operations' not in step:
                continue
                
            operations = step['operations']
            
            # Find sample operations that come after expensive operations
            sample_indices = []
            expensive_before_sample = False
            
            for i, op_ref in enumerate(operations):
                op_type = self._get_operation_type(op_ref, pipeline.get('operations', []))
                
                if op_type == 'sample':
                    sample_indices.append(i)
                elif op_type in expensive_ops and not sample_indices:
                    expensive_before_sample = True
            
            # If we have expensive operations before any sampling
            if expensive_before_sample and sample_indices:
                warnings.append(EfficiencyWarning(
                    rule_id=self.rule_id,
                    rule_name=self.name,
                    severity=self.severity,
                    message="Sample operation found after expensive operations",
                    step_index=step_idx,
                    operation_names=[operations[idx] for idx in sample_indices] if isinstance(operations[0], str) else None,
                    recommendation="Consider moving sample operation to the beginning of the step",
                    impact="Medium - can reduce processing time on large datasets"
                ))
        
        return warnings
    
    def _get_operation_type(self, op_ref: Any, operations: List[Dict]) -> Optional[str]:
        """Get the operation type from operation reference."""
        if isinstance(op_ref, str):
            # Find operation definition
            for op in operations:
                if op.get('name') == op_ref:
                    return op.get('type')
        elif isinstance(op_ref, dict):
            # Inline operation
            return list(op_ref.keys())[0] if op_ref else None
        return None



class BatchSizeOptimizationRule(EfficiencyRule):
    """Rule: Operations should use appropriate batch sizes for efficiency."""
    
    def __init__(self):
        super().__init__(
            rule_id="E004",
            name="Batch Size Optimization",
            description="Operations should use appropriate batch sizes",
            severity=SeverityLevel.MEDIUM
        )
    
    def check(self, pipeline: Dict[str, Any]) -> List[EfficiencyWarning]:
        warnings = []
        
        if 'operations' not in pipeline:
            return warnings
        
        batch_ops = {'map', 'filter', 'reduce', 'resolve', 'equijoin', 'rank', 'order'}
        
        for op in pipeline['operations']:
            op_type = op.get('type')
            if op_type not in batch_ops:
                continue
            
            # Check for very small batch sizes
            batch_size = op.get('batch_size', op.get('compare_batch_size', op.get('embedding_batch_size')))
            if batch_size is not None and batch_size < 5:
                warnings.append(EfficiencyWarning(
                    rule_id=self.rule_id,
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Operation '{op['name']}' has very small batch_size ({batch_size})",
                    operation_names=[op['name']],
                    recommendation="Consider increasing batch_size to 10-50 for better throughput",
                    impact="Medium - can improve API utilization and reduce latency"
                ))
            
            # Check for missing batch configuration on expensive operations
            if (op_type in {'resolve', 'equijoin'} and 
                'compare_batch_size' not in op and 
                'embedding_batch_size' not in op):
                warnings.append(EfficiencyWarning(
                    rule_id=self.rule_id,
                    rule_name=self.name,
                    severity=SeverityLevel.LOW,
                    message=f"Operation '{op['name']}' missing batch size configuration",
                    operation_names=[op['name']],
                    recommendation="Consider adding compare_batch_size and embedding_batch_size parameters",
                    impact="Low - can improve processing efficiency"
                ))
        
        return warnings


class RedundantClusteringRule(EfficiencyRule):
    """Rule: Avoid redundant clustering operations on the same keys."""
    
    def __init__(self):
        super().__init__(
            rule_id="E005",
            name="Redundant Clustering",
            description="Avoid redundant clustering operations on the same embedding keys",
            severity=SeverityLevel.MEDIUM
        )
    
    def check(self, pipeline: Dict[str, Any]) -> List[EfficiencyWarning]:
        warnings = []
        
        if 'pipeline' not in pipeline or 'steps' not in pipeline['pipeline']:
            return warnings
        
        for step_idx, step in enumerate(pipeline['pipeline']['steps']):
            if 'operations' not in step:
                continue
            
            # Track clustering operations and their embedding keys
            cluster_ops = []
            
            for op_ref in step['operations']:
                op_def = self._get_operation_definition(op_ref, pipeline.get('operations', []))
                if op_def and op_def.get('type') == 'cluster':
                    embedding_keys = op_def.get('embedding_keys', [])
                    cluster_ops.append((op_def['name'], set(embedding_keys)))
            
            # Check for redundant clustering
            for i in range(len(cluster_ops)):
                for j in range(i + 1, len(cluster_ops)):
                    name1, keys1 = cluster_ops[i]
                    name2, keys2 = cluster_ops[j]
                    
                    if keys1.intersection(keys2):  # Overlapping keys
                        warnings.append(EfficiencyWarning(
                            rule_id=self.rule_id,
                            rule_name=self.name,
                            severity=self.severity,
                            message=f"Cluster operations '{name1}' and '{name2}' have overlapping embedding keys",
                            step_index=step_idx,
                            operation_names=[name1, name2],
                            recommendation="Consider combining clustering operations or using different embedding keys",
                            impact="Medium - can reduce embedding computation costs"
                        ))
        
        return warnings
    
    def _get_operation_definition(self, op_ref: Any, operations: List[Dict]) -> Optional[Dict]:
        """Get the operation definition from operation reference."""
        if isinstance(op_ref, str):
            # Find operation definition
            for op in operations:
                if op.get('name') == op_ref:
                    return op
        elif isinstance(op_ref, dict):
            # Inline operation - return the operation dict itself
            return op_ref
        return None


class LargeDatasetSplitRule(EfficiencyRule):
    """Rule: Large documents should be split before expensive operations."""
    
    def __init__(self):
        super().__init__(
            rule_id="E006",
            name="Large Dataset Split",
            description="Large documents should be split before expensive LLM operations",
            severity=SeverityLevel.HIGH
        )
    
    def check(self, pipeline: Dict[str, Any]) -> List[EfficiencyWarning]:
        warnings = []
        
        if 'pipeline' not in pipeline or 'steps' not in pipeline['pipeline']:
            return warnings
        
        llm_ops = {'map', 'filter', 'reduce', 'extract', 'resolve', 'equijoin'}
        
        for step_idx, step in enumerate(pipeline['pipeline']['steps']):
            if 'operations' not in step:
                continue
            
            operations = step['operations']
            has_split = False
            has_expensive_llm_op = False
            
            for op_ref in operations:
                op_type = self._get_operation_type(op_ref, pipeline.get('operations', []))
                
                if op_type == 'split':
                    has_split = True
                elif op_type in llm_ops:
                    has_expensive_llm_op = True
            
            # If we have expensive LLM operations but no split, warn about potential efficiency
            if has_expensive_llm_op and not has_split:
                # Check if there's any indication of large documents
                input_source = step.get('input')
                if input_source:
                    warnings.append(EfficiencyWarning(
                        rule_id=self.rule_id,
                        rule_name=self.name,
                        severity=self.severity,
                        message="Expensive LLM operations without document splitting",
                        step_index=step_idx,
                        recommendation="Consider adding a split operation before LLM operations for large documents",
                        impact="High - can reduce token usage and improve processing efficiency"
                    ))
        
        return warnings
    
    def _get_operation_type(self, op_ref: Any, operations: List[Dict]) -> Optional[str]:
        """Get the operation type from operation reference."""
        if isinstance(op_ref, str):
            # Find operation definition
            for op in operations:
                if op.get('name') == op_ref:
                    return op.get('type')
        elif isinstance(op_ref, dict):
            # Inline operation
            return list(op_ref.keys())[0] if op_ref else None
        return None


class MissingBlockingRule(EfficiencyRule):
    """Rule: Resolve and equijoin operations should have blocking configurations for efficiency."""
    
    def __init__(self):
        super().__init__(
            rule_id="E007",
            name="Missing Blocking Configuration",
            description="Resolve and equijoin operations should use blocking to reduce comparisons",
            severity=SeverityLevel.CRITICAL
        )
    
    def check(self, pipeline: Dict[str, Any]) -> List[EfficiencyWarning]:
        warnings = []
        
        if 'operations' not in pipeline:
            return warnings
        
        blocking_operations = {'resolve', 'equijoin'}
        
        for op in pipeline['operations']:
            op_type = op.get('type')
            if op_type not in blocking_operations:
                continue
            
            op_name = op.get('name', 'unnamed')
            
            # Check for blocking configurations
            has_blocking_keys = op.get('blocking_keys') is not None
            has_blocking_threshold = op.get('blocking_threshold') is not None
            has_blocking_conditions = op.get('blocking_conditions') is not None and len(op.get('blocking_conditions', [])) > 0
            
            if not (has_blocking_keys or has_blocking_threshold or has_blocking_conditions):
                # Estimate potential performance impact
                impact_msg = self._estimate_blocking_impact(op_type)
                
                warnings.append(EfficiencyWarning(
                    rule_id=self.rule_id,
                    rule_name=self.name,
                    severity=self.severity,
                    message=f"Operation '{op_name}' ({op_type}) has no blocking configuration",
                    operation_names=[op_name],
                    recommendation=self._get_blocking_recommendation(op_type),
                    impact=impact_msg
                ))
            else:
                # Check for suboptimal blocking configurations
                if has_blocking_threshold and op.get('blocking_threshold', 0) < 0.7:
                    warnings.append(EfficiencyWarning(
                        rule_id=self.rule_id,
                        rule_name=self.name,
                        severity=SeverityLevel.MEDIUM,
                        message=f"Operation '{op_name}' has low blocking_threshold ({op.get('blocking_threshold')})",
                        operation_names=[op_name],
                        recommendation="Consider increasing blocking_threshold to 0.7-0.9 for better filtering",
                        impact="Medium - may allow too many false positive comparisons"
                    ))
        
        return warnings
    
    def _estimate_blocking_impact(self, op_type: str) -> str:
        """Estimate the performance impact based on operation type."""
        if op_type == 'resolve':
            return "Critical - without blocking, resolve operations perform O(n²) comparisons, leading to exponential cost growth with dataset size"
        elif op_type == 'equijoin':
            return "Critical - without blocking, equijoin operations compare every left item with every right item, leading to O(n*m) comparisons"
        return "Critical - blocking is essential for large-scale comparison operations"
    
    def _get_blocking_recommendation(self, op_type: str) -> str:
        """Get specific blocking recommendations based on operation type."""
        if op_type == 'resolve':
            return ("Add blocking configuration: 1) blocking_keys for exact key matching, "
                   "2) blocking_threshold (0.7-0.9) for embedding similarity, "
                   "3) blocking_conditions for custom logic")
        elif op_type == 'equijoin':
            return ("Add blocking configuration: 1) blocking_keys with 'left' and 'right' key mappings, "
                   "2) blocking_threshold for embedding-based filtering, "
                   "3) blocking_conditions for custom join logic")
        return "Add appropriate blocking configuration to reduce comparison overhead"


class DocETLEfficiencyChecker:
    """Main efficiency checker for DocETL pipelines."""
    
    def __init__(self):
        self.rules: List[EfficiencyRule] = []
        self._register_default_rules()
    
    def _register_default_rules(self):
        """Register all default efficiency rules."""
        self.rules.extend([
            PredicatePushdownRule(),
            EagerSamplingRule(),
            BatchSizeOptimizationRule(),
            RedundantClusteringRule(),
            LargeDatasetSplitRule(),
            MissingBlockingRule(),
        ])
    
    def add_rule(self, rule: EfficiencyRule):
        """Add a custom efficiency rule."""
        self.rules.append(rule)
    
    def remove_rule(self, rule_id: str):
        """Remove a rule by its ID."""
        self.rules = [rule for rule in self.rules if rule.rule_id != rule_id]
    
    def get_rules(self) -> List[EfficiencyRule]:
        """Get all registered rules."""
        return self.rules.copy()
    
    def check(self, pipeline_path: str) -> Dict[str, Any]:
        """
        Check pipeline efficiency from file path.
        
        Args:
            pipeline_path: Path to pipeline YAML file
            
        Returns:
            Dictionary with warnings and recommendations
        """
        try:
            with open(pipeline_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.check_string(content)
        except FileNotFoundError:
            return {
                "error": f"File not found: {pipeline_path}",
                "warnings": [],
                "recommendations": []
            }
        except Exception as e:
            return {
                "error": f"Error reading file: {str(e)}",
                "warnings": [],
                "recommendations": []
            }
    
    def check_string(self, yaml_content: str) -> Dict[str, Any]:
        """
        Check pipeline efficiency from YAML string.
        
        Args:
            yaml_content: YAML content as string
            
        Returns:
            Dictionary with warnings and recommendations
        """
        try:
            # Parse YAML
            pipeline = self._parse_yaml(yaml_content)
            
            # Collect all warnings
            all_warnings = []
            for rule in self.rules:
                try:
                    warnings = rule.check(pipeline)
                    all_warnings.extend(warnings)
                except Exception as e:
                    # Don't let one rule failure break the entire check
                    print(f"Warning: Rule {rule.rule_id} failed: {str(e)}", file=sys.stderr)
                    continue
            
            # Generate recommendations based on warnings
            recommendations = self._generate_recommendations(all_warnings)
            
            return {
                "warnings": [warning.to_dict() for warning in all_warnings],
                "recommendations": recommendations,
                "summary": {
                    "total_warnings": len(all_warnings),
                    "critical": len([w for w in all_warnings if w.severity == SeverityLevel.CRITICAL]),
                    "high": len([w for w in all_warnings if w.severity == SeverityLevel.HIGH]),
                    "medium": len([w for w in all_warnings if w.severity == SeverityLevel.MEDIUM]),
                    "low": len([w for w in all_warnings if w.severity == SeverityLevel.LOW])
                }
            }
            
        except Exception as e:
            return {
                "error": f"Error parsing pipeline: {str(e)}",
                "warnings": [],
                "recommendations": []
            }
    
    def _parse_yaml(self, content: str) -> Dict[str, Any]:
        """Parse YAML content."""
        if yaml:
            return yaml.safe_load(content)
        else:
            # Fallback simple parser (limited functionality)
            raise ImportError("PyYAML not available. Please install PyYAML for full functionality.")
    
    def _generate_recommendations(self, warnings: List[EfficiencyWarning]) -> List[Dict[str, Any]]:
        """Generate high-level recommendations based on warnings."""
        recommendations = []
        
        # Group warnings by severity
        severity_counts = {}
        for warning in warnings:
            severity_counts[warning.severity.value] = severity_counts.get(warning.severity.value, 0) + 1
        
        if severity_counts.get('critical', 0) > 0:
            recommendations.append({
                "priority": "critical",
                "message": f"Address {severity_counts['critical']} critical efficiency issues immediately",
                "actions": ["Add blocking configurations to resolve/equijoin operations", "Review pipeline structure", "Consider major refactoring"]
            })
        
        if severity_counts.get('high', 0) > 0:
            recommendations.append({
                "priority": "high",
                "message": f"Optimize {severity_counts['high']} high-impact efficiency issues",
                "actions": ["Reorder operations", "Add filtering/sampling early", "Optimize batch sizes"]
            })
        
        if severity_counts.get('medium', 0) > 2:
            recommendations.append({
                "priority": "medium",
                "message": f"Consider addressing {severity_counts['medium']} medium-priority optimizations",
                "actions": ["Review batch configurations", "Eliminate redundant operations"]
            })
        
        if not warnings:
            recommendations.append({
                "priority": "info",
                "message": "Pipeline appears well-optimized from an efficiency perspective",
                "actions": ["Monitor performance in production", "Consider A/B testing different configurations"]
            })
        
        return recommendations


def check_pipeline_efficiency_file(pipeline_path: str) -> Dict[str, Any]:
    """
    Check pipeline efficiency from file path.
    
    Args:
        pipeline_path: Path to pipeline YAML file
        
    Returns:
        Dictionary with efficiency analysis results
    """
    checker = DocETLEfficiencyChecker()
    return checker.check(pipeline_path)


def check_pipeline_efficiency_string(yaml_content: str) -> Dict[str, Any]:
    """
    Check pipeline efficiency from YAML string.
    
    Args:
        yaml_content: YAML content as string
        
    Returns:
        Dictionary with efficiency analysis results
    """
    checker = DocETLEfficiencyChecker()
    return checker.check_string(yaml_content)


def main():
    """Command-line interface for the efficiency checker."""
    if len(sys.argv) != 2:
        print(json.dumps({
            "error": "Usage: python efficiency_checker.py <pipeline.yaml>",
            "warnings": [],
            "recommendations": []
        }))
        sys.exit(1)
    
    pipeline_path = sys.argv[1]
    result = check_pipeline_efficiency_file(pipeline_path)
    
    # Print the result as JSON
    print(json.dumps(result, indent=2))
    
    # Exit with non-zero code if there are high or critical warnings
    if result.get('summary', {}).get('critical', 0) > 0 or result.get('summary', {}).get('high', 0) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()