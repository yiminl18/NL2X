"""
Path Management APIs for DocETL Pipeline Configuration

This module provides utilities to identify and modify dataset paths and output paths
in DocETL pipeline configurations. These APIs are used by the data management layer
to redirect pipelines to use processed datasets and custom output locations.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import copy


def get_dataset_paths(pipeline_config: Dict[str, Any]) -> Dict[str, str]:
    """Extract all dataset paths from pipeline configuration."""
    dataset_paths = {}

    datasets = pipeline_config.get('datasets', {})
    for dataset_name, dataset_config in datasets.items():
        if isinstance(dataset_config, dict) and 'path' in dataset_config:
            dataset_paths[dataset_name] = dataset_config['path']

    return dataset_paths


def set_dataset_paths(
    pipeline_config: Dict[str, Any],
    path_mapping: Dict[str, str],
    in_place: bool = False
) -> Dict[str, Any]:
    """Update dataset paths in pipeline configuration."""
    if not in_place:
        pipeline_config = copy.deepcopy(pipeline_config)

    datasets = pipeline_config.get('datasets', {})

    for dataset_name, new_path in path_mapping.items():
        if dataset_name in datasets:
            if isinstance(datasets[dataset_name], dict):
                datasets[dataset_name]['path'] = new_path

    return pipeline_config


def set_dataset_path(
    pipeline_config: Dict[str, Any],
    dataset_name: str,
    new_path: Union[str, Path],
    in_place: bool = False
) -> Dict[str, Any]:
    """Update a single dataset path in pipeline configuration."""
    return set_dataset_paths(
        pipeline_config,
        {dataset_name: str(new_path)},
        in_place=in_place
    )


def get_output_path(pipeline_config: Dict[str, Any]) -> Optional[str]:
    """Extract the output path from pipeline configuration."""
    pipeline = pipeline_config.get('pipeline', {})
    output = pipeline.get('output', {})

    if isinstance(output, dict) and 'path' in output:
        return output['path']

    return None


def set_output_path(
    pipeline_config: Dict[str, Any],
    new_path: Union[str, Path],
    in_place: bool = False
) -> Dict[str, Any]:
    """Update the output path in pipeline configuration."""
    if not in_place:
        pipeline_config = copy.deepcopy(pipeline_config)

    # Ensure pipeline structure exists
    if 'pipeline' not in pipeline_config:
        pipeline_config['pipeline'] = {}

    if 'output' not in pipeline_config['pipeline']:
        pipeline_config['pipeline']['output'] = {'type': 'file'}

    pipeline_config['pipeline']['output']['path'] = str(new_path)

    return pipeline_config


def get_intermediate_dir(pipeline_config: Dict[str, Any]) -> Optional[str]:
    """Extract the intermediate directory from pipeline configuration."""
    pipeline = pipeline_config.get('pipeline', {})
    output = pipeline.get('output', {})

    if isinstance(output, dict) and 'intermediate_dir' in output:
        return output['intermediate_dir']

    return None


def set_intermediate_dir(
    pipeline_config: Dict[str, Any],
    new_dir: Union[str, Path],
    in_place: bool = False
) -> Dict[str, Any]:
    """Update the intermediate directory in pipeline configuration."""
    if not in_place:
        pipeline_config = copy.deepcopy(pipeline_config)

    # Ensure pipeline structure exists
    if 'pipeline' not in pipeline_config:
        pipeline_config['pipeline'] = {}

    if 'output' not in pipeline_config['pipeline']:
        pipeline_config['pipeline']['output'] = {}

    pipeline_config['pipeline']['output']['intermediate_dir'] = str(new_dir)

    return pipeline_config


def apply_path_mappings(
    pipeline_config: Dict[str, Any],
    dataset_path_mapping: Optional[Dict[str, str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    intermediate_dir: Optional[Union[str, Path]] = None,
    in_place: bool = False
) -> Dict[str, Any]:
    """Apply multiple path mappings to pipeline configuration in one call."""
    if not in_place:
        pipeline_config = copy.deepcopy(pipeline_config)

    # Apply dataset path mappings
    if dataset_path_mapping:
        pipeline_config = set_dataset_paths(
            pipeline_config,
            dataset_path_mapping,
            in_place=True
        )

    # Apply output path
    if output_path is not None:
        pipeline_config = set_output_path(
            pipeline_config,
            output_path,
            in_place=True
        )

    # Apply intermediate directory
    if intermediate_dir is not None:
        pipeline_config = set_intermediate_dir(
            pipeline_config,
            intermediate_dir,
            in_place=True
        )

    return pipeline_config


def get_all_file_paths(pipeline_config: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract all file paths from pipeline configuration."""
    result = {
        'datasets': [],
        'output': [],
        'intermediate': []
    }

    # Get dataset paths
    dataset_paths = get_dataset_paths(pipeline_config)
    result['datasets'] = list(dataset_paths.values())

    # Get output path
    output_path = get_output_path(pipeline_config)
    if output_path:
        result['output'] = [output_path]

    # Get intermediate directory
    intermediate_dir = get_intermediate_dir(pipeline_config)
    if intermediate_dir:
        result['intermediate'] = [intermediate_dir]

    return result
