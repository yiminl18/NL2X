#!/usr/bin/env python3
"""
Pipeline DAG Visualization Tool

Generates an interactive HTML visualization of pipeline DAG structure,
showing nodes, dependencies, execution levels, and parallel execution groups.

Usage:
    python baselines/abstract/tools/visualize_pipeline.py <pipeline_yaml> [--output <html_file>]

Example:
    python baselines/abstract/tools/visualize_pipeline.py tests/pipeline_test/sum_and_average_pipeline.yaml
    # Output: tests/pipeline_test/sum_and_average_pipeline.html
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Set
import json
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

pipeline_module_path = project_root / "baselines" / "abstract" / "pipeline.py"
spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_module_path)
pipeline_module = importlib.util.module_from_spec(spec)
sys.modules['pipeline_module'] = pipeline_module
spec.loader.exec_module(pipeline_module)

Pipeline = pipeline_module.Pipeline


# Color scheme for node types
NODE_COLORS = {
    'fanout': '#FFA500',              # Orange - splits flow
    'normal': '#4A90E2',              # Blue - standard processing
    'aggregate': '#9B59B6',           # Purple - merges inputs
    'final': '#27AE60',               # Green - outputs to file
    'conditional_routing': '#E74C3C'  # Red - routes by condition
}

NODE_TYPE_DESCRIPTIONS = {
    'fanout': 'Splits to multiple outputs',
    'normal': 'Standard processing',
    'aggregate': 'Merges multiple inputs',
    'final': 'Outputs to file',
    'conditional_routing': 'Routes records by condition'
}


def compute_levels(pipeline: Pipeline) -> List[List[str]]:
    """
    Group nodes into execution levels for parallel execution visualization.

    Uses Kahn's algorithm to perform topological sorting by levels.
    Nodes in the same level have no dependencies on each other and can
    execute in parallel.

    Args:
        pipeline: Pipeline object

    Returns:
        List of levels, where each level is a list of node IDs

    Raises:
        ValueError: If pipeline contains cycles
    """
    levels = []
    in_degree = {node_id: len(node.parents)
                 for node_id, node in pipeline.nodes.items()}

    while in_degree:
        # Current level: all nodes with zero dependencies
        current_level = [nid for nid, deg in in_degree.items() if deg == 0]

        if not current_level:
            raise ValueError("Pipeline contains cycles - cannot compute execution levels")

        levels.append(current_level)

        # Remove current level nodes and update in-degrees
        for node_id in current_level:
            del in_degree[node_id]
            for child_id in pipeline.nodes[node_id].children:
                if child_id in in_degree:
                    in_degree[child_id] -= 1

    return levels


def load_and_analyze_pipeline(pipeline_path: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Load pipeline from YAML and extract visualization data.

    Args:
        pipeline_path: Path to pipeline YAML file
        verbose: Print analysis details

    Returns:
        Dictionary containing:
            - name: Pipeline name
            - properties: Pipeline properties dict
            - nodes: Dict of node_id -> node information
            - edges: Dict of from_id -> list of to_ids
            - levels: List of execution levels
            - file_inputs: Set of input file paths
            - file_outputs: Set of output file paths

    Raises:
        FileNotFoundError: If pipeline file doesn't exist
        ValueError: If pipeline is invalid
    """
    # Load pipeline
    pipeline_path = Path(pipeline_path).resolve()
    if not pipeline_path.exists():
        raise FileNotFoundError(f"Pipeline file not found: {pipeline_path}")

    if verbose:
        print(f"Loading pipeline from: {pipeline_path}")

    pipeline = Pipeline.load(str(pipeline_path))

    # Compute execution levels
    try:
        levels = compute_levels(pipeline)
    except ValueError as e:
        raise ValueError(f"Pipeline validation failed: {e}")

    if verbose:
        print(f"Pipeline: {pipeline.name}")
        print(f"Nodes: {len(pipeline.nodes)}")
        print(f"Execution levels: {len(levels)}")
        for i, level in enumerate(levels):
            print(f"  Level {i}: {len(level)} nodes - {level}")

    # Extract node information
    nodes_data = {}
    file_inputs: Set[str] = set()
    file_outputs: Set[str] = set()

    # Map node_id to level
    node_to_level = {}
    for level_idx, level_nodes in enumerate(levels):
        for node_id in level_nodes:
            node_to_level[node_id] = level_idx

    for node_id, node in pipeline.nodes.items():
        # Extract file inputs and outputs
        for ds in node.data_sources:
            if ds.ref_type == 'file':
                file_inputs.add(ds.ref)

        for output in node.outputs:
            if output.ref_type == 'file':
                file_outputs.add(output.ref)

        # Build node data
        nodes_data[node_id] = {
            'type': node.node_type.value,
            'description': node.metadata.get('description', ''),
            'procedure': Path(node.procedure_path).stem,
            'level': node_to_level[node_id],
            'parents': node.parents,
            'children': node.children,
            'file_inputs': [ds.ref for ds in node.data_sources if ds.ref_type == 'file'],
            'node_inputs': [ds.ref for ds in node.data_sources if ds.ref_type == 'node'],
            'file_outputs': [out.ref for out in node.outputs if out.ref_type == 'file'],
            'node_outputs': [
                {'ref': out.ref, 'output_name': out.output_name}
                for out in node.outputs if out.ref_type == 'node'
            ],
            'routing_field': node.metadata.get('routing_field')  # For conditional routing
        }

    return {
        'name': pipeline.name,
        'properties': pipeline.properties,
        'nodes': nodes_data,
        'edges': dict(pipeline.edges),
        'levels': levels,
        'file_inputs': sorted(file_inputs),
        'file_outputs': sorted(file_outputs)
    }


def build_vis_data(pipeline_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert pipeline data to vis.js network format.

    Args:
        pipeline_data: Pipeline analysis data from load_and_analyze_pipeline

    Returns:
        Dictionary with 'nodes' and 'edges' arrays in vis.js format
    """
    vis_nodes = []
    vis_edges = []

    # Build nodes
    for node_id, node_info in pipeline_data['nodes'].items():
        node_type = node_info['type']
        level = node_info['level']

        # Build tooltip HTML
        tooltip_parts = [
            f"<div style='max-width: 300px;'>",
            f"<strong>{node_id}</strong> [{node_type.upper()}]<br/>",
            f"<hr style='margin: 5px 0;'/>",
            f"<b>Description:</b> {node_info['description']}<br/>",
            f"<b>Procedure:</b> {node_info['procedure']}<br/>",
            f"<b>Level:</b> {level}<br/>"
        ]

        if node_info['file_inputs'] or node_info['node_inputs']:
            tooltip_parts.append(f"<hr style='margin: 5px 0;'/><b>Inputs:</b><br/>")
            for file_input in node_info['file_inputs']:
                tooltip_parts.append(f"• file: {Path(file_input).name}<br/>")
            for node_input in node_info['node_inputs']:
                tooltip_parts.append(f"• node: {node_input}<br/>")

        # Special handling for conditional routing nodes
        if node_type == 'conditional_routing':
            routing_field = node_info.get('routing_field', '_route_to')
            tooltip_parts.append(f"<hr style='margin: 5px 0;'/><b>Routing Field:</b> <code>{routing_field}</code><br/>")
            tooltip_parts.append(f"<b>Output Branches:</b><br/>")
            for output in node_info['node_outputs']:
                branch_name = output['output_name']
                target_node = output['ref']
                tooltip_parts.append(f"• <span style='color: #E74C3C; font-weight: bold;'>{branch_name}</span> → {target_node}<br/>")
        elif node_info['file_outputs'] or node_info['node_outputs']:
            tooltip_parts.append(f"<hr style='margin: 5px 0;'/><b>Outputs:</b><br/>")
            for file_output in node_info['file_outputs']:
                tooltip_parts.append(f"• file: {Path(file_output).name}<br/>")
            for node_output in node_info['node_outputs']:
                tooltip_parts.append(f"• node: {node_output['ref']}<br/>")

        tooltip_parts.append("</div>")
        tooltip_html = "".join(tooltip_parts)

        # Create vis.js node
        vis_node = {
            'id': node_id,
            'label': node_id,
            'title': tooltip_html,
            'level': level,
            'color': {
                'background': NODE_COLORS[node_type],
                'border': '#2c3e50',
                'highlight': {
                    'background': NODE_COLORS[node_type],
                    'border': '#000000'
                }
            },
            'font': {
                'size': 14,
                'color': '#ffffff' if node_type in ['aggregate', 'final', 'conditional_routing'] else '#000000'
            },
            'shape': 'diamond' if node_type == 'conditional_routing' else 'box',
            'margin': 10,
            'borderWidth': 3 if node_type == 'conditional_routing' else 2
        }

        vis_nodes.append(vis_node)

    # Add file nodes
    file_nodes_added = set()
    file_to_node_id = {}

    # Add input file nodes
    for file_path in pipeline_data['file_inputs']:
        if file_path not in file_nodes_added:
            file_name = Path(file_path).name
            file_id = f"file_input_{file_name}"
            vis_node = {
                'id': file_id,
                'label': file_name,
                'title': f"<div style='max-width: 250px;'><strong>Input File</strong><br/><hr style='margin: 5px 0;'/>{file_path}</div>",
                'level': -1,  # Place before Level 0
                'color': {
                    'background': '#ecf0f1',  # Light gray
                    'border': '#7f8c8d',
                    'highlight': {
                        'background': '#bdc3c7',
                        'border': '#2c3e50'
                    }
                },
                'font': {'size': 12, 'color': '#2c3e50'},
                'shape': 'ellipse',  # Oval for inputs
                'margin': 8
            }
            vis_nodes.append(vis_node)
            file_nodes_added.add(file_path)
            file_to_node_id[file_path] = file_id

    # Add output file nodes
    for file_path in pipeline_data['file_outputs']:
        if file_path not in file_nodes_added:
            file_name = Path(file_path).name
            file_id = f"file_output_{file_name}"
            vis_node = {
                'id': file_id,
                'label': file_name,
                'title': f"<div style='max-width: 250px;'><strong>Output File</strong><br/><hr style='margin: 5px 0;'/>{file_path}</div>",
                'level': len(pipeline_data['levels']),  # Place after last level
                'color': {
                    'background': '#d5f4e6',  # Light green
                    'border': '#27ae60',
                    'highlight': {
                        'background': '#a9dfbf',
                        'border': '#1e8449'
                    }
                },
                'font': {'size': 12, 'color': '#2c3e50'},
                'shape': 'ellipse',  # Ellipse for outputs
                'margin': 8
            }
            vis_nodes.append(vis_node)
            file_nodes_added.add(file_path)
            file_to_node_id[file_path] = file_id

    # Build edges (node-to-node)
    for from_id, to_ids in pipeline_data['edges'].items():
        from_node = pipeline_data['nodes'][from_id]

        for to_id in to_ids:
            vis_edge = {
                'from': from_id,
                'to': to_id,
                'arrows': 'to',
                'color': {
                    'color': '#7f8c8d',
                    'highlight': '#2c3e50'
                },
                'smooth': {
                    'type': 'cubicBezier'
                }
            }

            # Add label for conditional routing edges
            if from_node['type'] == 'conditional_routing':
                # Find the output_name for this specific to_id
                for output in from_node['node_outputs']:
                    if output['ref'] == to_id:
                        vis_edge['label'] = output['output_name']
                        vis_edge['font'] = {
                            'size': 12,
                            'color': '#E74C3C',
                            'strokeWidth': 0,
                            'align': 'horizontal',
                            'bold': True
                        }
                        # Use different color for routing edges
                        vis_edge['color'] = {
                            'color': '#E74C3C',
                            'highlight': '#C0392B'
                        }
                        break

            vis_edges.append(vis_edge)

    # Build edges (file-to-node and node-to-file)
    for node_id, node_info in pipeline_data['nodes'].items():
        # Input file → pipeline node
        for file_input in node_info['file_inputs']:
            if file_input in file_to_node_id:
                vis_edge = {
                    'from': file_to_node_id[file_input],
                    'to': node_id,
                    'arrows': 'to',
                    'dashes': True,  # Dashed line for files
                    'color': {
                        'color': '#95a5a6',
                        'highlight': '#7f8c8d'
                    },
                    'smooth': {'type': 'cubicBezier'}
                }
                vis_edges.append(vis_edge)

        # Pipeline node → output file
        for file_output in node_info['file_outputs']:
            if file_output in file_to_node_id:
                vis_edge = {
                    'from': node_id,
                    'to': file_to_node_id[file_output],
                    'arrows': 'to',
                    'dashes': True,  # Dashed line for files
                    'color': {
                        'color': '#95a5a6',
                        'highlight': '#7f8c8d'
                    },
                    'smooth': {'type': 'cubicBezier'}
                }
                vis_edges.append(vis_edge)

    return {
        'nodes': vis_nodes,
        'edges': vis_edges
    }


def generate_html(vis_data: Dict[str, Any], pipeline_data: Dict[str, Any],
                  enable_physics: bool = False) -> str:
    """
    Generate complete HTML with embedded vis.js visualization.

    Args:
        vis_data: vis.js formatted network data
        pipeline_data: Original pipeline analysis data
        enable_physics: Enable physics simulation for layout

    Returns:
        Complete HTML string
    """
    pipeline_name = pipeline_data['name']
    properties = pipeline_data['properties']
    levels = pipeline_data['levels']

    # Get parallel execution info
    parallel_enabled = properties.get('parallel_execution', False)
    max_workers = properties.get('max_parallel_workers', 4)

    # Build level info HTML
    level_info_html = []
    for i, level_nodes in enumerate(levels):
        level_info_html.append(
            f"<div class='level-item'>Level {i}: "
            f"<span class='level-nodes'>{', '.join(level_nodes)}</span> "
            f"<span class='level-count'>({len(level_nodes)} node{'s' if len(level_nodes) > 1 else ''} in parallel)</span></div>"
        )
    level_info_str = "\n".join(level_info_html)

    # Build legend HTML
    legend_html = []
    for node_type, description in NODE_TYPE_DESCRIPTIONS.items():
        color = NODE_COLORS[node_type]
        legend_html.append(
            f"<div class='legend-item'>"
            f"<span class='legend-box' style='background-color: {color};'></span>"
            f"<span class='legend-label'>{node_type.upper()}</span> - {description}"
            f"</div>"
        )

    # Add file node types to legend
    legend_html.append(
        "<div class='legend-item'>"
        "<span class='legend-box legend-ellipse' style='background-color: #ecf0f1; border: 2px solid #7f8c8d;'></span>"
        "<span class='legend-label'>INPUT FILE</span> - Data source (ellipse)"
        "</div>"
    )
    legend_html.append(
        "<div class='legend-item'>"
        "<span class='legend-box legend-ellipse' style='background-color: #d5f4e6; border: 2px solid #27ae60;'></span>"
        "<span class='legend-label'>OUTPUT FILE</span> - Data output (ellipse)"
        "</div>"
    )

    legend_str = "\n".join(legend_html)

    # Serialize data for JavaScript
    nodes_json = json.dumps(vis_data['nodes'], indent=2)
    edges_json = json.dumps(vis_data['edges'], indent=2)

    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Visualization - {pipeline_name}</title>
    <script src="https://unpkg.com/vis-network@9.1.6/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: #f5f5f5;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }}

        #header {{
            background: #2c3e50;
            color: white;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        #header h1 {{
            font-size: 24px;
            margin-bottom: 10px;
        }}

        #header .subtitle {{
            font-size: 14px;
            color: #ecf0f1;
        }}

        #main-container {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}

        #sidebar {{
            width: 350px;
            background: white;
            padding: 20px;
            overflow-y: auto;
            box-shadow: 2px 0 4px rgba(0,0,0,0.1);
        }}

        #sidebar h2 {{
            font-size: 18px;
            margin-bottom: 15px;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }}

        #sidebar section {{
            margin-bottom: 25px;
        }}

        .info-row {{
            display: flex;
            margin-bottom: 8px;
        }}

        .info-label {{
            font-weight: 600;
            margin-right: 8px;
            color: #34495e;
        }}

        .info-value {{
            color: #7f8c8d;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}

        .legend-box {{
            width: 20px;
            height: 20px;
            border: 2px solid #2c3e50;
            margin-right: 10px;
            border-radius: 3px;
            flex-shrink: 0;
        }}

        .legend-ellipse {{
            border-radius: 50%;
        }}

        .legend-label {{
            font-weight: 600;
            margin-right: 8px;
            min-width: 90px;
        }}

        .level-item {{
            padding: 8px;
            margin-bottom: 8px;
            background: #ecf0f1;
            border-radius: 4px;
            font-size: 13px;
        }}

        .level-nodes {{
            font-weight: 600;
            color: #2c3e50;
        }}

        .level-count {{
            color: #7f8c8d;
            font-size: 12px;
        }}

        .file-list {{
            list-style: none;
            padding-left: 0;
        }}

        .file-list li {{
            padding: 4px 0;
            color: #34495e;
            font-size: 13px;
        }}

        .file-list li::before {{
            content: "📄 ";
            margin-right: 5px;
        }}

        #network-container {{
            flex: 1;
            background: white;
            position: relative;
        }}

        #network {{
            width: 100%;
            height: 100%;
        }}

        .status-badge {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
            margin-left: 8px;
        }}

        .status-enabled {{
            background: #27ae60;
            color: white;
        }}

        .status-disabled {{
            background: #95a5a6;
            color: white;
        }}
    </style>
</head>
<body>
    <div id="header">
        <h1>Pipeline DAG Visualization</h1>
        <div class="subtitle">{pipeline_name}</div>
    </div>

    <div id="main-container">
        <div id="sidebar">
            <section>
                <h2>Pipeline Info</h2>
                <div class="info-row">
                    <span class="info-label">Name:</span>
                    <span class="info-value">{pipeline_name}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Total Nodes:</span>
                    <span class="info-value">{len(pipeline_data['nodes'])}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Execution Levels:</span>
                    <span class="info-value">{len(levels)}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Parallel Execution:</span>
                    <span class="status-badge {'status-enabled' if parallel_enabled else 'status-disabled'}">
                        {'Enabled' if parallel_enabled else 'Disabled'}
                    </span>
                </div>
                {f'<div class="info-row"><span class="info-label">Max Workers:</span><span class="info-value">{max_workers}</span></div>' if parallel_enabled else ''}
            </section>

            <section>
                <h2>Node Types</h2>
                {legend_str}
            </section>

            <section>
                <h2>Execution Levels</h2>
                {level_info_str}
            </section>

            <section>
                <h2>Input Files</h2>
                <ul class="file-list">
                    {chr(10).join(f'<li>{Path(f).name}</li>' for f in pipeline_data['file_inputs']) if pipeline_data['file_inputs'] else '<li style="color: #95a5a6;">None</li>'}
                </ul>
            </section>

            <section>
                <h2>Output Files</h2>
                <ul class="file-list">
                    {chr(10).join(f'<li>{Path(f).name}</li>' for f in pipeline_data['file_outputs']) if pipeline_data['file_outputs'] else '<li style="color: #95a5a6;">None</li>'}
                </ul>
            </section>
        </div>

        <div id="network-container">
            <div id="network"></div>
        </div>
    </div>

    <script>
        // Create network data
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});

        // Create network
        var container = document.getElementById('network');
        var data = {{
            nodes: nodes,
            edges: edges
        }};

        var options = {{
            layout: {{
                hierarchical: {{
                    enabled: true,
                    direction: 'UD',
                    sortMethod: 'directed',
                    levelSeparation: 150,
                    nodeSpacing: 200,
                    treeSpacing: 250
                }}
            }},
            physics: {{
                enabled: {str(enable_physics).lower()},
                hierarchicalRepulsion: {{
                    nodeDistance: 150
                }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                navigationButtons: true,
                keyboard: true
            }},
            nodes: {{
                shape: 'box',
                margin: 10,
                widthConstraint: {{
                    minimum: 120,
                    maximum: 200
                }},
                font: {{
                    size: 14
                }},
                borderWidth: 2,
                borderWidthSelected: 3
            }},
            edges: {{
                arrows: {{
                    to: {{
                        enabled: true,
                        scaleFactor: 0.8
                    }}
                }},
                smooth: {{
                    type: 'cubicBezier',
                    roundness: 0.5
                }},
                width: 2
            }}
        }};

        var network = new vis.Network(container, data, options);

        // Status message
        console.log('Pipeline visualization loaded successfully');
        console.log('Nodes:', nodes.length);
        console.log('Edges:', edges.length);
        console.log('Levels:', {len(levels)});
    </script>
</body>
</html>'''

    return html_template


def main():
    parser = argparse.ArgumentParser(
        description='Generate interactive HTML visualization of pipeline DAG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python baselines/abstract/tools/visualize_pipeline.py tests/pipeline_test/sum_and_average_pipeline.yaml
  python baselines/abstract/tools/visualize_pipeline.py pipeline.yaml --output viz/my_pipeline.html
  python baselines/abstract/tools/visualize_pipeline.py pipeline.yaml --verbose
        '''
    )

    parser.add_argument(
        'pipeline',
        help='Path to pipeline YAML file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output HTML file path (default: <input_name>.html in same directory)',
        default=None
    )
    parser.add_argument(
        '--physics',
        action='store_true',
        help='Enable physics simulation for dynamic layout'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print analysis details'
    )

    args = parser.parse_args()

    try:
        # Load and analyze pipeline
        pipeline_data = load_and_analyze_pipeline(args.pipeline, verbose=args.verbose)

        # Build vis.js data
        vis_data = build_vis_data(pipeline_data)

        # Generate HTML
        html_content = generate_html(vis_data, pipeline_data, enable_physics=args.physics)

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            input_path = Path(args.pipeline)
            output_path = input_path.parent / f"{input_path.stem}.html"

        # Write output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding='utf-8')

        print(f"✓ Visualization generated successfully!")
        print(f"  Output: {output_path}")
        print(f"  Open in browser to view: file://{output_path.absolute()}")

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
