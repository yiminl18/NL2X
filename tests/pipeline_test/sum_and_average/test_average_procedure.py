"""
Test script to execute the sum and average pipeline.

This script:
1. Loads the pipeline from sum_and_average_pipeline.yaml
2. Validates the pipeline structure
3. Executes the pipeline using AbstractExecutor
4. Verifies the output contains both total_score and average_score fields
"""

import json
import sys
from pathlib import Path

# Add PROJECT_ROOT to sys.path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from baselines.abstract
from baselines.abstract.pipeline import Pipeline
from baselines.abstract.engine import AbstractExecutor


def test_pipeline():
    """Load and execute the sum and average pipeline using AbstractExecutor."""

    # Get paths
    pipeline_test_dir = Path(__file__).parent
    pipeline_path = pipeline_test_dir / "sum_and_average_pipeline.yaml"
    output_path = pipeline_test_dir / "grade_with_average.json"

    print("=" * 60)
    print("Testing Student Grade Analysis Pipeline")
    print("=" * 60)

    # Check pipeline file exists
    if not pipeline_path.exists():
        print(f"❌ Pipeline file not found: {pipeline_path}")
        return False

    print(f"\n📁 Pipeline file: {pipeline_path}")

    # Load the pipeline
    print(f"\n📋 Loading pipeline...")
    try:
        pipeline = Pipeline.load(str(pipeline_path))
        print(f"  ✓ Loaded: {pipeline.name}")
        print(f"  ✓ Nodes: {len(pipeline.nodes)}")

        # Display nodes
        for node_id, node in pipeline.nodes.items():
            print(f"\n  Node: {node_id}")
            print(f"    Type: {node.node_type.value}")
            print(f"    Procedure: {Path(node.procedure_path).name}")
            print(f"    Data sources: {len(node.data_sources)}")
            for ds in node.data_sources:
                print(f"      - {ds.ref_type}: {ds.ref}")
            print(f"    Outputs: {len(node.outputs)}")
            for out in node.outputs:
                print(f"      - {out.ref_type}: {out.ref}")

    except Exception as e:
        print(f"  ❌ Failed to load pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Validate pipeline
    print(f"\n🔍 Validating pipeline...")
    try:
        pipeline.validate()
        print(f"  ✓ Pipeline structure is valid")
    except Exception as e:
        print(f"  ❌ Validation failed: {e}")
        return False

    # Create executor
    print(f"\n⚙️  Creating executor...")
    executor = AbstractExecutor(verbose=True, cache_enabled=False)
    print(f"  ✓ Executor created")

    # Execute pipeline
    print(f"\n▶️  Executing pipeline...")
    print(f"{'=' * 60}\n")

    result = executor.execute_pipeline(
        pipeline=pipeline,
        save_intermediates=True,
        intermediate_dir=str(pipeline_test_dir / "pipeline_intermediates")
    )

    print(f"\n{'=' * 60}")

    # Check result
    if result.success:
        print(f"\n✅ Pipeline execution successful!")
        print(f"  Nodes executed: {result.metadata.get('nodes_executed', 'N/A')}")
        print(f"  Final node: {result.metadata.get('final_node_id', 'N/A')}")

        if result.metadata.get('final_output_path'):
            print(f"  Output saved to: {result.metadata['final_output_path']}")

        # Verify output file
        if output_path.exists():
            with open(output_path, 'r') as f:
                output_data = json.load(f)

            print(f"\n📊 Output verification:")
            print(f"  Records: {len(output_data)}")

            # Check first record
            if output_data:
                first = output_data[0]
                print(f"\n  Sample record:")
                print(f"    Name: {first.get('name', 'N/A')}")
                print(f"    Calculus: {first.get('calculus', 'N/A')}")
                print(f"    Physics: {first.get('physics', 'N/A')}")
                print(f"    Biology: {first.get('biology', 'N/A')}")
                print(f"    Total: {first.get('total_score', 'N/A')}")
                print(f"    Average: {first.get('average_score', 'N/A')}")

                # Verify both total_score and average_score exist
                if 'total_score' not in first:
                    print(f"\n    ❌ total_score field missing!")
                    return False

                if 'average_score' not in first:
                    print(f"\n    ❌ average_score field missing!")
                    return False

                # Verify calculations are correct
                expected_total = first['calculus'] + first['physics'] + first['biology']
                actual_total = first['total_score']
                if expected_total != actual_total:
                    print(f"\n    ❌ Total mismatch: expected {expected_total}, got {actual_total}")
                    return False
                print(f"    ✓ Sum calculation verified!")

                expected_avg = actual_total / 3
                actual_avg = first['average_score']
                if abs(expected_avg - actual_avg) > 0.01:
                    print(f"\n    ❌ Average mismatch: expected {expected_avg:.2f}, got {actual_avg}")
                    return False
                print(f"    ✓ Average calculation verified!")

            print(f"\n✅ All checks passed!")
            return True
        else:
            print(f"\n❌ Output file not created: {output_path}")
            return False
    else:
        print(f"\n❌ Pipeline execution failed:")
        print(f"  {result.error}")
        if result.metadata:
            print(f"  Metadata: {result.metadata}")
        return False


if __name__ == "__main__":
    try:
        success = test_pipeline()

        print("\n" + "=" * 60)
        if success:
            print("✅ Pipeline test completed successfully!")
        else:
            print("❌ Pipeline test failed!")
        print("=" * 60)

        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
