"""
Test script for fixed loop functionality.

This script:
1. Loads the fixed_loop_pipeline.yaml pipeline
2. Executes the pipeline using AbstractExecutor (validation happens automatically)
3. Verifies values are multiplied by 2 three times (e.g., 1 → 2 → 4 → 8)
"""

import json
import sys
from pathlib import Path

# Add PROJECT_ROOT to sys.path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from baselines.abstract
from baselines.abstract.pipeline import Pipeline
from baselines.abstract.engine import AbstractExecutor


def test_fixed_loop():
    """Load and execute the fixed loop pipeline using AbstractExecutor."""

    # Get paths
    pipeline_test_dir = Path(__file__).parent
    pipeline_path = pipeline_test_dir / "fixed_loop_pipeline.yaml"
    output_path = pipeline_test_dir / "loop_output.json"

    print("=" * 60)
    print("Testing Fixed Loop Pipeline")
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
            print(f"    Iterations: {node.metadata.get('iterations', 'N/A')}")
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

            # Expected values after 3 iterations of multiplying by 2
            # 1 → 2 → 4 → 8
            # 2 → 4 → 8 → 16
            # 3 → 6 → 12 → 24
            expected_values = [8, 16, 24]

            print(f"\n  Expected final values: {expected_values}")
            print(f"  Actual values:")

            all_correct = True
            for i, record in enumerate(output_data):
                actual = record.get('value')
                expected = expected_values[i]
                status = "✓" if actual == expected else "❌"
                print(f"    {status} Record {i+1}: {actual} (expected: {expected})")

                if actual != expected:
                    all_correct = False

            if all_correct:
                print(f"\n✅ All values match expected results!")
                print(f"   Each value was multiplied by 2 three times (x2 x2 x2 = x8)")
                return True
            else:
                print(f"\n❌ Some values don't match expected results!")
                return False
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
        success = test_fixed_loop()

        print("\n" + "=" * 60)
        if success:
            print("✅ Fixed loop test completed successfully!")
        else:
            print("❌ Fixed loop test failed!")
        print("=" * 60)

        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
