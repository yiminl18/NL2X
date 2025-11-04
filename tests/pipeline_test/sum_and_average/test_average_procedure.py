"""
Test script to execute the sum and average pipeline.

This script:
1. Loads the pipeline from sum_and_average_pipeline.yaml
2. Executes the pipeline using AbstractExecutor (validation happens automatically)
3. Verifies the output contains both total_score and average_score fields
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


def test_pipeline():
    """Load and execute the sum and average pipeline using AbstractExecutor."""

    # Get paths
    pipeline_test_dir = Path(__file__).parent
    pipeline_path = pipeline_test_dir / "sum_and_average_pipeline.yaml"

    # Expected output files from the 4 FINAL nodes
    expected_outputs = {
        'deviation': pipeline_test_dir / "grade_with_deviation.json",
        'high': pipeline_test_dir / "high_performers.json",
        'medium': pipeline_test_dir / "medium_performers.json",
        'low': pipeline_test_dir / "low_performers.json",
    }

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

        # Verify all expected output files exist
        print(f"\n📊 Output verification:")
        all_passed = True

        for name, output_path in expected_outputs.items():
            if not output_path.exists():
                print(f"  ❌ {name} output missing: {output_path.name}")
                all_passed = False
            else:
                print(f"  ✓ {name} output exists: {output_path.name}")

        if not all_passed:
            return False

        # Verify deviation file has expected fields
        deviation_path = expected_outputs['deviation']
        with open(deviation_path, 'r') as f:
            deviation_data = json.load(f)

        print(f"\n  Checking deviation output ({len(deviation_data)} records):")
        if deviation_data:
            first = deviation_data[0]
            required_fields = ['name', 'calculus', 'physics', 'biology', 'total_score', 'median_score', 'deviation_ratio']
            for field in required_fields:
                if field not in first:
                    print(f"    ❌ Missing field: {field}")
                    return False
            print(f"    ✓ All required fields present")

        # Verify high performers file
        high_path = expected_outputs['high']
        with open(high_path, 'r') as f:
            high_data = json.load(f)

        print(f"\n  Checking high performers output ({len(high_data)} records):")
        if high_data and len(high_data) > 0:
            # Should have average_score >= 90
            for record in high_data:
                if record.get('average_score', 0) < 90:
                    print(f"    ❌ Invalid high performer: {record.get('name')} has average {record.get('average_score')}")
                    return False
            print(f"    ✓ All records have average >= 90")

        # Verify medium performers file
        medium_path = expected_outputs['medium']
        with open(medium_path, 'r') as f:
            medium_data = json.load(f)

        print(f"\n  Checking medium performers output ({len(medium_data)} records):")
        if medium_data and len(medium_data) > 0:
            # Should have 80 <= average_score < 90
            for record in medium_data:
                avg = record.get('average_score', 0)
                if avg < 80 or avg >= 90:
                    print(f"    ❌ Invalid medium performer: {record.get('name')} has average {avg}")
                    return False
            print(f"    ✓ All records have 80 <= average < 90")

        # Verify low performers file
        low_path = expected_outputs['low']
        with open(low_path, 'r') as f:
            low_data = json.load(f)

        print(f"\n  Checking low performers output ({len(low_data)} records):")
        if low_data and len(low_data) > 0:
            # Should have average_score < 80
            for record in low_data:
                if record.get('average_score', 0) >= 80:
                    print(f"    ❌ Invalid low performer: {record.get('name')} has average {record.get('average_score')}")
                    return False
            print(f"    ✓ All records have average < 80")

        print(f"\n✅ All checks passed!")
        return True
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
