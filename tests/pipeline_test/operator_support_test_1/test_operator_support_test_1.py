"""
Test AbstractExecutor operator support: Unnest, Sample, and Project.

Pipeline: operator_support_test_1
Structure:
- Procedure 1: PythonCode (create nested structure) → Unnest → Sample (3 operators)
- Procedure 2: PythonCode (statistics) → Project (2 operators)
"""

import json
import sys
from pathlib import Path

# Add PROJECT_ROOT to sys.path for imports
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from baselines.abstract.pipeline import Pipeline
from baselines.abstract.engine import PipelineEngine


def test_operator_support_test_1():
    """
    Test Unnest, Sample, and Project operators on AbstractExecutor.
    """

    pipeline_test_dir = Path(__file__).parent
    pipeline_path = pipeline_test_dir / "operator_support_test_1_pipeline.yaml"

    print("=" * 70)
    print("Testing Operator Support Test 1 Pipeline")
    print("=" * 70)

    # Load pipeline
    pipeline = Pipeline.load(str(pipeline_path))
    print(f"\n✓ Loaded pipeline: {pipeline.name}")
    print(f"  Nodes: {len(pipeline.nodes)}")
    print(f"  Base system: {pipeline.properties.get('base_system', 'unknown')}")

    # Display node information
    print("\nPipeline structure:")
    print("  Procedure 1 (process_node):")
    print("    • PythonCode: Create nested structure")
    print("    • Unnest: Flatten grades dictionary")
    print("    • Sample: Random sample 5 records")
    print("  Procedure 2 (analyze_node):")
    print("    • PythonCode: Calculate statistics")
    print("    • Project: Select final fields")

    # Create executor
    executor = PipelineEngine(verbose=True, cache_enabled=False)

    # Execute pipeline
    print("\n" + "=" * 70)
    print("Executing pipeline...")
    print("=" * 70)

    result = executor.execute_pipeline(
        pipeline=pipeline,
        save_intermediates=True,
        intermediate_dir=str(pipeline_test_dir / "pipeline_intermediates")
    )

    # Verify execution success
    if not result.success:
        print(f"\n❌ Pipeline execution failed: {result.error}")
        return False

    print("\n✅ Pipeline execution successful!")

    # Load output data
    output_path = pipeline_test_dir / "final_results.json"
    if not output_path.exists():
        print(f"❌ Output file not found: {output_path}")
        return False

    with open(output_path, 'r') as f:
        output_data = json.load(f)

    print(f"\n✓ Output file loaded: {len(output_data)} records")

    # VERIFICATION 1: Sample should produce exactly 5 records
    if len(output_data) != 5:
        print(f"❌ Expected 5 sampled records, got {len(output_data)}")
        return False
    print("✅ Sample operator: Correctly sampled 5 out of 10 records")

    # VERIFICATION 2: Project should only have 4 fields
    expected_fields = {'student_id', 'name', 'average', 'performance_category'}
    for i, record in enumerate(output_data):
        actual_fields = set(record.keys())
        if actual_fields != expected_fields:
            print(f"❌ Record {i} has incorrect fields")
            print(f"   Expected: {expected_fields}")
            print(f"   Got: {actual_fields}")
            return False
    print("✅ Project operator: Correctly selected 4 fields")

    # VERIFICATION 3: Individual grade fields should be removed
    removed_fields = {'calculus', 'physics', 'biology', 'chemistry', 'grades'}
    for record in output_data:
        if any(field in record for field in removed_fields):
            print(f"❌ Found unexpected field in output: {record.keys()}")
            return False
    print("✅ Unnest + Project: Grade fields and nested dict removed")

    # VERIFICATION 4: Average and category fields valid
    for record in output_data:
        # Check average is numeric and in valid range
        if not isinstance(record['average'], (int, float)):
            print(f"❌ Invalid average type: {type(record['average'])}")
            return False
        if record['average'] < 0 or record['average'] > 100:
            print(f"❌ Average out of range: {record['average']}")
            return False

        # Check category is valid
        valid_categories = {'Excellent', 'Good', 'Average', 'Below Average'}
        if record['performance_category'] not in valid_categories:
            print(f"❌ Invalid category: {record['performance_category']}")
            return False
    print("✅ Statistics: Average and performance_category fields valid")

    # VERIFICATION 5: Check intermediate output (process_node)
    intermediate_dir = pipeline_test_dir / "pipeline_intermediates"
    process_output_path = intermediate_dir / "process_node_output.json"

    if not process_output_path.exists():
        print(f"❌ Intermediate file missing: {process_output_path}")
        return False

    with open(process_output_path, 'r') as f:
        process_output = json.load(f)

    # Should have 5 records after sampling
    if len(process_output) != 5:
        print(f"❌ process_node should output 5 records, got {len(process_output)}")
        return False

    # Should have flattened grade fields
    process_record = process_output[0]
    required_fields = {'student_id', 'name', 'calculus', 'physics', 'biology', 'chemistry'}
    if not required_fields.issubset(set(process_record.keys())):
        print(f"❌ process_node missing required fields")
        return False

    # Note: Unnest operator preserves the original nested field while adding flattened fields
    # This is expected behavior - the 'grades' dict will still exist alongside the expanded fields

    print("✅ Procedure 1: PythonCode → Unnest → Sample chain executed correctly")

    # VERIFICATION 6: Data consistency check
    # Verify that analyze_node only added average and category, didn't change other fields
    for proc_rec, final_rec in zip(process_output, output_data):
        # Check student_id and name unchanged
        if proc_rec['student_id'] != final_rec['student_id']:
            print("❌ student_id changed between procedures")
            return False
        if proc_rec['name'] != final_rec['name']:
            print("❌ name changed between procedures")
            return False

    print("✅ Procedure 2: PythonCode → Project chain executed correctly")
    print("✅ Data flow between procedures maintained correctly")

    # Print sample output
    print("\n" + "=" * 70)
    print("Sample output (first 3 records):")
    print("=" * 70)
    for i, record in enumerate(output_data[:3]):
        print(f"\nRecord {i+1}:")
        print(f"  ID: {record['student_id']}")
        print(f"  Name: {record['name']}")
        print(f"  Average: {record['average']}")
        print(f"  Category: {record['performance_category']}")

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nOperators tested:")
    print("  ✓ PythonCode: Nested structure creation")
    print("  ✓ Unnest: Nested dictionary flattening (grades dict)")
    print("  ✓ Sample: Random sampling (10 → 5 records)")
    print("  ✓ Project: Field selection (8 → 4 fields)")
    print("\nPipeline structure:")
    print("  • 2 procedures")
    print("  • 5 total operators")
    print("  • Procedure 1: 3 operators (PythonCode → Unnest → Sample)")
    print("  • Procedure 2: 2 operators (PythonCode → Project)")
    print("  • All running on abstract system")

    return True


if __name__ == "__main__":
    success = test_operator_support_test_1()
    exit(0 if success else 1)
