"""
Test script for mixed procedure pipeline (Abstract-DocETL-Abstract).

This script:
1. Loads the mixed_unnest_pipeline.yaml pipeline
2. Executes a pipeline with three procedures using different base systems:
   - Procedure 1: Abstract (PythonCode) - Data validation
   - Procedure 2: DocETL (Unnest) - Flatten nested grades dictionary
   - Procedure 3: Abstract (PythonCode) - Statistics calculation
3. Verifies the unnest operator correctly flattens nested structures
4. Validates calculation accuracy for averages and categories
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
from baselines.abstract.engine import PipelineEngine


def test_mixed_unnest():
    """Load and execute the mixed unnest pipeline using PipelineEngine."""

    # Get paths
    pipeline_test_dir = Path(__file__).parent
    pipeline_path = pipeline_test_dir / "mixed_unnest_pipeline.yaml"
    output_path = pipeline_test_dir / "final_statistics.json"

    print("=" * 70)
    print("Testing Mixed Procedure Pipeline (Abstract-DocETL-Abstract)")
    print("=" * 70)

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

        # Display nodes and their procedures
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

        # Load and verify procedures
        print(f"\n  📊 Procedure details:")
        from baselines.abstract.procedure import Procedure

        expected_systems = {
            'prepare_node': 'abstract',
            'unnest_node': 'docetl',
            'calculate_node': 'abstract'
        }

        for node_id, node in pipeline.nodes.items():
            procedure = Procedure.load(node.procedure_path)
            op_count = len(procedure.operators)
            print(f"    {node_id}:")
            print(f"      Base system: {procedure.base_system}")
            print(f"      Operators: {op_count}")

            # Verify base system
            expected_system = expected_systems.get(node_id)
            if procedure.base_system != expected_system:
                print(f"      ❌ Expected base_system '{expected_system}', got '{procedure.base_system}'")
                return False

            # Show operator types
            for op in procedure.operators:
                print(f"        - {op.name} ({op.type}, system: {op.source.get('system')})")

    except Exception as e:
        print(f"  ❌ Failed to load pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Create executor
    print(f"\n⚙️  Creating executor...")
    executor = PipelineEngine(verbose=True, cache_enabled=False)
    print(f"  ✓ Executor created")

    # Execute pipeline
    print(f"\n▶️  Executing pipeline...")
    print(f"{'=' * 70}\n")

    # Create config with default_model from pipeline properties
    config = {
        'default_model': pipeline.properties.get('default_model', 'gpt-4o-mini')
    }

    result = executor.execute_pipeline(
        pipeline=pipeline,
        config=config,
        save_intermediates=True,
        intermediate_dir=str(pipeline_test_dir / "pipeline_intermediates")
    )

    print(f"\n{'=' * 70}")

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

            # Expected 5 students
            if len(output_data) != 5:
                print(f"    ❌ Expected 5 records, got {len(output_data)}")
                return False

            # Expected fields after all operators
            expected_fields = [
                'name', 'calculus', 'physics', 'biology',
                'average_grade', 'category'
            ]

            print(f"\n  Field verification (expecting {len(expected_fields)} fields):")
            all_fields_present = True
            for record in output_data:
                missing_fields = [f for f in expected_fields if f not in record]
                if missing_fields:
                    print(f"    ❌ Student {record.get('name', 'Unknown')}: Missing {missing_fields}")
                    all_fields_present = False

            if all_fields_present:
                print(f"    ✓ All expected fields present in all records")
            else:
                print(f"    ❌ Some fields are missing!")
                return False

            # Verify unnest worked correctly - check that grades dict is flattened
            print(f"\n  Unnest verification:")
            for record in output_data:
                if 'grades' in record:
                    print(f"    ❌ 'grades' field still present - unnest didn't flatten the dict!")
                    return False
            print(f"    ✓ Grades dictionary successfully flattened (no 'grades' field in output)")

            # Verify calculations for Tom (calculus=80, physics=70, biology=75, avg=75)
            print(f"\n  Calculation verification:")
            tom = next((r for r in output_data if r['name'] == 'Tom'), None)
            if tom:
                print(f"    Student: {tom['name']}")

                # Verify individual grades
                if tom['calculus'] != 80 or tom['physics'] != 70 or tom['biology'] != 75:
                    print(f"      ❌ Grades incorrect: calculus={tom['calculus']}, physics={tom['physics']}, biology={tom['biology']}")
                    return False
                print(f"      ✓ Grades: calculus={tom['calculus']}, physics={tom['physics']}, biology={tom['biology']}")

                # Verify average (75.0)
                expected_avg = 75.0
                actual_avg = tom['average_grade']
                if abs(actual_avg - expected_avg) > 0.01:
                    print(f"      ❌ Average: {actual_avg} (expected {expected_avg})")
                    return False
                print(f"      ✓ Average grade: {actual_avg} (correct)")

                # Verify category (Average)
                expected_category = 'Average'
                actual_category = tom['category']
                if actual_category != expected_category:
                    print(f"      ❌ Category: {actual_category} (expected {expected_category})")
                    return False
                print(f"      ✓ Category: {actual_category} (correct)")

            # Verify Alice (calculus=95, physics=88, biology=92, avg=91.67, category=Excellent)
            alice = next((r for r in output_data if r['name'] == 'Alice'), None)
            if alice:
                print(f"\n    Student: {alice['name']}")

                # Verify average
                expected_avg = round((95 + 88 + 92) / 3.0, 2)
                actual_avg = alice['average_grade']
                if abs(actual_avg - expected_avg) > 0.01:
                    print(f"      ❌ Average: {actual_avg} (expected {expected_avg})")
                    return False
                print(f"      ✓ Average grade: {actual_avg} (correct)")

                # Verify category
                expected_category = 'Excellent'
                actual_category = alice['category']
                if actual_category != expected_category:
                    print(f"      ❌ Category: {actual_category} (expected {expected_category})")
                    return False
                print(f"      ✓ Category: {actual_category} (correct)")

            # Display summary for all students
            print(f"\n  Student summary:")
            for i, record in enumerate(output_data):
                print(f"    {i+1}. {record['name']}")
                print(f"       Calculus: {record['calculus']}, Physics: {record['physics']}, Biology: {record['biology']}")
                print(f"       Average: {record['average_grade']} | Category: {record['category']}")

            print(f"\n✅ All verifications passed!")
            print(f"\n🎯 Test validates:")
            print(f"   ✓ Mixed abstract-docetl-abstract procedure execution")
            print(f"   ✓ DocETL Unnest operator flattens nested dictionaries")
            print(f"   ✓ Data flows correctly between different base systems")
            print(f"   ✓ Schema transformation works as expected")
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
        success = test_mixed_unnest()

        print("\n" + "=" * 70)
        if success:
            print("✅ Mixed unnest pipeline test completed successfully!")
        else:
            print("❌ Mixed unnest pipeline test failed!")
        print("=" * 70)

        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
