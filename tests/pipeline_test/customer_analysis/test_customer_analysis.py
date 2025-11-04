"""
Test script for customer analysis pipeline.

This script:
1. Loads the customer_analysis_pipeline.yaml pipeline
2. Executes the pipeline using AbstractExecutor (validation happens automatically)
3. Verifies all expected fields are present in the output
4. Validates calculation accuracy
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


def test_customer_analysis():
    """Load and execute the customer analysis pipeline using AbstractExecutor."""

    # Get paths
    pipeline_test_dir = Path(__file__).parent
    pipeline_path = pipeline_test_dir / "customer_analysis_pipeline.yaml"
    output_path = pipeline_test_dir / "customer_insights.json"

    print("=" * 60)
    print("Testing Customer Analysis Pipeline")
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

        # Load and count operators
        print(f"\n  📊 Operator count:")
        total_operators = 0
        for node_id, node in pipeline.nodes.items():
            from baselines.abstract.procedure import Procedure
            procedure = Procedure.load(node.procedure_path)
            op_count = len(procedure.operators)
            total_operators += op_count
            print(f"    {node_id}: {op_count} operators")
        print(f"  Total operators in pipeline: {total_operators}")

        if total_operators != 5:
            print(f"  ⚠️  Expected 5 operators (3 + 2), got {total_operators}")

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

            # Expected fields after all 5 operators
            expected_fields = [
                'customer_id', 'name', 'purchases', 'join_date', 'region',
                # After operator 1 (calculate_stats)
                'total_spent', 'item_count', 'avg_purchase', 'top_category',
                # After operator 2 (enrich_profile)
                'tenure_months', 'spending_tier', 'category_diversity',
                # After operator 3 (flatten_details)
                'purchase_summary', 'price_min', 'price_max',
                # After operator 4 (calculate_engagement)
                'engagement_score', 'engagement_level',
                # After operator 5 (generate_recommendations)
                'recommended_category', 'recommendation_message', 'estimated_next_purchase'
            ]

            print(f"\n  Field verification (expecting {len(expected_fields)} fields):")
            all_fields_present = True
            for record in output_data:
                missing_fields = [f for f in expected_fields if f not in record]
                if missing_fields:
                    print(f"    ❌ Customer {record.get('customer_id', 'Unknown')}: Missing {missing_fields}")
                    all_fields_present = False

            if all_fields_present:
                print(f"    ✓ All expected fields present in all records")
            else:
                print(f"    ❌ Some fields are missing!")
                return False

            # Verify calculations for first customer (Alice Johnson)
            print(f"\n  Calculation verification:")
            alice = next((r for r in output_data if r['customer_id'] == 'C001'), None)
            if alice:
                print(f"    Customer: {alice['name']}")

                # Verify total_spent (1200 + 25 + 350 = 1575)
                expected_total = 1575
                actual_total = alice['total_spent']
                if actual_total == expected_total:
                    print(f"      ✓ Total spent: ${actual_total} (correct)")
                else:
                    print(f"      ❌ Total spent: ${actual_total} (expected ${expected_total})")
                    return False

                # Verify item_count (3 items)
                expected_count = 3
                actual_count = alice['item_count']
                if actual_count == expected_count:
                    print(f"      ✓ Item count: {actual_count} (correct)")
                else:
                    print(f"      ❌ Item count: {actual_count} (expected {expected_count})")
                    return False

                # Verify avg_purchase (1575 / 3 = 525)
                expected_avg = 525.0
                actual_avg = alice['avg_purchase']
                if actual_avg == expected_avg:
                    print(f"      ✓ Average purchase: ${actual_avg} (correct)")
                else:
                    print(f"      ❌ Average purchase: ${actual_avg} (expected ${expected_avg})")
                    return False

                # Verify top_category (Electronics appears twice)
                expected_category = 'Electronics'
                actual_category = alice['top_category']
                if actual_category == expected_category:
                    print(f"      ✓ Top category: {actual_category} (correct)")
                else:
                    print(f"      ❌ Top category: {actual_category} (expected {expected_category})")
                    return False

                # Verify spending_tier (total_spent 1575 >= 1000 = Platinum)
                expected_tier = 'Platinum'
                actual_tier = alice['spending_tier']
                if actual_tier == expected_tier:
                    print(f"      ✓ Spending tier: {actual_tier} (correct)")
                else:
                    print(f"      ❌ Spending tier: {actual_tier} (expected {expected_tier})")
                    return False

                # Display engagement metrics
                print(f"      ✓ Engagement score: {alice['engagement_score']}")
                print(f"      ✓ Engagement level: {alice['engagement_level']}")
                print(f"      ✓ Recommended category: {alice['recommended_category']}")
                print(f"      ✓ Estimated next purchase: ${alice['estimated_next_purchase']}")

            # Display summary for all customers
            print(f"\n  Customer summary:")
            for i, record in enumerate(output_data):
                print(f"    {i+1}. {record['name']} ({record['customer_id']})")
                print(f"       Spent: ${record['total_spent']} | Tier: {record['spending_tier']}")
                print(f"       Items: {record['item_count']} | Top category: {record['top_category']}")
                print(f"       Engagement: {record['engagement_level']} ({record['engagement_score']})")
                print(f"       Recommendation: {record['recommended_category']}")

            print(f"\n✅ All verifications passed!")
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
        success = test_customer_analysis()

        print("\n" + "=" * 60)
        if success:
            print("✅ Customer analysis test completed successfully!")
        else:
            print("❌ Customer analysis test failed!")
        print("=" * 60)

        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
