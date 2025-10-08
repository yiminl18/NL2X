#!/opt/homebrew/Caskroom/miniconda/base/envs/karma/bin/python
"""
Subplan Pipeline Execution Test
Based on examples/example2.yaml

This script demonstrates executing a pipeline in multiple segments:
1. Load YAML and sample 5 records
2. Execute first 2 operators (extract_medications, unnest_medications) -> subplan_1.json
3. Execute remaining operators (resolve_medications, summarize_prescriptions) -> subplan_2.json

Uses execute_pipeline_range() to run specific operator ranges.

Usage:
    python ./tests/test_subplan.py
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add baselines to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
EXAMPLES_DIR = PROJECT_ROOT / "examples"
sys.path.insert(0, str(PROJECT_ROOT / "baselines"))

# Import abstract layer components
from abstract import DatasetManager, SamplingConfig
from abstract.engine import AbstractExecutor
from abstract.convert.docetl import (
    yaml_to_abstract_pipeline,
    operator_to_dict
)


class SubplanTestRunner:
    """Test runner for executing pipeline in multiple segments."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.test_dir = SCRIPT_DIR
        self.examples_dir = EXAMPLES_DIR
        self.yaml_path = self.examples_dir / "example2.yaml"

        # Create output directories
        self.sample_data_dir = self.test_dir / "sample_data"
        self.output_dir = self.test_dir / "output"

        for dir_path in [self.sample_data_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Storage for test artifacts
        self.abstract_operators = None
        self.pipeline_config = None
        self.data_manager = None
        self.original_dataset_path = None
        self.sampled_dataset_path = None

        # Subplan outputs
        self.subplan_1_output = None
        self.subplan_2_output = None

    def log(self, message, level="INFO"):
        """Print log message with timestamp."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def step_1_load_and_sample(self):
        """
        Step 1-2: Load YAML, convert to abstract, and sample 5 records.
        """
        self.log("=" * 70)
        self.log("STEP 1-2: Load YAML and Sample Data")
        self.log("=" * 70)

        try:
            # Check if YAML file exists
            if not self.yaml_path.exists():
                raise FileNotFoundError(f"YAML file not found: {self.yaml_path}")

            self.log(f"Loading YAML pipeline from: {self.yaml_path}")

            # Convert YAML to abstract operators
            self.abstract_operators = yaml_to_abstract_pipeline(
                self.yaml_path,
                verbose=False
            )

            self.log(f"Converted {len(self.abstract_operators)} operators to abstract representation")

            # Display operator information
            for i, op in enumerate(self.abstract_operators):
                self.log(f"  Operator {i}: {op.name} (type: {op.type})")

            # Load complete pipeline config
            import yaml
            with open(self.yaml_path, 'r') as f:
                self.pipeline_config = yaml.safe_load(f)

            # Initialize DatasetManager
            self.data_manager = DatasetManager(
                temp_dir=self.sample_data_dir,
                keep_temp_files=True,
                verbose=False
            )

            # Extract dataset path from pipeline config
            datasets = self.pipeline_config.get('datasets', {})
            dataset_name, dataset_config = list(datasets.items())[0]
            dataset_path = dataset_config.get('path')

            # Resolve dataset path (relative to examples directory)
            self.original_dataset_path = self.examples_dir / dataset_path

            if not self.original_dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.original_dataset_path}")

            self.log(f"Loading dataset: {dataset_name} from {self.original_dataset_path}")

            # Load dataset
            dataset = self.data_manager.read_dataset(self.original_dataset_path)
            self.log(f"Dataset loaded: {len(dataset)} records")

            # Sample 5 records
            sampling_config = SamplingConfig(
                mode="fixed",
                size=5,
                random_seed=42
            )

            self.log(f"Sampling 5 records with seed=42")

            sampled_output_path = self.sample_data_dir / "sampled_medical_transcripts.json"
            self.sampled_dataset_path = self.data_manager.sample_dataset(
                source_path=self.original_dataset_path,
                config=sampling_config,
                output_path=sampled_output_path
            )

            self.log(f"Sampled dataset saved to: {self.sampled_dataset_path}")

            # Verify sampled data
            sampled_data = self.data_manager.read_dataset(self.sampled_dataset_path)
            self.log(f"Sampled {len(sampled_data)} records successfully")

            for i, record in enumerate(sampled_data, 1):
                file_id = record.get('file', 'N/A')
                self.log(f"  Sample {i}: {file_id}")

            self.log("✓ Steps 1-2 completed successfully\n")
            return True

        except Exception as e:
            self.log(f"✗ Steps 1-2 failed: {e}", level="ERROR")
            import traceback
            self.log(traceback.format_exc(), level="ERROR")
            return False

    def step_3_execute_subplan_1(self):
        """
        Step 3: Execute first 2 operators (extract_medications, unnest_medications).
        Output to subplan_1.json
        """
        self.log("=" * 70)
        self.log("STEP 3: Execute Subplan 1 (Operators 0-1)")
        self.log("=" * 70)

        try:
            if not self.abstract_operators:
                raise RuntimeError("Abstract operators not loaded")

            if not self.sampled_dataset_path:
                raise RuntimeError("Sampled dataset not available")

            # Create pipeline config with operators
            pipeline_data = {
                'operators': [operator_to_dict(op) for op in self.abstract_operators]
            }

            # Load sampled input data
            input_data = self.data_manager.read_dataset(self.sampled_dataset_path)
            self.log(f"Input: {len(input_data)} records from sampled dataset")

            # Create executor
            executor = AbstractExecutor(verbose=self.verbose, cache_enabled=True)

            # Execute operators 0-1 (extract_medications, unnest_medications)
            self.log("Executing operators 0-1: extract_medications, unnest_medications")

            result = executor.execute_pipeline_range(
                pipeline=pipeline_data,
                input_data=input_data,
                start_index=0,
                end_index=1,
                config={
                    'default_model': self.pipeline_config.get('default_model', 'gpt-4o-mini'),
                    'system_prompt': self.pipeline_config.get('system_prompt', {})
                }
            )

            if not result.success:
                raise RuntimeError(f"Subplan 1 execution failed: {result.error}")

            # Save output to subplan_1.json
            subplan_1_path = self.output_dir / "subplan_1.json"
            with open(subplan_1_path, 'w') as f:
                json.dump(result.data, f, indent=2)

            self.subplan_1_output = result.data

            self.log(f"✓ Subplan 1 completed successfully")
            self.log(f"  Execution time: {result.metadata.get('execution_time', 0):.2f}s")
            self.log(f"  Operators executed: {result.metadata.get('operators_executed', 0)}")
            self.log(f"  Output records: {len(result.data) if isinstance(result.data, list) else 'N/A'}")
            self.log(f"  Output saved to: {subplan_1_path}")
            self.log("✓ Step 3 completed successfully\n")
            return True

        except Exception as e:
            self.log(f"✗ Step 3 failed: {e}", level="ERROR")
            import traceback
            self.log(traceback.format_exc(), level="ERROR")
            return False

    def step_4_execute_subplan_2(self):
        """
        Step 4: Execute remaining operators (resolve_medications, summarize_prescriptions).
        Input from subplan_1.json, output to subplan_2.json
        """
        self.log("=" * 70)
        self.log("STEP 4: Execute Subplan 2 (Operators 2-3)")
        self.log("=" * 70)

        try:
            if not self.abstract_operators:
                raise RuntimeError("Abstract operators not loaded")

            if not self.subplan_1_output:
                raise RuntimeError("Subplan 1 output not available")

            # Create pipeline config with operators
            pipeline_data = {
                'operators': [operator_to_dict(op) for op in self.abstract_operators]
            }

            # Use output from subplan 1 as input
            input_data = self.subplan_1_output
            self.log(f"Input: {len(input_data)} records from subplan_1.json")

            # Create executor
            executor = AbstractExecutor(verbose=self.verbose, cache_enabled=True)

            # Execute operators 2-3 (resolve_medications, summarize_prescriptions)
            self.log("Executing operators 2-3: resolve_medications, summarize_prescriptions")

            result = executor.execute_pipeline_range(
                pipeline=pipeline_data,
                input_data=input_data,
                start_index=2,
                end_index=3,
                config={
                    'default_model': self.pipeline_config.get('default_model', 'gpt-4o-mini'),
                    'system_prompt': self.pipeline_config.get('system_prompt', {})
                }
            )

            if not result.success:
                raise RuntimeError(f"Subplan 2 execution failed: {result.error}")

            # Save output to subplan_2.json
            subplan_2_path = self.output_dir / "subplan_2.json"
            with open(subplan_2_path, 'w') as f:
                json.dump(result.data, f, indent=2)

            self.subplan_2_output = result.data

            self.log(f"✓ Subplan 2 completed successfully")
            self.log(f"  Execution time: {result.metadata.get('execution_time', 0):.2f}s")
            self.log(f"  Operators executed: {result.metadata.get('operators_executed', 0)}")
            self.log(f"  Output records: {len(result.data) if isinstance(result.data, list) else 'N/A'}")
            self.log(f"  Output saved to: {subplan_2_path}")
            self.log("✓ Step 4 completed successfully\n")
            return True

        except Exception as e:
            self.log(f"✗ Step 4 failed: {e}", level="ERROR")
            import traceback
            self.log(traceback.format_exc(), level="ERROR")
            return False

    def run_all_steps(self):
        """Run all test steps in sequence."""
        self.log("\n" + "=" * 70)
        self.log("SUBPLAN PIPELINE EXECUTION TEST: example2.yaml")
        self.log("=" * 70)
        self.log(f"Test directory: {self.test_dir}")
        self.log(f"YAML file: {self.yaml_path}")
        self.log("=" * 70 + "\n")

        steps = [
            ("Load YAML and Sample Data", self.step_1_load_and_sample),
            ("Execute Subplan 1 (Operators 0-1)", self.step_3_execute_subplan_1),
            ("Execute Subplan 2 (Operators 2-3)", self.step_4_execute_subplan_2),
        ]

        results = {}

        for step_name, step_func in steps:
            try:
                success = step_func()
                results[step_name] = success

                if not success:
                    self.log(f"Stopping test execution due to failure in: {step_name}", level="ERROR")
                    break

            except Exception as e:
                self.log(f"Unexpected error in {step_name}: {e}", level="ERROR")
                results[step_name] = False
                break

        # Print summary
        self.log("\n" + "=" * 70)
        self.log("TEST EXECUTION SUMMARY")
        self.log("=" * 70)

        for step_name, success in results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            self.log(f"{status}: {step_name}")

        total_steps = len(results)
        passed_steps = sum(1 for success in results.values() if success)

        self.log("=" * 70)
        self.log(f"Total: {passed_steps}/{total_steps} steps passed")
        self.log("=" * 70 + "\n")

        # Show output files if all passed
        if all(results.values()):
            self.log("Output files:")
            self.log(f"  - Sampled data: {self.sample_data_dir / 'sampled_medical_transcripts.json'}")
            self.log(f"  - Subplan 1 output: {self.output_dir / 'subplan_1.json'}")
            self.log(f"  - Subplan 2 output: {self.output_dir / 'subplan_2.json'}")

            # Show sample output from subplan 2
            if self.subplan_2_output:
                self.log(f"\nFinal output preview (first record):")
                if isinstance(self.subplan_2_output, list) and len(self.subplan_2_output) > 0:
                    first_record = self.subplan_2_output[0]
                    self.log(f"  Fields: {list(first_record.keys())}")
                    if 'medication' in first_record:
                        self.log(f"  Medication: {first_record.get('medication', 'N/A')}")

        return all(results.values())

    def cleanup(self):
        """Cleanup resources."""
        if self.data_manager:
            self.log("Keeping test files for inspection (keep_temp_files=True)")


def main():
    """Main entry point."""
    runner = SubplanTestRunner(verbose=True)

    try:
        success = runner.run_all_steps()

        if success:
            print("\n🎉 All tests passed successfully!")
            print(f"\nOutput files created:")
            print(f"  - {runner.sample_data_dir / 'sampled_medical_transcripts.json'}")
            print(f"  - {runner.output_dir / 'subplan_1.json'}")
            print(f"  - {runner.output_dir / 'subplan_2.json'}")
            return 0
        else:
            print("\n❌ Tests failed. Check logs above for details.")
            return 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        runner.cleanup()


if __name__ == "__main__":
    sys.exit(main())
