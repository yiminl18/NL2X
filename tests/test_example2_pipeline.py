"""
Test Script for Abstract Layer Pipeline Execution
Based on examples/example2.yaml

This script demonstrates the complete workflow:
1. Load YAML and transform to abstract layer using function calls
2. Use DatasetManager to load the dataset from the transferred yaml
3. Get 5 sample records and store them in a temp location
4. Change the pipeline to run on sample data with output to tests/
5. Execute the pipeline and collect all intermediate results

Uses function calls rather than file I/O as much as possible.

Usage:
    # Option 1: Run directly (if executable)
    ./tests/test_example2_pipeline.py
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
    yaml_to_abstract_json,
    operator_to_dict,
    set_dataset_path,
    set_output_path,
    set_intermediate_dir,
    get_all_file_paths
)


class TestPipelineRunner:
    """Test runner for example2.yaml pipeline with sampling and tracking."""

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.test_dir = SCRIPT_DIR
        self.examples_dir = EXAMPLES_DIR
        self.yaml_path = self.examples_dir / "example2.yaml"

        # Create output directories
        self.sample_data_dir = self.test_dir / "sample_data"
        self.output_dir = self.test_dir / "output"
        self.intermediate_dir = self.test_dir / "intermediates"

        for dir_path in [self.sample_data_dir, self.output_dir, self.intermediate_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Storage for test artifacts
        self.abstract_operators = None
        self.pipeline_config = None
        self.data_manager = None
        self.original_dataset_path = None
        self.sampled_dataset_path = None
        self.execution_result = None

    def log(self, message, level="INFO"):
        """Print log message with timestamp."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")

    def step_1_load_and_convert_yaml(self):
        """
        Step 1: Load YAML and transform to abstract layer using function calls.
        """
        self.log("=" * 70)
        self.log("STEP 1: Load YAML and Convert to Abstract Layer")
        self.log("=" * 70)

        try:
            # Check if YAML file exists
            if not self.yaml_path.exists():
                raise FileNotFoundError(f"YAML file not found: {self.yaml_path}")

            self.log(f"Loading YAML pipeline from: {self.yaml_path}")

            # Convert YAML to abstract operators
            self.abstract_operators = yaml_to_abstract_pipeline(
                self.yaml_path,
                verbose=self.verbose
            )

            self.log(f"Converted {len(self.abstract_operators)} operators to abstract representation")

            # Display operator information
            for i, op in enumerate(self.abstract_operators, 1):
                self.log(f"  Operator {i}: {op.name} (type: {op.type})")

            # Load complete pipeline config as dict
            import yaml
            with open(self.yaml_path, 'r') as f:
                self.pipeline_config = yaml.safe_load(f)

            self.log(f"Pipeline config loaded: {self.pipeline_config.get('default_model', 'N/A')} model")
            self.log("✓ Step 1 completed successfully\n")
            return True

        except Exception as e:
            self.log(f"✗ Step 1 failed: {e}", level="ERROR")
            import traceback
            self.log(traceback.format_exc(), level="ERROR")
            return False

    def step_2_load_dataset_with_manager(self):
        """
        Step 2: Use DatasetManager to load the dataset from the transferred yaml.
        """
        self.log("=" * 70)
        self.log("STEP 2: Load Dataset with DatasetManager")
        self.log("=" * 70)

        try:
            # Initialize DatasetManager
            self.data_manager = DatasetManager(
                temp_dir=self.sample_data_dir,
                keep_temp_files=True,  # Keep files for inspection
                verbose=self.verbose
            )

            self.log("DatasetManager initialized")

            # Extract dataset path from pipeline config
            datasets = self.pipeline_config.get('datasets', {})
            if not datasets:
                raise ValueError("No datasets found in pipeline configuration")

            # Get the first dataset (transcripts)
            dataset_name, dataset_config = list(datasets.items())[0]
            dataset_path = dataset_config.get('path')

            if not dataset_path:
                raise ValueError(f"No path found for dataset: {dataset_name}")

            # Resolve dataset path (relative to examples directory)
            self.original_dataset_path = self.examples_dir / dataset_path

            if not self.original_dataset_path.exists():
                raise FileNotFoundError(f"Dataset not found: {self.original_dataset_path}")

            self.log(f"Loading dataset: {dataset_name} from {self.original_dataset_path}")

            # Load dataset using DatasetManager
            dataset = self.data_manager.read_dataset(self.original_dataset_path)

            self.log(f"Dataset loaded: {len(dataset)} records")

            # Display sample record structure
            if dataset:
                self.log(f"Record fields: {list(dataset[0].keys())}")

            self.log("✓ Step 2 completed successfully\n")
            return True

        except Exception as e:
            self.log(f"✗ Step 2 failed: {e}", level="ERROR")
            import traceback
            self.log(traceback.format_exc(), level="ERROR")
            return False

    def step_3_sample_and_store_records(self):
        """
        Step 3: Get 5 sample records and store them in a temp location.
        """
        self.log("=" * 70)
        self.log("STEP 3: Sample 5 Records and Store")
        self.log("=" * 70)

        try:
            if not self.data_manager:
                raise RuntimeError("DatasetManager not initialized")

            if not self.original_dataset_path:
                raise RuntimeError("Original dataset path not set")

            # Create sampling configuration for 5 records
            sampling_config = SamplingConfig(
                mode="fixed",
                size=5,
                random_seed=42  # For reproducibility
            )

            self.log(f"Sampling configuration: mode={sampling_config.mode}, size={sampling_config.size}")

            # Sample dataset
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

            # Display sample record IDs if available
            for i, record in enumerate(sampled_data, 1):
                file_id = record.get('file', 'N/A')
                self.log(f"  Sample {i}: {file_id}")

            # Register path mapping
            path_mappings = self.data_manager.get_path_mappings()
            self.log(f"Path mapping registered: {len(path_mappings)} mapping(s)")

            self.log("✓ Step 3 completed successfully\n")
            return True

        except Exception as e:
            self.log(f"✗ Step 3 failed: {e}", level="ERROR")
            import traceback
            self.log(traceback.format_exc(), level="ERROR")
            return False

    def step_4_modify_pipeline_for_sample_data(self):
        """
        Step 4: Change the pipeline to run on sample data and store output to tests/.
        """
        self.log("=" * 70)
        self.log("STEP 4: Modify Pipeline for Sample Data")
        self.log("=" * 70)

        try:
            if not self.pipeline_config:
                raise RuntimeError("Pipeline config not loaded")

            if not self.sampled_dataset_path:
                raise RuntimeError("Sampled dataset path not set")

            # Update dataset path in pipeline config
            original_dataset_name = list(self.pipeline_config.get('datasets', {}).keys())[0]
            original_dataset_path = self.pipeline_config['datasets'][original_dataset_name]['path']

            self.log(f"Original dataset path: {original_dataset_path}")

            # Update to sampled dataset path
            self.pipeline_config = set_dataset_path(
                self.pipeline_config,
                original_dataset_name,
                str(self.sampled_dataset_path)
            )

            self.log(f"Updated dataset path to: {self.sampled_dataset_path}")

            # Update output path
            original_output_config = self.pipeline_config.get('pipeline', {}).get('output', {})
            original_output_path = original_output_config.get('path', 'medication_summaries.json')

            # Set new output path
            new_output_path = str(self.output_dir / "medication_summaries.json")
            self.pipeline_config = set_output_path(
                self.pipeline_config,
                new_output_path
            )

            self.log(f"Original output path: {original_output_path}")
            self.log(f"Updated output path to: {new_output_path}")

            # Set intermediate directory
            original_intermediate_dir = original_output_config.get('intermediate_dir', 'intermediate_results')
            new_intermediate_dir = str(self.intermediate_dir)

            self.pipeline_config = set_intermediate_dir(
                self.pipeline_config,
                new_intermediate_dir
            )

            self.log(f"Original intermediate dir: {original_intermediate_dir}")
            self.log(f"Updated intermediate dir to: {new_intermediate_dir}")

            # Register output path with DatasetManager
            self.data_manager.register_output_path(
                original_output_path,
                new_output_path
            )

            self.log("✓ Step 4 completed successfully\n")
            return True

        except Exception as e:
            self.log(f"✗ Step 4 failed: {e}", level="ERROR")
            import traceback
            self.log(traceback.format_exc(), level="ERROR")
            return False

    def step_5_execute_pipeline_with_tracking(self):
        """
        Step 5: Execute the pipeline and collect all intermediate results.
        """
        self.log("=" * 70)
        self.log("STEP 5: Execute Pipeline with Intermediate Tracking")
        self.log("=" * 70)

        try:
            if not self.pipeline_config:
                raise RuntimeError("Pipeline config not loaded")

            if not self.data_manager:
                raise RuntimeError("DatasetManager not initialized")

            # Create AbstractExecutor with DatasetManager
            executor = AbstractExecutor(
                verbose=self.verbose,
                cache_enabled=True,
                data_manager=self.data_manager
            )

            self.log("AbstractExecutor initialized with DatasetManager")

            # Save modified pipeline config to temp file for execution
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                # Convert pipeline config to abstract JSON format
                pipeline_json = {
                    "operators": [operator_to_dict(op) for op in self.abstract_operators]
                }

                # Add other config fields
                for key, value in self.pipeline_config.items():
                    if key != "operations":
                        pipeline_json[key] = value

                json.dump(pipeline_json, f)
                temp_pipeline_path = f.name

            self.log(f"Temporary pipeline config saved to: {temp_pipeline_path}")

            # Execute pipeline with tracking
            self.log("Starting pipeline execution...")
            self.log("This may take a while depending on the LLM API...")

            self.execution_result = executor.execute_with_tracking(
                pipeline=temp_pipeline_path,
                input_data=None,  # Already set in pipeline config
                output_dir=str(self.intermediate_dir),
                config={
                    'default_model': self.pipeline_config.get('default_model', 'gpt-4o-mini')
                }
            )

            # Clean up temp file
            os.unlink(temp_pipeline_path)

            # Process execution result
            if self.execution_result.success:
                self.log("✓ Pipeline execution completed successfully!")

                # Display execution metadata
                exec_time = self.execution_result.metadata.get('execution_time', 0)
                self.log(f"Execution time: {exec_time:.2f}s")

                # Display intermediate results
                intermediate_results = self.execution_result.metadata.get('intermediate_results', {})
                if intermediate_results:
                    self.log(f"\nIntermediate results collected:")
                    for step_name, operations in intermediate_results.items():
                        self.log(f"  Step: {step_name}")
                        for op_name, data in operations.items():
                            record_count = len(data) if isinstance(data, list) else "N/A"
                            self.log(f"    Operation '{op_name}': {record_count} records")
                else:
                    self.log("No intermediate results metadata found")

                # Display final output summary
                final_data = self.execution_result.data
                if isinstance(final_data, list):
                    self.log(f"\nFinal output: {len(final_data)} records")

                    # Show first record summary if available
                    if final_data:
                        first_record = final_data[0]
                        self.log(f"First record fields: {list(first_record.keys())}")
                else:
                    self.log(f"\nFinal output type: {type(final_data)}")

                # List all files created in output and intermediate directories
                self.log(f"\nOutput files created:")
                output_files = list(self.output_dir.glob("*"))
                for file_path in output_files:
                    if file_path.is_file():
                        self.log(f"  - {file_path.name} ({file_path.stat().st_size} bytes)")

                self.log(f"\nIntermediate files created:")
                intermediate_files = list(self.intermediate_dir.glob("**/*"))
                for file_path in intermediate_files:
                    if file_path.is_file():
                        relative_path = file_path.relative_to(self.intermediate_dir)
                        self.log(f"  - {relative_path} ({file_path.stat().st_size} bytes)")

                self.log("\n✓ Step 5 completed successfully\n")
                return True
            else:
                self.log(f"✗ Pipeline execution failed: {self.execution_result.error}", level="ERROR")
                return False

        except Exception as e:
            self.log(f"✗ Step 5 failed: {e}", level="ERROR")
            import traceback
            self.log(traceback.format_exc(), level="ERROR")
            return False

    def run_all_steps(self):
        """Run all test steps in sequence."""
        self.log("\n" + "=" * 70)
        self.log("TEST PIPELINE EXECUTION: example2.yaml")
        self.log("=" * 70)
        self.log(f"Test directory: {self.test_dir}")
        self.log(f"YAML file: {self.yaml_path}")
        self.log("=" * 70 + "\n")

        steps = [
            ("Load and Convert YAML", self.step_1_load_and_convert_yaml),
            ("Load Dataset with Manager", self.step_2_load_dataset_with_manager),
            ("Sample 5 Records", self.step_3_sample_and_store_records),
            ("Modify Pipeline for Sample Data", self.step_4_modify_pipeline_for_sample_data),
            ("Execute Pipeline with Tracking", self.step_5_execute_pipeline_with_tracking),
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

        return all(results.values())

    def cleanup(self):
        """Cleanup resources."""
        if self.data_manager:
            # Don't cleanup since we want to keep files for inspection
            self.log("Keeping test files for inspection (keep_temp_files=True)")


def main():
    """Main entry point."""
    runner = TestPipelineRunner(verbose=True)

    try:
        success = runner.run_all_steps()

        if success:
            print("\n🎉 All tests passed successfully!")
            print(f"\nTest artifacts saved to:")
            print(f"  - Sample data: {runner.sample_data_dir}")
            print(f"  - Output: {runner.output_dir}")
            print(f"  - Intermediates: {runner.intermediate_dir}")
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
