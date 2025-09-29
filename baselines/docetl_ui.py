from typing import Any, Dict, List


class DocETLUserInterface:
    """
    Handles user interactions for DocETL step-by-step pipeline generation.

    This class manages confirmation dialogs, step previews, and user input
    for debug and confirm modes during pipeline generation.
    """

    def __init__(self, config):
        self.config = config

    def confirm_pipeline_execution(self, pipeline_file: str, query: str, attempt: int) -> bool:
        """Ask user for confirmation before executing pipeline in confirm/debug mode."""
        if not (self.config.confirm or self.config.debug):
            return True

        print("\n" + "="*80)
        mode_text = "[DEBUG MODE]" if self.config.debug else "[CONFIRM MODE]"
        print(f"{mode_text} Pipeline Generated - Attempt {attempt + 1}")
        print("="*80)

        print(f"\n📋 Query:")
        print("-"*40)
        print(query)
        print("-"*40)

        print(f"\n📄 Generated Pipeline File:")
        print(f"  {pipeline_file}")

        print(f"\n📝 Pipeline Preview (first 50 lines):")
        print("-"*40)
        try:
            with open(pipeline_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:50]):
                    print(f"{i+1:3d}: {line.rstrip()}")
                if len(lines) > 50:
                    print(f"... ({len(lines) - 50} more lines)")
        except Exception as e:
            print(f"Error reading pipeline file: {e}")
        print("-"*40)

        print("\n⚠️  Execute this pipeline? (Y/n): ", end="")
        user_input = input().strip().lower()

        if user_input and user_input != 'y':
            print("❌ Pipeline execution skipped by user")
            return False

        print("✅ Proceeding with pipeline execution...")
        return True

    def _get_next_step_info(self, current_step: str) -> str:
        """Get information about the next step."""
        step_mapping = {
            "Step 1: Operator Selection": "Step 2/5 - Operator Framework Creation",
            "Step 2: Operator Framework": "Step 3-4/5 - Operator Details Generation",
            "Step 3-4: Operator Details": "Step 5/5 - Pipeline Connection",
            "Step 5: Pipeline Connection": "Pipeline Execution"
        }

        # Handle step names that might start with the key
        for key, value in step_mapping.items():
            if current_step.startswith(key):
                return value

        return "Next step"

    def confirm_step_execution(self, step_name: str, step_data: Any, query: str, attempt: int) -> bool:
        """
        Ask user for confirmation after each step in confirm/debug mode.

        Args:
            step_name: Name of the step just completed
            step_data: Data generated in this step
            query: The original query
            attempt: Current attempt number

        Returns:
            True if user wants to continue, False to abort
        """
        if not (self.config.confirm or self.config.debug):
            return True

        print("\n" + "="*80)
        mode_text = "[DEBUG MODE]" if self.config.debug else "[CONFIRM MODE]"
        print(f"{mode_text} Step Completed - Attempt {attempt + 1}")
        print("="*80)

        print(f"\n📋 Query:")
        print("-"*40)
        print(query)
        print("-"*40)

        print(f"\n✅ Completed Step: {step_name}")
        print("-"*40)

        # Display step-specific data
        if step_name == "Step 1: Operator Selection":
            print("Selected Operators:")
            for i, op in enumerate(step_data, 1):
                print(f"  {i}. {op['type']}: {op['purpose']}")

        elif step_name == "Step 2: Operator Framework":
            print("Created Frameworks:")
            for framework in step_data:
                print(f"  - {framework['name']} ({framework['type']})")
                print(f"    Purpose: {framework['purpose']}")

                # Show complete framework structure with TO_BE_GENERATED placeholders
                print("    Framework structure:")
                for key, value in framework.items():
                    if key not in ['name', 'type', 'purpose']:
                        if isinstance(value, dict):
                            print(f"      {key}: {value}")
                        elif isinstance(value, list):
                            print(f"      {key}: {value}")
                        else:
                            print(f"      {key}: {value}")
                print()

        elif step_name.startswith("Step 3-4: Operator Details"):
            print("Filled Operators:")
            for op in step_data:
                print(f"  - {op['name']} ({op['type']})")

                # Show prompts for operators that use prompts
                if 'prompt' in op:
                    print(f"    Prompt: {op['prompt']}")

                # Show code for code operators
                if op['type'] in ['code_map', 'code_filter'] and 'code' in op:
                    print(f"    Code: {op['code']}")

                # Show different details based on operator type
                if op['type'] == 'extract':
                    if 'document_keys' in op:
                        print(f"    Document keys: {op['document_keys']}")
                    # Extract operators don't use output schema, they extract to predefined fields
                    expected_fields = self._parse_operator_output_fields(op)
                    print(f"    Expected output fields: {expected_fields}")
                elif 'output' in op and 'schema' in op['output']:
                    print(f"    Output schema: {list(op['output']['schema'].keys()) if op['output']['schema'] else 'empty'}")

        elif step_name == "Step 5: Pipeline Connection":
            print("Final Pipeline Preview:")
            lines = step_data.split('\n')
            for line in lines:
                print(f"  {line}")

        print("-"*40)

        next_step = self._get_next_step_info(step_name)
        print(f"\n⚠️  Next: {next_step}. Continue to next step? (Y/n): ", end="")
        user_input = input().strip().lower()

        if user_input and user_input != 'y':
            print("❌ Pipeline generation aborted by user")
            return False

        print("✅ Proceeding to next step...")
        return True

    def _parse_operator_output_fields(self, operator: Dict[str, Any]) -> List[str]:
        """
        Parse output fields from an operator's configuration.

        Returns a list of field names that this operator will add to the data.
        """
        output_fields = []

        # Special handling for extract operator
        if operator['type'] == 'extract':
            if 'document_keys' in operator and operator['document_keys'] != "TO_BE_GENERATED":
                if isinstance(operator['document_keys'], list) and operator['document_keys']:
                    for doc_key in operator['document_keys']:
                        suffix = operator.get('extraction_key_suffix', f"_extracted_{operator.get('name', 'extract')}")
                        output_fields.append(f"{doc_key}{suffix}")
                else:
                    # Fallback TODO Can be optimized.
                    suffix = operator.get('extraction_key_suffix', f"_extracted_{operator.get('name', 'extract')}")
                    output_fields.append(f"src{suffix}")
            return output_fields

        if 'output' in operator and isinstance(operator['output'], dict):
            schema = operator['output'].get('schema', {})

            if isinstance(schema, dict):
                output_fields.extend(schema.keys())
            elif isinstance(schema, str):
                import re
                matches = re.findall(r'(\w+)\s*:\s*\w+', schema)
                output_fields.extend(matches)

        if operator['type'] == 'unnest' and 'unnest_key' in operator:
            pass # Fields remain the same, just expanded # TODO Need fix for dict unnest
        elif operator['type'] == 'split':
            output_fields.append('_split_id')
            output_fields.append('_split_index')
        elif operator['type'] == 'gather':
            if 'output_key' in operator:
                output_fields.append(operator['output_key'])

        return output_fields

    def format_query_as_comments(self, query: str) -> str:
        """
        Format a potentially multi-line query as YAML comments.

        Args:
            query: The query string, which may contain multiple lines

        Returns:
            Formatted string with each line prefixed by '# '
        """
        lines = query.strip().split('\n')
        commented_lines = []

        for i, line in enumerate(lines):
            if i == 0:
                commented_lines.append(f"# Query: {line}")
            else:
                commented_lines.append(f"# {line}")

        return '\n'.join(commented_lines)