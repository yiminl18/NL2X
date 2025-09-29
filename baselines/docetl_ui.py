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

        print("\n➡️  Execute this pipeline? (Y/n): ", end="")
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

    def confirm_step_before_llm(self, step_name: str, step_number: str, total_steps: str = "5") -> str:
        """
        Ask for confirmation before calling LLM for a step.

        Args:
            step_name: Name of the step
            step_number: Current step number (e.g., "1", "3-4")
            total_steps: Total number of steps

        Returns:
            'continue' if user wants to continue
            'regenerate' if user wants to regenerate (bypass cache)
            'abort' if user wants to abort
        """
        if not (self.config.confirm or self.config.debug):
            return 'continue'

        print(f"\n➡️  Next: Step {step_number}/{total_steps} - {step_name}")
        print("Continue? (Y/r/n): ", end="")
        user_input = input().strip().lower()

        if user_input == 'r':
            print("🔄 Regenerating response (bypassing cache)...")
            return 'regenerate'
        elif user_input == 'n':
            print("❌ User aborted pipeline generation")
            return 'abort'
        else:  # Default to 'y' or empty input
            return 'continue'

    def confirm_operator_before_llm(self, operator_index: int, total_operators: int,
                                    operator_type: str, operator_purpose: str,
                                    prompt: str, prompt_file: str) -> str:
        """
        Ask for confirmation before generating a single operator in Step 4.

        Args:
            operator_index: Current operator index (0-based)
            total_operators: Total number of operators to generate
            operator_type: Type of the operator (map, filter, etc.)
            operator_purpose: Purpose of the operator
            prompt: The prompt that will be sent to LLM
            prompt_file: Path where the prompt was saved

        Returns:
            'continue' if user wants to continue
            'regenerate' if user wants to regenerate (bypass cache)
            'abort' if user wants to abort
        """
        if not (self.config.confirm or self.config.debug):
            return 'continue'

        print("\n" + "="*80)
        mode_text = "[DEBUG MODE]" if self.config.debug else "[CONFIRM MODE]"
        print(f"{mode_text} Step 4 - Operator {operator_index + 1}/{total_operators}")
        print("="*80)

        print(f"\n📋 Operator Type: {operator_type}")
        print(f"📋 Purpose: {operator_purpose}")
        print("-"*40)

        print(f"\n📄 Prompt saved to: {prompt_file}")
        print("-"*40)

        # Show prompt preview (first 500 chars)
        prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
        print("\n📝 Prompt Preview:")
        print("-"*40)
        print(prompt_preview)
        print("-"*40)

        print("\n➡️  Generate this operator? (Y/r/n): ", end="")
        user_input = input().strip().lower()

        if user_input == 'r':
            print("🔄 Regenerating operator (bypassing cache)...")
            return 'regenerate'
        elif user_input == 'n':
            print("❌ User aborted operator generation")
            return 'abort'
        else:  # Default to 'y' or empty input
            return 'continue'

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
        print(f"\n➡️  Next: {next_step}. Continue to next step? (Y/n): ", end="")
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