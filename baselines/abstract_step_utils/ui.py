"""
User Interface for Abstract Step Pipeline Generation

Handles user interactions for abstract layer step-by-step pipeline generation,
including confirmation dialogs, operator display, and progress feedback.
"""

import json
from typing import Any, Dict, List


class AbstractStepUserInterface:
    """
    Handles user interactions for Abstract Step pipeline generation.

    This class manages confirmation dialogs, operator previews, and user input
    for debug and confirm modes during abstract pipeline generation.
    """

    def __init__(self, config):
        """
        Initialize the user interface.

        Args:
            config: Configuration object with confirm and debug flags
        """
        self.config = config

    def confirm_step_before_llm(self, step_name: str, step_number: str, total_steps: str = "2", step_prompt: str = "") -> str:
        """
        Ask for confirmation before calling LLM for a step.

        Args:
            step_name: Name of the step
            step_number: Current step number
            total_steps: Total number of steps
            step_prompt: Prompt for the current step

        Returns:
            'continue' if user wants to continue
            'regenerate' if user wants to regenerate (bypass cache)
            'abort' if user wants to abort
        """
        if not (self.config.confirm or self.config.debug):
            return 'continue'

        if step_prompt:
            print(f"\n📝 Step Prompt:")
            print("-"*40)
            print(step_prompt)
            print("-"*40)

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

    def confirm_step_execution(self, step_name: str, result: Any, query: str, attempt: int) -> bool:
        """
        Display step result and ask for confirmation to continue.

        Args:
            step_name: Name of the completed step
            result: Result from the step
            query: Original query
            attempt: Attempt number

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
        print(query[:200] + "..." if len(query) > 200 else query)
        print("-"*40)

        print(f"\n✅ Completed Step: {step_name}")
        print("-"*40)

        # Display step-specific data based on step name
        if "Step 1" in step_name or "Operator Selection" in step_name:
            # Step 1: Operator Selection
            if isinstance(result, list):
                print(f"Selected {len(result)} operators:")
                for i, op in enumerate(result, 1):
                    if isinstance(op, dict):
                        op_type = op.get('type', 'Unknown')
                        purpose = op.get('purpose', 'N/A')
                        print(f"  {i}. {op_type}: {purpose}")
                    else:
                        print(f"  {i}. {op}")
            else:
                print(result)

        elif "Step 2" in step_name or "Operator Details" in step_name:
            # Step 2: Operator Details Generation
            # Note: This is typically handled by display_generated_operator for each operator
            # But if called for the whole step, display summary
            if isinstance(result, list):
                print(f"Generated {len(result)} operators:")
                for i, op in enumerate(result, 1):
                    if isinstance(op, dict):
                        op_name = op.get('name', f'op_{i}')
                        op_type = op.get('type', 'Unknown')
                        print(f"  {i}. {op_name} ({op_type})")

                        # Show key fields based on operator type
                        if op.get('type') in ['map', 'filter', 'reduce']:
                            if 'prompt' in op:
                                prompt_preview = op['prompt'][:100] + "..." if len(op.get('prompt', '')) > 100 else op.get('prompt', '')
                                print(f"     Prompt: {prompt_preview}")

                        # Show output schema if available
                        output_fields = self._parse_operator_output_fields(op)
                        if output_fields:
                            print(f"     Output fields: {', '.join(output_fields)}")
                    else:
                        print(f"  {i}. {op}")
            else:
                print(result)

        else:
            # Generic display for other steps
            if isinstance(result, list):
                print(f"Generated {len(result)} items:")
                for i, item in enumerate(result, 1):
                    if isinstance(item, dict):
                        item_type = item.get('type', 'Unknown')
                        item_name = item.get('name', item.get('purpose', 'N/A'))
                        print(f"  {i}. {item_type}: {item_name}")
                    else:
                        print(f"  {i}. {item}")
            else:
                print(result)

        print("-"*40)

        # Get next step info
        next_step = self._get_next_step_info(step_name)
        print(f"\n➡️  Next: {next_step}. Continue to next step? (Y/n): ", end="")
        user_input = input().strip().lower()

        if user_input and user_input != 'y':
            print("❌ Pipeline generation aborted by user")
            return False

        print("✅ Proceeding to next step...")
        return True

    def confirm_pipeline_execution(self, pipeline_file: str, query: str, attempt: int, validation_passed: bool = True) -> bool:
        """
        Ask user for confirmation before executing pipeline in confirm/debug mode.

        Args:
            pipeline_file: Path to the generated pipeline YAML file
            query: Original query
            attempt: Attempt number
            validation_passed: Whether static validation passed (defaults to True for backward compatibility)

        Returns:
            True if user wants to continue, False to abort
        """
        if not (self.config.confirm or self.config.debug):
            return True

        print("\n" + "="*80)
        mode_text = "[DEBUG MODE]" if self.config.debug else "[CONFIRM MODE]"
        print(f"{mode_text} Pipeline Generated - Attempt {attempt + 1}")
        print("="*80)

        print(f"\n📋 Query:")
        print("-"*40)
        print(query[:200] + "..." if len(query) > 200 else query)
        print("-"*40)

        print(f"\n📄 Generated Pipeline File:")
        print(f"  {pipeline_file}")

        # Adjust prompt based on validation status
        if validation_passed:
            print("\n➡️  Execute this pipeline? (Y/n): ", end="")
            default_is_yes = True
        else:
            print("\n⚠️  WARNING: Pipeline has validation errors!")
            print("➡️  Execute this pipeline anyway? (y/N): ", end="")
            default_is_yes = False

        user_input = input().strip().lower()

        # Handle default behavior based on validation status
        if not user_input:
            # Empty input - use default
            if default_is_yes:
                print("✅ Proceeding with pipeline execution...")
                return True
            else:
                print("❌ Pipeline execution skipped (validation failed)")
                return False
        elif user_input == 'y':
            print("✅ Proceeding with pipeline execution...")
            return True
        else:
            print("❌ Pipeline execution skipped by user")
            return False

    def _get_next_step_info(self, current_step: str) -> str:
        """Get information about the next step."""
        step_mapping = {
            "Step 1: Operator Selection": "Step 2/2 - Operator Details Generation",
            "Operator Details Generation": "Pipeline Conversion and Execution"
        }

        for key, value in step_mapping.items():
            if key in current_step:
                return value

        return "Next step"

    def confirm_operator_before_llm(
        self,
        operator_index: int,
        total_operators: int,
        operator_type: str,
        operator_purpose: str,
        prompt: str,
        is_cached: bool = False,
    ) -> str:
        """
        Ask for confirmation before generating a single operator.

        Args:
            operator_index: Current operator index (0-based)
            total_operators: Total number of operators to generate
            operator_type: Type of the operator (Map, Filter, etc.)
            operator_purpose: Purpose of the operator
            prompt: The prompt that will be sent to LLM
            is_cached: Whether the prompt is already cached

        Returns:
            'continue' if user wants to continue
            'regenerate' if user wants to regenerate (bypass cache)
            'abort' if user wants to abort
        """
        if not (self.config.confirm or self.config.debug):
            return 'continue'

        print("\n" + "="*80)
        mode_text = "[DEBUG MODE]" if self.config.debug else "[CONFIRM MODE]"
        cache_text = " (CACHED)" if is_cached else ""
        print(f"{mode_text} Operator {operator_index + 1}/{total_operators} - Generation{cache_text}")
        print("="*80)

        print(f"\n📋 Operator Type: {operator_type}")
        print(f"📋 Purpose: {operator_purpose}")
        print("-"*40)

        # Show prompt preview (first 500 chars)
        print("\n📝 Prompt Preview:")
        print("-"*40)
        print(prompt)
        print("-"*40)

        # Show cache status and appropriate prompt
        if is_cached:
            print("\n💾 This prompt is CACHED. Type 'r' to bypass the cache and regenerate.")
            print("➡️  Generate this operator? (Y/r/n): ", end="")
        else:
            print("\n➡️  Generate this operator? (Y/n): ", end="")

        user_input = input().strip().lower()

        if user_input == 'r':
            if is_cached:
                print("🔄 Regenerating operator (bypassing cache)...")
                return 'regenerate'
            else:
                # If not cached and user types 'r', treat as invalid and default to continue
                print("✅ Generating operator...")
                return 'continue'
        elif user_input == 'n':
            print("❌ User aborted operator generation")
            return 'abort'
        else:  # Default to 'y' or empty input
            if is_cached:
                print("✅ Using cached response...")
            else:
                print("✅ Generating operator...")
            return 'continue'

    def display_generated_operator(
        self,
        operator_index: int,
        total_operators: int,
        operator_type: str,
        operator_config: Dict[str, Any]
    ) -> str:
        """
        Display generated operator configuration and ask for confirmation.

        Args:
            operator_index: Current operator index (0-based)
            total_operators: Total number of operators
            operator_type: Type of the operator
            operator_config: Generated operator configuration

        Returns:
            'continue' to proceed to next operator
            'regenerate' to regenerate this operator
            'abort' to stop generation
        """
        if not (self.config.confirm or self.config.debug):
            return 'continue'

        print("\n" + "="*80)
        print(f"✅ Generated Operator {operator_index + 1}/{total_operators}: {operator_type}")
        print("="*80)

        print(f"\n📋 Operator Configuration:")
        print("-"*40)

        # Format and display the configuration
        config_str = json.dumps(operator_config, indent=2, ensure_ascii=False)
        print(config_str)
        print("-"*40)

        # Show key information
        if 'prompt' in operator_config:
            prompt_preview = operator_config['prompt'][:200] + "..." if len(operator_config.get('prompt', '')) > 200 else operator_config.get('prompt', '')
            print(f"\n📝 Prompt: {prompt_preview}")

        if 'input' in operator_config:
            print(f"\n📥 Input Schema: {json.dumps(operator_config['input'], indent=2)}")

        if 'output' in operator_config:
            print(f"\n📤 Output Schema: {json.dumps(operator_config['output'], indent=2)}")

        # Ask for confirmation
        if operator_index + 1 < total_operators:
            print(f"\n➡️  Continue to next operator ({operator_index + 2}/{total_operators})? (Y/r/n): ", end="")
        else:
            print(f"\n➡️  This is the last operator. Continue to pipeline conversion? (Y/r/n): ", end="")

        user_input = input().strip().lower()

        if user_input == 'r':
            print("🔄 Regenerating this operator...")
            return 'regenerate'
        elif user_input == 'n':
            print("❌ User aborted operator generation")
            return 'abort'
        else:
            if operator_index + 1 < total_operators:
                print("✅ Proceeding to next operator...")
            else:
                print("✅ Proceeding to pipeline conversion...")
            return 'continue'

    def display_pipeline_summary(
        self,
        operators: List[Any],
        query: str
    ):
        """
        Display summary of the generated abstract pipeline.

        Args:
            operators: List of generated operators
            query: Original query
        """
        if not (self.config.confirm or self.config.debug):
            return

        print("\n" + "="*80)
        print("📊 Abstract Pipeline Summary")
        print("="*80)

        print(f"\n📋 Query:")
        print("-"*40)
        print(query[:200] + "..." if len(query) > 200 else query)
        print("-"*40)

        print(f"\n🔧 Pipeline Operators ({len(operators)} total):")
        print("-"*40)
        for i, op in enumerate(operators):
            op_type = op.type if hasattr(op, 'type') else 'Unknown'
            op_name = op.name if hasattr(op, 'name') else f'op_{i}'
            print(f"  {i+1}. {op_name} ({op_type})")
        print("-"*40)

    def _parse_operator_output_fields(self, operator: Dict[str, Any]) -> List[str]:
        """
        Parse output fields from an operator's configuration.

        Returns a list of field names that this operator will add to the data.

        Args:
            operator: Operator configuration dictionary

        Returns:
            List of output field names
        """
        output_fields = []

        # Special handling for extract operator
        if operator.get('type') == 'extract':
            if 'document_keys' in operator and operator['document_keys'] != "TO_BE_GENERATED":
                if isinstance(operator['document_keys'], list) and operator['document_keys']:
                    for doc_key in operator['document_keys']:
                        suffix = operator.get('extraction_key_suffix', f"_extracted_{operator.get('name', 'extract')}")
                        output_fields.append(f"{doc_key}{suffix}")
                else:
                    # Fallback
                    suffix = operator.get('extraction_key_suffix', f"_extracted_{operator.get('name', 'extract')}")
                    output_fields.append(f"src{suffix}")
            return output_fields

        # Parse output schema
        if 'output' in operator and isinstance(operator['output'], dict):
            schema = operator['output'].get('schema', {})

            if isinstance(schema, dict):
                output_fields.extend(schema.keys())
            elif isinstance(schema, str):
                import re
                matches = re.findall(r'(\w+)\s*:\s*\w+', schema)
                output_fields.extend(matches)

        # Special handling for specific operator types
        if operator.get('type') == 'unnest' and 'unnest_key' in operator:
            pass  # Fields remain the same, just expanded
        elif operator.get('type') == 'split':
            output_fields.append('_split_id')
            output_fields.append('_split_index')
        elif operator.get('type') == 'gather':
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

    def confirm_validation_error(
        self,
        operator_index: int,
        total_operators: int,
        errors: List[str],
        warnings: List[str]
    ) -> str:
        """
        Display validation errors and ask user whether to continue.

        Args:
            operator_index: Current operator index (0-based)
            total_operators: Total number of operators
            errors: List of validation error messages
            warnings: List of validation warning messages

        Returns:
            'continue' if user wants to continue despite errors
            'abort' if user wants to abort pipeline generation
        """
        # ANSI color codes
        ORANGE = '\033[33m'
        RED = '\033[31m'
        RESET = '\033[0m'

        print("\n" + "="*80)
        print(f"{ORANGE}⚠️  VALIDATION ERROR - Operator {operator_index + 1}/{total_operators}{RESET}")
        print("="*80)

        print(f"\n{ORANGE}Static validation found issues with the current pipeline:{RESET}")
        print("-"*40)

        # Display errors
        if errors:
            print(f"\n{RED}Errors ({len(errors)}):{RESET}")
            for i, error in enumerate(errors, 1):
                # Extract message from error dict if available
                error_msg = error.get('message', str(error)) if isinstance(error, dict) else str(error)
                print(f"  {i}. {error_msg}")

        # Display warnings
        if warnings:
            print(f"\n{ORANGE}Warnings ({len(warnings)}):{RESET}")
            for i, warning in enumerate(warnings, 1):
                # Extract message from warning dict if available
                warning_msg = warning.get('message', str(warning)) if isinstance(warning, dict) else str(warning)
                print(f"  {i}. {warning_msg}")

        print("-"*40)

        # If not in confirm/debug mode, default to abort
        if not (self.config.confirm or self.config.debug):
            print(f"\n{RED}❌ Aborting due to validation errors{RESET}")
            return 'abort'

        # Ask user for decision in confirm/debug mode
        print(f"\n{ORANGE}The generated pipeline has validation errors.{RESET}")
        print("This may cause issues during execution.")
        print("\nOptions:")
        print("  Y - Continue anyway (errors will be recorded)")
        print("  N - Abort pipeline generation")
        print("\n➡️  Continue despite validation errors? (y/N): ", end="")

        user_input = input().strip().lower()

        if user_input == 'y':
            print(f"{ORANGE}⚠️  Continuing with validation errors...{RESET}")
            return 'continue'
        else:
            print(f"{RED}❌ Pipeline generation aborted due to validation errors{RESET}")
            return 'abort'
