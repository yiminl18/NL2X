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

    def confirm_step_before_llm(self, step_name: str, step_number: str, total_steps: str = "2") -> str:
        """
        Ask for confirmation before calling LLM for a step.

        Args:
            step_name: Name of the step
            step_number: Current step number
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

        # Display result summary
        if isinstance(result, list):
            print(f"Generated {len(result)} operators:")
            for i, item in enumerate(result):
                if isinstance(item, dict):
                    op_type = item.get('type', 'Unknown')
                    purpose = item.get('purpose', 'N/A')
                    print(f"  {i+1}. {op_type}: {purpose}")
                else:
                    print(f"  {i+1}. {item}")
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
        prompt_file: str
    ) -> str:
        """
        Ask for confirmation before generating a single operator.

        Args:
            operator_index: Current operator index (0-based)
            total_operators: Total number of operators to generate
            operator_type: Type of the operator (Map, Filter, etc.)
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
        print(f"{mode_text} Operator {operator_index + 1}/{total_operators} - Generation")
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
