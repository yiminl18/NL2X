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

    # ANSI Color codes
    CYAN = '\033[96m'      # Cyan - framework/borders
    BLUE = '\033[94m'      # Blue - info labels
    GREEN = '\033[92m'     # Green - success messages
    YELLOW = '\033[93m'    # Yellow - warnings
    MAGENTA = '\033[95m'   # Magenta - special markers
    RED = '\033[91m'       # Red - errors/failures
    RESET = '\033[0m'      # Reset all formatting
    BOLD = '\033[1m'       # Bold text

    def __init__(self, config):
        """
        Initialize the user interface.

        Args:
            config: Configuration object with confirm and debug flags
        """
        self.config = config

    def display_pipeline_progress(
        self,
        filled_operators: List[Any],
        current_position: int,
        status: str = "generating",
        failed_position: int = None,
        max_display_slots: int = 10
    ) -> None:
        """
        Display visual pipeline generation progress.

        Args:
            filled_operators: List of successfully added operators
            current_position: Current position being processed (0-based)
            status: Current status - "generating", "repair", or "complete"
            failed_position: Position of failed operator (used in repair mode)
            max_display_slots: Unused, kept for compatibility
        """
        # Build pipeline visualization
        slots = []

        # Add filled operators
        for i, op in enumerate(filled_operators):
            op_type = op.type if hasattr(op, 'type') else str(op)
            label = f"{i+1}:{op_type}"

            if status == "repair" and i == failed_position:
                # Failed operator - red double brackets
                slots.append(f"{self.RED}[[{label}]{self.RESET}")
            else:
                # Completed operator - green text
                slots.append(f"{self.GREEN}[{label}]{self.RESET}")

        # Add current position indicator
        if status == "generating" or (status == "repair" and current_position < len(filled_operators)):
            # Current position - bold magenta
            if current_position < len(filled_operators):
                # Replace the current slot with highlighted version
                op = filled_operators[current_position]
                op_type = op.type if hasattr(op, 'type') else str(op)
                label = f"{current_position+1}:{op_type}"
                slots[current_position] = f"{self.BOLD}{self.MAGENTA}**[{label}]**{self.RESET}"
            else:
                # Current position is after all filled operators
                label = f"{current_position+1}"
                slots.append(f"{self.BOLD}{self.MAGENTA}**[{label}]**{self.RESET}")
        elif status == "repair" and current_position not in (None, failed_position):
            # Insert position - bold magenta (for INSERT_BEFORE)
            label = f"{current_position+1}"
            # Insert at the correct position
            if current_position <= len(slots):
                slots.insert(current_position, f"{self.BOLD}{self.MAGENTA}**[{label}]**{self.RESET}")

        # No pending slots - keep it clean

        # Display the pipeline
        pipeline_str = "".join(slots)
        print(f"\n{self.CYAN}Pipeline:{self.RESET} {pipeline_str}\n")

    def confirm_step_before_llm(self, filled_operators: List[Any], step_name: str, iteration: int, step_type: str, step_prompt: str = "") -> tuple[str, List[str]]:
        """
        Ask for confirmation before calling LLM for a step in JIT pipeline generation.

        Args:
            filled_operators: List of successfully added operators
            step_name: Descriptive name of what's being done (e.g., "Operator 1", "Map operator")
            iteration: Current iteration/operator number (1-based)
            step_type: "selection" for operator selection, "filling" for operator configuration
            step_prompt: Prompt content (press 'p' to view)

        Returns:
            Tuple of (action, context):
            - action: 'continue', 'regenerate', or 'abort'
            - context: List of user-provided context/constraints (empty if none)
        """
        step_emoji = "🎯" if step_type == "selection" else "⚙️"
        step_label = "Select" if step_type == "selection" else "Configure"
        title = f"Operator {iteration} - {step_emoji} {step_label}: {self.BLUE}{step_name}{self.RESET}"

        return self._confirm_before_llm_base(
            filled_operators=filled_operators,
            operator_index=iteration - 1,  # Convert 1-based to 0-based
            title=title,
            details={},  # No details for step selection
            prompt=step_prompt,
            allow_context=True,
            allow_regenerate=True,
            operator_type="TBD" if step_type == "selection" else "Selected",
            operator_purpose=f"{step_label} phase"
        )

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

    def _display_prompt(self, prompt: str) -> None:
        """Display formatted prompt to user."""
        print(f"\n{self.CYAN}📝 Prompt:{self.RESET}")
        print(f"{self.CYAN}{'-'*40}{self.RESET}")
        print(prompt)
        print(f"{self.CYAN}{'-'*40}{self.RESET}")

    def _process_confirmation_choice(
        self,
        user_input: str,
        prompt: str,
        allow_context: bool,
        allow_regenerate: bool,
        operator_type: str,
        operator_purpose: str,
        current_position: int
    ) -> tuple[str, List[str]]:
        """
        Process user's confirmation choice and return action + context.

        Args:
            user_input: Initial user input
            prompt: Full prompt to display if requested
            allow_context: Whether 'c' option is available
            allow_regenerate: Whether 'r' option is available
            operator_type: Type of operator (for context)
            operator_purpose: Purpose of operator (for context)
            current_position: Position in pipeline (for context)

        Returns:
            Tuple of (action, context_list)
        """
        user_context = []

        # Handle 'c' to add context
        if user_input == 'c' and allow_context:
            user_context = self.prompt_for_context(
                operator_type=operator_type,
                operator_purpose=operator_purpose,
                current_position=current_position
            )
            # After adding context, ask what to do next
            print(f"\n{self.CYAN}Proceed? (Y/r/p/n):{self.RESET} ", end="")
            user_input = input().strip().lower()

        # Handle 'p' to view full prompt
        if user_input == 'p' and prompt:
            self._display_prompt(prompt)
            # Re-prompt after showing full prompt
            print(f"\n{self.CYAN}Proceed? (Y/r/n):{self.RESET} ", end="")
            user_input = input().strip().lower()

        # Handle final decision
        if user_input == 'r':
            if allow_regenerate:
                print(f"{self.YELLOW}🔄 Regenerating (bypassing cache)...{self.RESET}")
                return 'regenerate', user_context
            else:
                # If regenerate not allowed, default to continue
                print(f"{self.GREEN}✅ Continuing...{self.RESET}")
                return 'continue', user_context
        elif user_input == 'n':
            print(f"{self.YELLOW}❌ Aborted{self.RESET}")
            return 'abort', user_context
        else:
            return 'continue', user_context

    def _confirm_before_llm_base(
        self,
        filled_operators: List[Any],
        operator_index: int,
        title: str,
        details: Dict[str, str],
        prompt: str,
        allow_context: bool,
        allow_regenerate: bool,
        operator_type: str = "",
        operator_purpose: str = "",
        cache_status: str = ""
    ) -> tuple[str, List[str]]:
        """
        Base confirmation method for LLM calls - extracts common confirmation flow.

        Args:
            filled_operators: List of successfully added operators
            operator_index: Current operator index (0-based)
            title: Title to display (e.g., "Operator 1 - 🎯 Select")
            details: Dict of detail lines to display (e.g., {"Type": "Map", "Purpose": "..."})
            prompt: Full prompt for LLM (shown with 'p')
            allow_context: Whether to show 'C' option
            allow_regenerate: Whether to show 'R' option
            operator_type: Type for context collection
            operator_purpose: Purpose for context collection
            cache_status: Cache status message (e.g., "(CACHED)" or "")

        Returns:
            Tuple of (action, context):
            - action: 'continue', 'regenerate', or 'abort'
            - context: List of user-provided context/constraints
        """
        if not (self.config.confirm or self.config.debug):
            return 'continue', []

        # Display pipeline progress
        self.display_pipeline_progress(
            filled_operators=filled_operators,
            current_position=operator_index,
            status="generating"
        )

        # Display title
        print(f"{self.CYAN}{self.BOLD}➡️  {title}{self.RESET}{cache_status}")

        # Display details if provided
        if details:
            for label, value in details.items():
                print(f"{self.BLUE}{label}:{self.RESET} {value}")
            print(f"{self.CYAN}{'-'*40}{self.RESET}")

        # Show options
        print(f"\n{self.CYAN}Options:{self.RESET}")
        print("  Y - Continue")
        if allow_regenerate:
            print("  R - Regenerate (bypass cache)")
        if allow_context:
            print("  C - Add context/constraints")
        if prompt:
            print("  P - View full prompt")
        print("  N - Abort")

        # Build choice prompt
        choices = ['Y']
        if allow_regenerate:
            choices.append('r')
        if allow_context:
            choices.append('c')
        if prompt:
            choices.append('p')
        choices.append('n')
        choice_str = '/'.join(choices)
        print(f"\n{self.CYAN}Choice ({choice_str}):{self.RESET} ", end="")

        user_input = input().strip().lower()

        # Process choice using helper
        return self._process_confirmation_choice(
            user_input=user_input,
            prompt=prompt,
            allow_context=allow_context,
            allow_regenerate=allow_regenerate,
            operator_type=operator_type or "TBD",
            operator_purpose=operator_purpose or "TBD",
            current_position=operator_index
        )

    def confirm_operator_before_llm(
        self,
        filled_operators: List[Any],
        operator_index: int,
        total_operators: int,
        operator_type: str,
        operator_purpose: str,
        prompt: str,
        is_cached: bool = False,
    ) -> tuple[str, List[str]]:
        """
        Ask for confirmation before generating a single operator.

        Args:
            filled_operators: List of successfully added operators
            operator_index: Current operator index (0-based)
            total_operators: Total number of operators (unused, kept for compatibility)
            operator_type: Type of the operator (Map, Filter, etc.)
            operator_purpose: Purpose of the operator
            prompt: The prompt that will be sent to LLM
            is_cached: Whether the prompt is already cached

        Returns:
            Tuple of (action, context):
            - action: 'continue', 'regenerate', or 'abort'
            - context: List of user-provided context/constraints (empty if none)
        """
        if not (self.config.confirm or self.config.debug):
            return 'continue', []

        # Display header
        self.display_pipeline_progress(
            filled_operators=filled_operators,
            current_position=operator_index,
            status="generating"
        )
        print(f"{self.CYAN}{'='*80}{self.RESET}")
        mode_text = f"{self.BLUE}[DEBUG MODE]{self.RESET}" if self.config.debug else f"{self.BLUE}[CONFIRM MODE]{self.RESET}"
        cache_text = f" {self.YELLOW}(CACHED){self.RESET}" if is_cached else ""
        print(f"{mode_text} {self.BOLD}Operator {operator_index + 1} - Filling Details{self.RESET}{cache_text}")
        print(f"{self.CYAN}{'='*80}{self.RESET}")

        # Show cache status message
        if is_cached:
            print(f"\n{self.YELLOW}💾 This prompt is CACHED. Type 'r' to bypass cache.{self.RESET}")

        # Use base method with details
        action, context = self._confirm_before_llm_base(
            filled_operators=filled_operators,
            operator_index=operator_index,
            title="",  # Already displayed above
            details={"📋 Type": operator_type, "📋 Purpose": operator_purpose},
            prompt=prompt,
            allow_context=True,
            allow_regenerate=is_cached,  # Only allow regenerate if cached
            operator_type=operator_type,
            operator_purpose=operator_purpose,
            cache_status=""
        )

        # Custom message based on action and cache status
        if action == 'continue':
            if is_cached:
                print(f"{self.GREEN}✅ Using cached response...{self.RESET}")
            else:
                print(f"{self.GREEN}✅ Generating operator...{self.RESET}")
        elif action == 'regenerate' and not is_cached:
            # If user tried to regenerate but it's not cached, default to continue
            print(f"{self.GREEN}✅ Generating operator...{self.RESET}")
            action = 'continue'

        return action, context

    def display_generated_operator(
        self,
        filled_operators: List[Any],
        operator_index: int,
        total_operators: int,
        operator_type: str,
        operator_config: Dict[str, Any]
    ) -> str:
        """
        Display generated operator configuration and ask for confirmation.

        Args:
            filled_operators: List of successfully added operators
            operator_index: Current operator index (0-based)
            total_operators: Total number of operators (unused, kept for compatibility)
            operator_type: Type of the operator
            operator_config: Generated operator configuration

        Returns:
            'continue' to proceed to next operator
            'regenerate' to regenerate this operator
            'edit' to edit the operator configuration
            'abort' to stop generation
        """
        if not (self.config.confirm or self.config.debug):
            return 'continue'

        # Display pipeline progress
        self.display_pipeline_progress(
            filled_operators=filled_operators,
            current_position=operator_index,
            status="generating"
        )

        print(f"{self.CYAN}{'='*80}{self.RESET}")
        print(f"{self.GREEN}✅ Generated Operator {operator_index + 1}: {self.BOLD}{operator_type}{self.RESET}")
        print(f"{self.CYAN}{'='*80}{self.RESET}")

        print(f"\n{self.BLUE}📋 Configuration:{self.RESET}")
        print(f"{self.CYAN}{'-'*40}{self.RESET}")

        # Format and display the configuration
        config_str = json.dumps(operator_config, indent=2, ensure_ascii=False)
        print(config_str)
        print(f"{self.CYAN}{'-'*40}{self.RESET}")

        # Show key information
        if 'prompt' in operator_config:
            prompt_preview = operator_config['prompt'][:200] + "..." if len(operator_config.get('prompt', '')) > 200 else operator_config.get('prompt', '')
            print(f"\n{self.BLUE}📝 Prompt:{self.RESET} {prompt_preview}")

        if 'input' in operator_config:
            print(f"\n{self.BLUE}📥 Input Schema:{self.RESET}")
            print(json.dumps(operator_config['input'], indent=2))

        if 'output' in operator_config:
            print(f"\n{self.BLUE}📤 Output Schema:{self.RESET}")
            print(json.dumps(operator_config['output'], indent=2))

        # Ask for confirmation
        print(f"\n{self.CYAN}Options:{self.RESET}")
        print("  Y - Continue (accept this operator)")
        print("  R - Regenerate (with higher temperature)")
        print("  E - Edit configuration")
        print("  N - Abort")
        print(f"\n{self.CYAN}Choice (Y/r/e/n):{self.RESET} ", end="")

        user_input = input().strip().lower()

        if user_input == 'r':
            print(f"{self.YELLOW}🔄 Regenerating operator...{self.RESET}")
            return 'regenerate'
        elif user_input == 'e':
            print(f"{self.BLUE}📝 Opening editor...{self.RESET}")
            return 'edit'
        elif user_input == 'n':
            print(f"{self.YELLOW}❌ Aborted{self.RESET}")
            return 'abort'
        else:
            print(f"{self.GREEN}✅ Proceeding...{self.RESET}")
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

        print(f"\n{self.CYAN}{'='*80}{self.RESET}")
        print(f"{self.BOLD}{self.GREEN}📊 Pipeline Complete - Summary{self.RESET}")
        print(f"{self.CYAN}{'='*80}{self.RESET}")

        print(f"\n{self.BLUE}📋 Query:{self.RESET}")
        print(f"{self.CYAN}{'-'*40}{self.RESET}")
        print(query[:200] + "..." if len(query) > 200 else query)
        print(f"{self.CYAN}{'-'*40}{self.RESET}")

        print(f"\n{self.BLUE}🔧 Pipeline Operators ({len(operators)} total):{self.RESET}")
        print(f"{self.CYAN}{'-'*40}{self.RESET}")
        for i, op in enumerate(operators):
            op_type = op.type if hasattr(op, 'type') else 'Unknown'
            op_name = op.name if hasattr(op, 'name') else f'op_{i}'
            print(f"  {self.GREEN}{i+1}.{self.RESET} {op_name} ({self.BOLD}{op_type}{self.RESET})")
        print(f"{self.CYAN}{'-'*40}{self.RESET}")

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
        filled_operators: List[Any],
        operator_index: int,
        total_operators: int,
        errors: List[str],
        warnings: List[str]
    ) -> str:
        """
        Display validation errors and ask user whether to continue.

        Args:
            filled_operators: List of successfully added operators
            operator_index: Current operator index (0-based)
            total_operators: Total number of operators (unused, kept for compatibility)
            errors: List of validation error messages
            warnings: List of validation warning messages

        Returns:
            'continue' if user wants to continue despite errors
            'abort' if user wants to abort pipeline generation
        """
        RED = '\033[91m'  # Bright red for errors

        # Display pipeline progress with failed operator highlighted
        self.display_pipeline_progress(
            filled_operators=filled_operators,
            current_position=operator_index,
            status="repair",
            failed_position=operator_index
        )

        print(f"{self.CYAN}{'='*80}{self.RESET}")
        print(f"{self.YELLOW}⚠️  VALIDATION ERROR - Operator {operator_index + 1}{self.RESET}")
        print(f"{self.CYAN}{'='*80}{self.RESET}")

        print(f"\n{self.YELLOW}Validation found issues with this operator:{self.RESET}")
        print(f"{self.CYAN}{'-'*40}{self.RESET}")

        # Display errors
        if errors:
            print(f"\n{RED}❌ Errors ({len(errors)}):{self.RESET}")
            for i, error in enumerate(errors, 1):
                # Extract message from error dict if available
                error_msg = error.get('message', str(error)) if isinstance(error, dict) else str(error)
                print(f"  {i}. {error_msg}")

        # Display warnings
        if warnings:
            print(f"\n{self.YELLOW}⚠️  Warnings ({len(warnings)}):{self.RESET}")
            for i, warning in enumerate(warnings, 1):
                # Extract message from warning dict if available
                warning_msg = warning.get('message', str(warning)) if isinstance(warning, dict) else str(warning)
                print(f"  {i}. {warning_msg}")

        print(f"{self.CYAN}{'-'*40}{self.RESET}")

        # If not in confirm/debug mode, default to abort
        if not (self.config.confirm or self.config.debug):
            print(f"\n{RED}❌ Aborting due to validation errors{self.RESET}")
            return 'abort'

        # Ask user for decision in confirm/debug mode
        print(f"\n{self.YELLOW}This operator has validation errors that may cause issues.{self.RESET}")
        print(f"\n{self.CYAN}Options:{self.RESET}")
        print("  Y - Continue anyway (errors will be recorded)")
        print("  N - Abort pipeline generation")
        print(f"\n{self.CYAN}Continue despite errors? (y/N):{self.RESET} ", end="")

        user_input = input().strip().lower()

        if user_input == 'y':
            print(f"{self.YELLOW}⚠️  Continuing with validation errors...{self.RESET}")
            return 'continue'
        else:
            print(f"{RED}❌ Aborted{self.RESET}")
            return 'abort'

    def confirm_repair_analysis_before_llm(
        self,
        filled_operators: List[Any],
        operator_index: int,
        prompt: str
    ) -> str:
        """
        Ask for confirmation before calling LLM for repair analysis.

        Args:
            filled_operators: List of successfully added operators
            operator_index: Current operator index (0-based)
            prompt: Repair analysis prompt (press 'p' to view)

        Returns:
            'continue' if user wants to continue
            'regenerate' if user wants to regenerate (bypass cache)
            'abort' if user wants to abort
        """
        if not (self.config.confirm or self.config.debug):
            return 'continue'

        # Display pipeline progress with failed operator highlighted
        self.display_pipeline_progress(
            filled_operators=filled_operators,
            current_position=operator_index,
            status="repair",
            failed_position=operator_index
        )

        # Use base method for confirmation (without context support)
        title = f"Operator {operator_index + 1} - 🔍 Repair Analysis"
        action, _ = self._confirm_before_llm_base(
            filled_operators=filled_operators,
            operator_index=operator_index,
            title=title,
            details={},
            prompt=prompt,
            allow_context=False,  # No context for repair analysis
            allow_regenerate=True,
            operator_type="Repair",
            operator_purpose="Analysis"
        )

        return action

    def confirm_repair_suggestion(
        self,
        filled_operators: List[Any],
        operator_index: int,
        total_operators: int,
        repair_suggestion: Dict[str, Any]
    ) -> str:
        """
        Display LLM's repair suggestion and get user confirmation.

        Args:
            filled_operators: List of successfully added operators
            operator_index: Current operator index (0-based)
            total_operators: Total number of operators (unused, kept for compatibility)
            repair_suggestion: LLM's repair suggestion dict with:
                - analysis: Error analysis
                - action: Repair action (DELETE, INSERT_BEFORE, etc.)
                - new_operator: New operator info (for INSERT/REPLACE)
                - rationale: Reason for this repair

        Returns:
            'accept' - Accept and apply the suggestion
            'skip' - Skip the suggestion and continue with errors
            'abort' - Abort pipeline generation
        """
        # Display pipeline progress with failed operator highlighted
        self.display_pipeline_progress(
            filled_operators=filled_operators,
            current_position=operator_index,
            status="repair",
            failed_position=operator_index
        )

        print(f"{self.CYAN}{'='*80}{self.RESET}")
        print(f"{self.MAGENTA}{self.BOLD}🔧 REPAIR SUGGESTION - Operator {operator_index + 1}{self.RESET}")
        print(f"{self.CYAN}{'='*80}{self.RESET}")

        # Display error analysis
        print(f"\n{self.BLUE}📊 Error Analysis:{self.RESET}")
        print(f"{self.CYAN}{'-'*40}{self.RESET}")
        print(repair_suggestion.get('analysis', 'N/A'))
        print(f"{self.CYAN}{'-'*40}{self.RESET}")

        # Display suggested operator repair
        print(f"\n{self.GREEN}💡 Repair Suggestion:{self.RESET}")
        print(f"{self.CYAN}{'-'*40}{self.RESET}")

        new_operator = repair_suggestion.get('new_operator', {})
        if new_operator:
            print(f"{self.BLUE}New Operator:{self.RESET}")
            print(f"  Type: {new_operator.get('type', 'N/A')}")
            print(f"  Purpose: {new_operator.get('purpose', 'N/A')}")
        print(f"{self.CYAN}{'-'*40}{self.RESET}")

        # Display rationale
        print(f"\n{self.BLUE}📝 Rationale:{self.RESET}")
        print(f"{self.CYAN}{'-'*40}{self.RESET}")
        print(repair_suggestion.get('rationale', 'N/A'))
        print(f"{self.CYAN}{'-'*40}{self.RESET}")

        # Get user decision
        print(f"\n{self.CYAN}Options:{self.RESET}")
        print("  A - Accept and apply this repair")
        print("  S - Skip and continue with errors")
        print("  N - Abort generation")
        print(f"\n{self.CYAN}Apply this repair? (A/s/n):{self.RESET} ", end="")

        user_input = input().strip().lower()

        if user_input == 'a' or user_input == '':
            print(f"{self.GREEN}✅ Applying repair...{self.RESET}")
            return 'accept'
        elif user_input == 's':
            print(f"{self.YELLOW}⚠️  Skipping repair...{self.RESET}")
            return 'skip'
        else:
            print(f"{self.YELLOW}❌ Aborted{self.RESET}")
            return 'abort'

    def edit_operator_config(
        self,
        operator_config: Dict[str, Any],
        operator_type: str,
        operator_index: int
    ) -> Dict[str, Any]:
        """
        Open operator configuration in external editor for user modification.

        Args:
            operator_config: Current operator configuration
            operator_type: Type of the operator (Map, Filter, etc.)
            operator_index: Position in pipeline

        Returns:
            Modified operator configuration, or original if editing failed/cancelled
        """
        import os
        import tempfile
        import subprocess

        print(f"\n{self.CYAN}{'='*80}{self.RESET}")
        print(f"{self.BLUE}📝 EDIT OPERATOR {operator_index + 1} - {operator_type}{self.RESET}")
        print(f"{self.CYAN}{'='*80}{self.RESET}")

        # Create temporary file with current configuration
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix=f'_op{operator_index}_{operator_type}.json',
            delete=False,
            prefix='nl2x_edit_'
        ) as f:
            json.dump(operator_config, f, indent=2)
            temp_file = f.name

        print(f"\n{self.BLUE}Configuration saved to:{self.RESET} {temp_file}")

        # Determine editor
        editor = os.environ.get('EDITOR')
        if not editor:
            # Try common editors
            for candidate in ['vim', 'nano', 'vi']:
                if subprocess.run(['which', candidate], capture_output=True).returncode == 0:
                    editor = candidate
                    break

        if not editor:
            print(f"{self.RED}❌ No editor found. Set $EDITOR environment variable.{self.RESET}")
            print(f"{self.YELLOW}Available editors: vim, nano, vi{self.RESET}")
            os.unlink(temp_file)
            return operator_config

        print(f"{self.BLUE}Opening editor:{self.RESET} {editor}")
        print(f"{self.YELLOW}💡 Tip: Edit the JSON configuration, then save and exit.{self.RESET}\n")

        # Open editor
        try:
            subprocess.call([editor, temp_file])
        except Exception as e:
            print(f"{self.RED}❌ Editor failed: {e}{self.RESET}")
            os.unlink(temp_file)
            return operator_config

        # Read modified configuration
        try:
            with open(temp_file, 'r') as f:
                edited_config = json.load(f)
            print(f"\n{self.GREEN}✅ Configuration loaded successfully{self.RESET}")
        except json.JSONDecodeError as e:
            print(f"\n{self.RED}❌ JSON parsing error:{self.RESET} {e}")
            print(f"{self.YELLOW}Reverting to original configuration.{self.RESET}")
            os.unlink(temp_file)
            return operator_config
        except Exception as e:
            print(f"\n{self.RED}❌ Error reading file:{self.RESET} {e}")
            os.unlink(temp_file)
            return operator_config

        # Clean up
        os.unlink(temp_file)

        # Show diff summary
        changes = []
        for key in set(list(operator_config.keys()) + list(edited_config.keys())):
            if key not in operator_config:
                changes.append(f"  + Added: {key}")
            elif key not in edited_config:
                changes.append(f"  - Removed: {key}")
            elif operator_config[key] != edited_config[key]:
                changes.append(f"  ~ Modified: {key}")

        if changes:
            print(f"\n{self.BLUE}📝 Changes detected:{self.RESET}")
            for change in changes[:5]:  # Show first 5 changes
                print(change)
            if len(changes) > 5:
                print(f"  ... and {len(changes) - 5} more changes")
        else:
            print(f"\n{self.YELLOW}No changes detected.{self.RESET}")

        return edited_config

    def prompt_for_context(
        self,
        operator_type: str,
        operator_purpose: str,
        current_position: int
    ) -> List[str]:
        """
        Prompt user for additional context/constraints for operator generation.

        Args:
            operator_type: Type of operator to be generated
            operator_purpose: Purpose of the operator
            current_position: Position in pipeline

        Returns:
            List of user-provided constraints
        """
        print(f"\n{self.CYAN}{'='*80}{self.RESET}")
        print(f"{self.BLUE}💬 ADD CONTEXT FOR OPERATOR {current_position + 1}{self.RESET}")
        print(f"{self.CYAN}{'='*80}{self.RESET}")

        print(f"\n{self.BLUE}Operator Details:{self.RESET}")
        print(f"  Type: {operator_type}")
        print(f"  Purpose: {operator_purpose}")

        print(f"\n{self.YELLOW}You can provide additional constraints or guidance:{self.RESET}")
        print(f"{self.CYAN}Examples:{self.RESET}")
        print("  • Use only fields from section X")
        print("  • Avoid using regex patterns")
        print("  • Generate concise prompts")
        print("  • Focus on extracting specific information")
        print("  • Limit output to 3 fields maximum")

        print(f"\n{self.BLUE}Enter constraints (one per line, empty line to finish):{self.RESET}")

        constraints = []
        while True:
            try:
                constraint = input(f"  {self.CYAN}>{self.RESET} ").strip()
                if not constraint:
                    break
                constraints.append(constraint)
                print(f"    {self.GREEN}✓ Added{self.RESET}")
            except (EOFError, KeyboardInterrupt):
                print(f"\n{self.YELLOW}Input cancelled{self.RESET}")
                break

        if constraints:
            print(f"\n{self.GREEN}✅ Added {len(constraints)} constraint(s){self.RESET}")
        else:
            print(f"\n{self.YELLOW}No constraints added{self.RESET}")

        return constraints
