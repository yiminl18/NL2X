#!/usr/bin/env python3
import os
import re
import yaml
from typing import Any


class PipelineObfuscator:
    
    def __init__(self):
        """Initialize obfuscator with clean mapping state."""
        self.operator_mapping = {}  # original_name -> obfuscated_name
        self.operator_counter = 1
        
    def obfuscate(self, pipeline_content: str) -> str:
        """
        Obfuscate pipeline content while preserving computational logic.
        
        Args:
            pipeline_content: YAML pipeline content as string
            
        Returns:
            str: Obfuscated pipeline YAML content
        """
        try:
            # Parse YAML to work with structured data
            pipeline_dict = yaml.safe_load(pipeline_content)
            
            # Obfuscate the pipeline dictionary
            obfuscated_dict = self._obfuscate_dict(pipeline_dict)
            
            # Convert back to YAML
            return yaml.dump(obfuscated_dict, default_flow_style=False, sort_keys=False)
            
        except Exception as e:
            print(f"Warning: Failed to obfuscate pipeline: {e}")
            return pipeline_content
    
    def _obfuscate_dict(self, obj: Any) -> Any:
        """Recursively obfuscate dictionary structure."""
        if isinstance(obj, dict):
            obfuscated = {}
            for key, value in obj.items():
                # Obfuscate operation names
                if key == 'name' and isinstance(value, str):
                    obfuscated[key] = self._get_obfuscated_operator_name(value)
                else:
                    obfuscated[key] = self._obfuscate_dict(value)
            return obfuscated
            
        elif isinstance(obj, list):
            # Handle operation references in pipeline steps
            obfuscated_list = []
            for item in obj:
                if isinstance(item, str):
                    # Check if this is an operation reference
                    if item in self.operator_mapping:
                        obfuscated_list.append(self.operator_mapping[item])
                    else:
                        # Might be a new operation reference, try to obfuscate
                        obfuscated_item = self._get_obfuscated_operator_name(item)
                        obfuscated_list.append(obfuscated_item)
                else:
                    obfuscated_list.append(self._obfuscate_dict(item))
            return obfuscated_list
            
        else:
            return obj
    
    
    
    def _get_obfuscated_operator_name(self, original_name: str) -> str:
        """Get obfuscated operator name."""
        if original_name not in self.operator_mapping:
            self.operator_mapping[original_name] = f"operator_{self.operator_counter}"
            self.operator_counter += 1
        return self.operator_mapping[original_name]


def obfuscate_pipeline(input_path: str = None, 
                      pipeline_yaml: str = None,
                      output_path: str = None) -> str:
    """
    Obfuscate a DocETL pipeline to prevent intent leakage.
    
    Args:
        input_path: Path to input pipeline YAML file (optional)
        pipeline_yaml: Pipeline YAML content as string (optional)
        output_path: Path to save obfuscated pipeline (optional)
    
    Returns:
        str: Obfuscated pipeline YAML content
        
    Raises:
        ValueError: If neither input_path nor pipeline_yaml is provided
        FileNotFoundError: If input_path does not exist
    """
    # Load pipeline content
    if pipeline_yaml is not None:
        content = pipeline_yaml
        source = "provided YAML string"
    elif input_path is not None:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input pipeline file not found: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        source = input_path
    else:
        raise ValueError("Either input_path or pipeline_yaml must be provided")
    
    print(f"Obfuscating pipeline from {source}...")
    
    # Create obfuscator and process pipeline
    obfuscator = PipelineObfuscator()
    obfuscated_content = obfuscator.obfuscate(content)
    
    # Save to output file if specified
    if output_path:
        output_dir = os.path.dirname(output_path)
        if output_dir:  # Only create directory if there is one
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(obfuscated_content)
        print(f"Obfuscated pipeline saved to: {output_path}")
        
        # Print mapping summary
        print(f"\nObfuscation Summary:")
        print(f"  Operators: {len(obfuscator.operator_mapping)} obfuscated")
        
        if obfuscator.operator_mapping:
            print(f"\n  Operator mappings:")
            for orig, obf in obfuscator.operator_mapping.items():
                print(f"    {orig} -> {obf}")
    
    return obfuscated_content


def main():
    """Command-line interface for the pipeline obfuscator."""
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python3 pipeline_obfuscator.py <input_pipeline.yaml> <output_pipeline.yaml>")
        print("Example: python3 pipeline_obfuscator.py original.yaml obfuscated.yaml")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        obfuscated_content = obfuscate_pipeline(
            input_path=input_path,
            output_path=output_path
        )
        print(f"\n✅ Pipeline obfuscation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()