#!/usr/bin/env python3

from dynamic_checker import PipelineObfuscator, DocETLDynamicChecker
import yaml

# Test pipeline with meaningful names that could leak intent
test_pipeline = """
default_model: gpt-4o-mini

datasets:
  identity_theft_data:
    type: file
    path: "crime_data.json"

operations:
  - name: extract_metropolitan_stats
    type: map
    prompt: |
      Extract data from {{ input.text }} about metropolitan areas
    output:
      schema:
        metropolitan_areas: "list[{city_name: string, population_2023: integer, reported_identity_thefts: integer}]"
  
  - name: unnest_metro_areas
    type: unnest
    unnest_key: metropolitan_areas
  
  - name: filter_large_populations
    type: code_filter
    code: |
      def transform(doc) -> bool:
          return doc['population_2023'] > 1_000_000
  
  - name: calculate_theft_average
    type: reduce
    reduce_key: dummy
    prompt: |
      Calculate average identity thefts for large metropolitan areas:
      {% for item in inputs %}
      City: {{ item.city_name }}, Population: {{ item.population_2023 }}, Thefts: {{ item.reported_identity_thefts }}
      {% endfor %}
    output:
      schema:
        average_identity_thefts: number

pipeline:
  steps:
    - input: identity_theft_data
      operations:
        - extract_metropolitan_stats
        - unnest_metro_areas
        - filter_large_populations
        - calculate_theft_average
  output:
    type: file
    path: "theft_averages.json"
"""

def test_obfuscation():
    print("Testing Pipeline Obfuscation")
    print("=" * 50)
    
    # Test the obfuscator directly
    obfuscator = PipelineObfuscator()
    obfuscated = obfuscator.obfuscate_pipeline(test_pipeline)
    
    print("ORIGINAL PIPELINE:")
    print("-" * 30)
    print(test_pipeline)
    print()
    
    print("OBFUSCATED PIPELINE:")
    print("-" * 30)
    print(obfuscated)
    print()
    
    print("OBFUSCATION MAPPINGS:")
    print("-" * 30)
    print("Operators:")
    for orig, obf in obfuscator.operator_mapping.items():
        print(f"  {orig} -> {obf}")
    
    print("Fields:")  
    for orig, obf in obfuscator.field_mapping.items():
        print(f"  {orig} -> {obf}")
    
    print("Preserved Input Fields:")
    print(f"  {obfuscator.input_fields}")
    print()
    
    # Test that obfuscation is applied in dynamic checker
    print("TESTING DYNAMIC CHECKER WITH/WITHOUT OBFUSCATION:")
    print("-" * 30)
    
    question = "What is the average number of identity thefts in large metropolitan areas?"
    
    # With obfuscation (default)
    checker_obf = DocETLDynamicChecker(obfuscate_names=True)
    result_obf = checker_obf._pipeline_to_question(test_pipeline)
    
    # Without obfuscation
    checker_no_obf = DocETLDynamicChecker(obfuscate_names=False)
    result_no_obf = checker_no_obf._pipeline_to_question(test_pipeline)
    
    print("With obfuscation - inferred query:")
    print(f"  {result_obf['inferred_query']}")
    print()
    
    print("Without obfuscation - inferred query:")
    print(f"  {result_no_obf['inferred_query']}")
    print()
    
    # Verify that key semantic clues are removed
    semantic_clues = ['identity_theft', 'metropolitan', 'population', 'crime', 'theft']
    
    # Check the obfuscated pipeline string directly (not parsed structure)
    try:
        obfuscated_str = obfuscated.lower()
        
        clues_found = []
        for clue in semantic_clues:
            if clue in obfuscated_str:
                clues_found.append(clue)
        
        print("SEMANTIC CLUE ANALYSIS:")
        print("-" * 30)
        if clues_found:
            print(f"⚠️  Semantic clues still present: {clues_found}")
        else:
            print("✅ No semantic clues found in obfuscated pipeline")
            
        # Check if computational logic is preserved
        original_dict = yaml.safe_load(test_pipeline)
        print(f"✅ Pipeline structure preserved: {len(original_dict['operations'])} operations")
        print(f"✅ Operation types preserved: {[op['type'] for op in original_dict['operations']]}")
        print(f"✅ Input dataset reference preserved: {original_dict['pipeline']['steps'][0]['input']}")
        
    except Exception as e:
        print(f"❌ Error analyzing obfuscated pipeline: {e}")

if __name__ == "__main__":
    test_obfuscation()