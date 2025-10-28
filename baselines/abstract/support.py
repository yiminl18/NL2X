"""
Base System Support
"""

from typing import Dict, List, Tuple, Any
from enum import Enum


class BaseSystem(Enum):
    DOCETL = "docetl"


ALL_ABSTRACT_OPERATORS = {
    'Map', 'Filter', 'Reduce', 'Resolve', 'Join',
    'Rank', 'TopK', 'Extract', 'Cluster',
    'Split', 'Gather', 'Unnest', 'Sample',
    'Index', 'Project', 'Search'
}

BASE_SYSTEM_SUPPORT: Dict[BaseSystem, List[str]] = {
    BaseSystem.DOCETL: [
        'Map', 'Filter', 'Reduce', 'Resolve', 'Unnest', 'Extract',
        # 'Rank', 'Join', 'TopK', 'Cluster',
        # 'Split', 'Gather', 'Sample'
    ],
}

OPERATOR_DESCRIPTIONS = {
    'Map': {
        'description': 'Transform each record independently using an LLM',
        'core_function': 'Analyzes and transforms individual records to generate new fields through LLM-based processing. Performs operations like sentiment analysis, summarization, theme extraction, classification, or any transformation that requires understanding content. Each record is processed independently.',
        'input_schema': {
            'description': 'Individual records with fields to be analyzed or transformed',
            'example': '{"content": "The new product exceeded expectations...", "title": "Product Review", "date": "2024-01-15"}'
        },
        'output_schema': {
            'description': 'Newly generated fields from extraction, transformation, analysis, classification, or any other operation that results in new information.',
            'example': '{"sentiment": "positive", "themes": ["quality", "value"], "summary": "Positive review highlighting product quality"}'
        }
    },
    'Filter': {
        'description': 'Select records based on condition',
        'core_function': 'Keeps or discards records based on LLM evaluation of conditions. Uses semantic understanding to filter data based on relevance, quality, topic match, or any criteria requiring content comprehension. Returns boolean decisions.',
        'input_schema': {
            'description': 'Records to evaluate for filtering',
            'example': '{"article_text": "Climate change impacts...", "source": "Scientific Journal", "quality_score": 8}'
        },
        'output_schema': {
            'description': 'Boolean field indicating whether to keep the record',
            'example': '{"keep_record": true}'
        }
    },
    'Reduce': {
        'description': 'Aggregate records by grouping key',
        'core_function': 'Groups multiple records by a common key and aggregates them into summary records. Ideal for summarizing themes across documents, consolidating viewpoints, computing statistics per group, or creating aggregated insights from grouped data.',
        'input_schema': {
            'description': 'Multiple records sharing common grouping keys',
            'example': '[{"theme": "sustainability", "viewpoint": "Important for future", "source": "Article1"}, {"theme": "sustainability", "viewpoint": "Cost-effective", "source": "Article2"}]'
        },
        'output_schema': {
            'description': 'Single aggregated record per group with summary information',
            'example': '{"theme": "sustainability", "aggregated_viewpoints": "Multiple perspectives emphasizing importance and cost-effectiveness", "source_count": 2, "consensus": "positive"}'
        }
    },
    'Resolve': {
        'description': 'Merge/deduplicate similar records',
        'core_function': 'Identifies and merges duplicate or similar records into canonical versions. Uses comparison prompts to detect similarity and resolution prompts to merge matching records. Essential for entity resolution, deduplication, and standardization.',
        'input_schema': {
            'description': 'Records potentially containing duplicates or similar entities',
            'example': '[{"name": "John Smith", "email": "jsmith@example.com"}, {"name": "J. Smith", "email": "john.smith@example.com"}]'
        },
        'output_schema': {
            'description': 'Deduplicated records with standardized/merged information',
            'example': '[{"canonical_name": "John Smith", "email": "jsmith@example.com"}, {"canonical_name": "John Smith", "email": "john.smith@example.com"}]'
        }
    },
    'Join': {
        'description': 'Combine two datasets based on key',
        'core_function': 'Merges records from two different datasets based on matching keys. Similar to SQL JOIN operations but with semantic matching capabilities. Supports inner, left, right, and outer joins.',
        'input_schema': {
            'description': 'Two datasets with common keys for joining',
            'example': 'Dataset1: [{"product_id": "P001", "name": "Widget"}], Dataset2: [{"product_id": "P001", "reviews": 150}]'
        },
        'output_schema': {
            'description': 'Combined records with fields from both datasets',
            'example': '{"product_id": "P001", "name": "Widget", "reviews": 150}'
        }
    },
    'Rank': {
        'description': 'Order records by criteria',
        'core_function': 'Sorts records based on LLM-evaluated criteria for relevance, quality, or custom scoring. Adds rank positions to records based on semantic understanding rather than simple numeric sorting.',
        'input_schema': {
            'description': 'Records to be ranked with relevant fields for scoring',
            'example': '[{"title": "Advanced ML Techniques", "content": "...", "citations": 45}, {"title": "Basic Statistics", "content": "...", "citations": 12}]'
        },
        'output_schema': {
            'description': 'Records with added rank field indicating position',
            'example': '{"title": "Advanced ML Techniques", "content": "...", "citations": 45, "rank": 1}'
        }
    },
    'TopK': {
        'description': 'Select top K records',
        'core_function': 'Retrieves the K most relevant records using semantic search (embeddings) or keyword matching. Ideal for finding most similar documents, relevant passages, or best matches to a query.',
        'input_schema': {
            'description': 'Collection of records to search within',
            'example': '[{"title": "Machine Learning Basics", "content": "..."}, {"title": "Deep Learning Guide", "content": "..."}]'
        },
        'output_schema': {
            'description': 'Top K most relevant records based on search criteria',
            'example': 'Top 5 records matching "neural networks" query'
        }
    },
    'Extract': {
        'description': 'Verbatim extract span from text',
        'core_function': 'Pulls exact text spans from documents without transformation. Only extracts verbatim content like quotes, findings, or specific sections. Note that extract can only generate ONE field per input field. Use Map for transformations or multiple fields extraction from the same input field.',
        'input_schema': {
            'description': 'Documents containing text to extract from',
            'example': '{"research_paper": "Abstract: This study examines... Findings: We discovered that... Conclusion: The results suggest..."}'
        },
        'output_schema': {
            'description': 'Extracted verbatim text spans as a new field',
            'example': '{"key_findings": "We discovered that..."}'
        }
    },
    'Cluster': {
        'description': 'Group similar records into clusters',
        'core_function': 'Groups similar records using embedding-based similarity. Creates clusters of related documents, identifies topics across corpus, or groups items by semantic similarity. Includes cluster summarization capabilities.',
        'input_schema': {
            'description': 'Records to be clustered based on content similarity',
            'example': '[{"article": "AI in healthcare..."}, {"article": "Machine learning diagnosis..."}, {"article": "Climate models..."}]'
        },
        'output_schema': {
            'description': 'Records with cluster assignments and optional cluster summaries',
            'example': '{"article": "AI in healthcare...", "cluster_id": 1, "cluster_theme": "AI/ML in Medicine"}'
        }
    },
    'Split': {
        'description': 'Split records into smaller chunks',
        'core_function': 'Breaks long text fields into smaller, manageable chunks for processing. Supports splitting by token count, sentences, or custom delimiters. Essential for handling documents exceeding LLM context limits.',
        'input_schema': {
            'description': 'Records with long text fields to be split',
            'example': '{"document_id": "DOC001", "content": "Very long text content spanning multiple pages..."}'
        },
        'output_schema': {
            'description': 'Multiple chunk records with split tracking fields',
            'example': '{"document_id": "DOC001", "content": "First chunk of text...", "_split_id": "DOC001_1", "chunk_index": 0, "total_chunks": 5}'
        }
    },
    'Gather': {
        'description': 'Reassemble previously split chunks',
        'core_function': 'Adds surrounding context to split chunks by including content from adjacent chunks. Provides peripheral context for better understanding without full reassembly. Used after Split operations.',
        'input_schema': {
            'description': 'Split chunks with tracking information',
            'example': '{"content": "Middle chunk text...", "document_id": "DOC001", "chunk_index": 2}'
        },
        'output_schema': {
            'description': 'Chunks enriched with context from surrounding chunks',
            'example': '{"content": "Middle chunk text...", "previous_context": "End of previous chunk...", "next_context": "Start of next chunk..."}'
        }
    },
    'Unnest': {
        'description': 'Flatten complex structures',
        'core_function': 'This is very important because other opeartors\' prompts almost always access simple and flat fields. Expands arrays into separate records or flattens nested dictionaries. Transforms List fields into individual records (one per item) or brings nested fields to top level. Critical for processing structured data.',
        'input_schema': {
            'description': 'Records containing arrays or nested structures',
            'example': '{"product": "Widget", "reviews": [{"text": "Great!", "rating": 5}, {"text": "Good", "rating": 4}]}'
        },
        'output_schema': {
            'description': 'Flattened records with expanded/unnested data',
            'example': '[{"product": "Widget", "review_text": "Great!", "review_rating": 5}, {"product": "Widget", "review_text": "Good", "review_rating": 4}]'
        }
    },
    'Sample': {
        'description': 'Sample a subset of records',
        'core_function': 'Selects a representative subset of records for processing. Supports uniform random sampling or stratified sampling (balanced across categories). Useful for testing, reducing data volume, or ensuring balanced representation.',
        'input_schema': {
            'description': 'Full dataset to sample from',
            'example': '[{"category": "A", "data": "..."}, {"category": "B", "data": "..."}, ... (1000 records)]'
        },
        'output_schema': {
            'description': 'Sampled subset of records',
            'example': '10% random sample or 100 records balanced across categories'
        }
    },
    'Index': {
        'description': 'Create searchable index for records',
        'core_function': 'Builds an index structure for efficient searching. Creates embeddings or keyword indices for fast retrieval. Note: Currently not supported by DocETL base system.',
        'input_schema': {
            'description': 'Records to be indexed',
            'example': '[{"id": "1", "content": "..."}, {"id": "2", "content": "..."}]'
        },
        'output_schema': {
            'description': 'Indexed structure for efficient search',
            'example': 'Index with searchable embeddings or keywords'
        }
    },
    'Project': {
        'description': 'Select specific fields from records',
        'core_function': 'Selects and keeps only specified fields from records, removing all others. Simple field selection without transformation. Note: Currently not supported by DocETL base system.',
        'input_schema': {
            'description': 'Records with multiple fields',
            'example': '{"name": "John", "age": 30, "email": "john@example.com", "address": "..."}'
        },
        'output_schema': {
            'description': 'Records with only selected fields',
            'example': '{"name": "John", "email": "john@example.com"}'
        }
    },
    'Search': {
        'description': 'Search indexed records by query',
        'core_function': 'Searches through previously indexed records using queries. Performs semantic or keyword search on indexed data. Note: Currently not supported by DocETL base system.',
        'input_schema': {
            'description': 'Search query and indexed data',
            'example': 'Query: "machine learning applications"'
        },
        'output_schema': {
            'description': 'Matching records from the index',
            'example': '[{"id": "5", "content": "ML in healthcare...", "relevance_score": 0.95}]'
        }
    }
}


def get_supported_operators(base_system: BaseSystem) -> List[str]:
    """
    Get the list of operators supported by the specified base system.

    Args:
        base_system: Base system enum value

    Returns:
        List of supported operator types in order
    """
    return BASE_SYSTEM_SUPPORT.get(base_system, []).copy()


def is_operator_supported(operator_type: str, base_system: BaseSystem) -> bool:
    """
    Check if an operator is supported by the base system.

    Args:
        operator_type: Operator type (e.g., 'Map', 'Filter')
        base_system: Base system enum value

    Returns:
        True if supported, False otherwise
    """
    return operator_type in BASE_SYSTEM_SUPPORT.get(base_system, set())

def check_pipeline_compatibility(
    operator_types: List[str],
    base_system: BaseSystem
) -> Tuple[bool, List[str]]:
    """
    Validate whether all operators in a pipeline are compatible with the base system.

    Args:
        operator_types: List of operator types
        base_system: Base system enum value

    Returns:
        (is_compatible, unsupported_operators)
        - is_compatible: True if all operators supported
        - unsupported_operators: List of unsupported operator types
    """
    supported = get_supported_operators(base_system)
    unsupported = [op for op in operator_types if op not in supported]
    return len(unsupported) == 0, unsupported

def format_operators_with_descriptions(base_system: BaseSystem) -> str:
    """
    Format operators with descriptions for LLM prompts.

    Args:
        base_system: Base system enum value

    Returns:
        Formatted string with detailed operator information
    """
    supported = get_supported_operators(base_system)
    lines = []

    for op_type in supported:
        op_info = OPERATOR_DESCRIPTIONS.get(op_type)

        # Format enhanced description
        lines.append(f"- **{op_type}**: {op_info['description']}")
        lines.append(f"  * Core Function: {op_info['core_function']}")

        # Add input/output schema info
        if 'input_schema' in op_info:
            lines.append(f"  * Input: {op_info['input_schema']['description']}")
            lines.append(f"    Example: {op_info['input_schema']['example']}")
        
        if 'output_schema' in op_info:
            lines.append(f"  * Output: {op_info['output_schema']['description']}")
            lines.append(f"    Example: {op_info['output_schema']['example']}")

        lines.append("")  # Empty line for readability

    return "\n".join(lines).strip()


def get_base_system_info(base_system: BaseSystem) -> Dict[str, any]:
    """
    Get complete information about a base system.

    Args:
        base_system: Base system enum value

    Returns:
        Dictionary containing support information
    """
    return {
        'name': base_system.value,
        'supported_operators': get_supported_operators(base_system),
        'operator_count': len(get_supported_operators(base_system))
    }
