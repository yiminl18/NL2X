1. Map
Input schema:
    a. Fields from prompt
Output schema:
    a. Fields from output.schema fields.

2. Resolve
Input schema:
    a. Fields from comparison_prompt and resolution_prompt.
Output schema:
    a. Fields from output.schema fields.

3. Reduce
Input schema:
    a. reduce_key field
    b. Fields from Map output schema.
Output schema:
    a. Fields from output.schema fields.

4. Filter
Input schema:
    a. Fields from prompt.
Output schema:
    a. Fields from output.schema fields (at least one is boolean).

5. Join
Input schema:
    a. comparison_prompt
Output schema:
    a. No output schema.

6. Rank
Input schema:
    a. input_keys, such as ["content", "title", "date"]. The types need to be infered.
Output schema:
    a. _rank: interger

7. Extract
Input schema:
    a. document_keys, such as ["name", "age"]. The types need to be infered.
Output schema:
    a. document_keys with '_extracted_findings' suffix like ["name_extracted_findings", "age_extracted_findings"]

8. Cluster
Input schema:
    a. from embedding_keys
    b. from summary_schema
    c. from summary_prompt
Output schema:
    a. from output_key (This is a dict with summary_schema and distance: number) [Optional]
    b. If no specified output_key, use "clusters" that has the same structure as (a).

9. Split
Input schema:
    a. split_key
Output schema:
    a. {split_key}_chunk: string
    b. {op_name}_id: A unique identifier for each original document. This is from source.name field from Abstract Op fields.
    c. {op_name}_chunk_num: The sequential number of the chunk within its original document.

10. Gather
Input schema: 
    a. content_key
    b. doc_id_key
    c. order_key
    d. content_key in peripheral_chunks field like:
    ```
    peripheral_chunks:
        previous:
        middle:
            content_key: agreement_text_chunk_summary
        tail:
            content_key: agreement_text_chunk
        next:
        head:
            count: 1
            content_key: agreement_text_chun
    ```
    e. doc_header_key [Optional]
Output schema:
    a. {content_key}_rendered

11. Unnest
Input schema:
    a. unnest_key
Output schema:
    Special Note: 
    1. unnest_key should be removed from the current available fields.
    2. Unnest have different behaviour given different unnest_key
        2.0. Non-list or Non-dict unnest_key is not allowed.
        2.1. If unnest_key is list such as participants: list[str] or infos: "list[{name: str, age: integer}]". Unnest create multiple records based on the length of the list. In this cased, the output shcema will be "participants: str" or "infos: {name:str, age: integer}". Note the change of field types.
        2.2 If unnest_key is a dict like "personal_info: "{name: str, age: integer}"". Note that personal_info field should be removed. And the output schema will be "name: str" and "age: integer".
        2.3 We temporarily do not handle the `depth` and `recursive`
 configurations.

12. Sample
No change to schema.

13. Top-k
Input schema:
    a. keys
Output schema:
    a. No output schema

14. Code
We temporarily do not handle this.