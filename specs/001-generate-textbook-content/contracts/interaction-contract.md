# Interaction Contract: Content Generation

This document defines the structured input (parameters) and expected output for the content generation process, which is primarily driven by CLI commands and agent interactions rather than traditional API calls. It serves as a contract for how the agent receives requests for content and what kind of response can be expected.

## Input Contract: Content Generation Request

This defines the parameters expected when requesting content generation (e.g., via a `/sp.specify` command or similar internal invocation).

*   **Endpoint/Command**: `generate_content` (Conceptual; invoked through CLI commands or internal agent logic)
*   **Method**: CLI Command Execution / Agent Internal Invocation
*   **Parameters**:
    *   `content_type`:
        *   **Type**: String
        *   **Required**: Yes
        *   **Description**: Specifies the type of content to be generated.
        *   **Allowed Values**: "outline", "chapter", "section"
    *   `topic`:
        *   **Type**: String
        *   **Required**: Yes (for "chapter" or "section")
        *   **Description**: The main subject or title for the content to be generated.
    *   `chapter_context`:
        *   **Type**: String
        *   **Required**: No (Only if `content_type` is "section" and needs parent chapter context)
        *   **Description**: Provides the title or key themes of the parent chapter, to ensure contextual relevance for section generation.
    *   `section_context`:
        *   **Type**: String
        *   **Required**: No (Only if generating a sub-section or content within a larger section and needs parent section context)
        *   **Description**: Provides specific details or prior content of the parent section, for granular content generation.
    *   `word_count_target`:
        *   **Type**: Integer
        *   **Required**: No
        *   **Description**: The desired word count for the generated content. If not provided, defaults to values specified in the `constitution.md` (e.g., 500-1000 for section, 1000-2000 for chapter).
    *   `specific_requirements`:
        *   **Type**: String
        *   **Required**: No
        *   **Description**: Any additional, detailed instructions or constraints for content generation (e.g., "focus on ethical implications", "include a pseudocode example for path planning", "emphasize current research up to 2024").
    *   `citations_required`:
        *   **Type**: Boolean
        *   **Required**: No
        *   **Default**: `true`
        *   **Description**: Flag to indicate if external citations are mandatory for the generated content. Overrides constitution if set to `false` for specific contexts (e.g., intro paragraphs).
    *   `examples_required`:
        *   **Type**: Boolean
        *   **Required**: No
        *   **Default**: `true`
        *   **Description**: Flag to indicate if hands-on examples or real-world use cases are required.

## Output Contract: Content Generation Response

This defines the structure of the response after a content generation request is processed.

*   **Success Response**:
    *   `status`:
        *   **Type**: String
        *   **Value**: "success"
    *   `file_path`:
        *   **Type**: String
        *   **Description**: The absolute path(s) to the generated Markdown file(s).
    *   `content_summary`:
        *   **Type**: String
        *   **Description**: A brief, high-level summary of the generated content.
    *   `word_count`:
        *   **Type**: Integer
        *   **Description**: The actual word count of the primary generated content.
    *   `citations_list`:
        *   **Type**: List of Strings
        *   **Description**: A list of extracted, formatted citations found in the generated content.

*   **Error Response**:
    *   `status`:
        *   **Type**: String
        *   **Value**: "error"
    *   `error_code`:
        *   **Type**: String
        *   **Description**: A machine-readable code indicating the type of error.
        *   **Possible Values**:
            *   `CONSTITUTION_VIOLATION`: Generated content failed to adhere to constitutional rules.
            *   `INVALID_INPUT`: Provided parameters were invalid or insufficient.
            *   `GENERATION_FAILED`: Internal error during content generation process.
            *   `VALIDATION_FAILED`: Post-generation validation (e.g., word count, readability) failed.
            *   `UNKNOWN_ERROR`: Catch-all for unhandled exceptions.
    *   `message`:
        *   **Type**: String
        *   **Description**: A human-readable message describing the error.
    *   `details`:
        *   **Type**: String (Optional)
        *   **Description**: Additional technical details or debugging information for the error.