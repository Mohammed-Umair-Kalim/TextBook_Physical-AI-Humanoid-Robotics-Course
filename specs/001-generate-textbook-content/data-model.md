# Data Model: Textbook Content Generation

This document outlines the data model for the "Physical AI & Humanoid Robotics" textbook content, based on the `spec.md` and `constitution.md`. The entities described here represent the logical components of the textbook content that will be generated and managed.

## Entities

### Entity: Textbook

Represents the entire textbook project.

*   **Name**: "Physical AI & Humanoid Robotics"
*   **Fields**:
    *   `title`: String - The full title of the textbook (e.g., "Physical AI & Humanoid Robotics").
    *   `word_count_target`: Integer - The target total word count for the entire textbook (1000-5000 words as per constitution, to be adjusted in plan for full textbook).
    *   `chapters`: List of `Chapter` entities - The collection of chapters within the textbook.

### Entity: Chapter

Represents a major division of the textbook content.

*   **Name**: e.g., "Embodied Intelligence"
*   **Fields**:
    *   `title`: String - The title of the chapter.
    *   `slug`: String - A URL-friendly identifier for the chapter (for Docusaurus).
    *   `sidebar_position`: Integer - The position of the chapter in the Docusaurus sidebar navigation.
    *   `word_count_target`: Integer - The target word count for this specific chapter (1000-2000 words).
    *   `sections`: List of `Section` entities - The sub-divisions within the chapter.
    *   `learning_objectives`: List of Strings - Key learning goals for the chapter.
    *   `required_examples`: List of Strings - Descriptions or types of examples required for the chapter.
    *   `case_studies`: List of Strings - Descriptions or types of case studies required for the chapter.
*   **Relationships**:
    *   Belongs to `Textbook`.
    *   Contains multiple `Section` entities.
*   **Validation Rules (from `spec.md` & `constitution.md`)**:
    *   `word_count_target`: Must be between 1000 and 2000 words.

### Entity: Section

Represents a modular unit of content within a chapter. This will correspond to individual Markdown files in Docusaurus.

*   **Name**: e.g., "ROS2 Tutorials"
*   **Fields**:
    *   `title`: String - The title of the section.
    *   `slug`: String - A URL-friendly identifier for the section (for Docusaurus).
    *   `sidebar_position`: Integer - The position of the section in the Docusaurus sidebar navigation.
    *   `content_structure`: List of Strings - The required sub-sections/content types within a section, in order: "Overview", "Detailed Explanation", "Algorithms/Equations/Pseudocode", "Real-world Examples", "Case Study (optional)", "Hands-on Exercise", "References".
    *   `word_count_target`: Integer - The target word count for this specific section (500-1000 words).
    *   `citations`: List of `Citation` entities - References used within the section.
    *   `code_samples`: List of `CodeSample` entities - Programming code examples.
    *   `diagrams`: List of `DiagramDescription` entities - Textual descriptions of diagrams to be rendered later.
*   **Relationships**:
    *   Belongs to `Chapter`.
*   **Validation Rules (from `spec.md` & `constitution.md`)**:
    *   `word_count_target`: Must be between 500 and 1000 words.
    *   `content_structure`: Must strictly follow the defined order and include all mandatory components.
    *   `style`: Must adhere to a neutral academic tone and a Flesch-Kincaid Grade level of 11-14.
    *   `plagiarism`: Must contain zero plagiarized content.

### Entity: Citation

Represents a source reference used to support factual claims.

*   **Fields**:
    *   `text`: String - The full citation in APA 7 format.
    *   `url`: String (Optional) - A verifiable URL to the source, if available.
    *   `type`: Enum (e.g., "Peer-reviewed", "Industry report", "Standards doc", "Academic Book") - Categorization of the source.
*   **Relationships**:
    *   Associated with `Section`.
*   **Validation Rules (from `spec.md` & `constitution.md`)**:
    *   Must be real and verifiable.
    *   APA 7 format.
    *   Minimum 50% peer-reviewed research or authoritative robotics literature.
    *   No hallucinated citations.

### Entity: CodeSample

Represents a block of programming code provided as an example.

*   **Fields**:
    *   `language`: String - The programming language (e.g., "Python", "C++", "ROS2 XML").
    *   `code`: String - The actual code snippet.
    *   `description`: String - A brief explanation of the code's purpose and functionality.
*   **Relationships**:
    *   Associated with `Section`.
*   **Validation Rules (from `spec.md`)**:
    *   Language must be labeled.
    *   Must use ROS2, Python, control theory equations, or simulation tools.

### Entity: DiagramDescription

Represents a textual description of a diagram that would be rendered visually.

*   **Fields**:
    *   `text_description`: String - A detailed textual description sufficient to generate the diagram later (e.g., "Flowchart showing ROS2 node communication: node A (publisher) -> topic -> node B (subscriber)").
*   **Relationships**:
    *   Associated with `Section`.