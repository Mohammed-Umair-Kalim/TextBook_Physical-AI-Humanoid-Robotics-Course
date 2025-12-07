# Quickstart Guide: Textbook Content Generation

This guide provides a quick overview of how to interact with the agent to generate content for the "Physical AI & Humanoid Robotics" textbook. The content generation process is driven by structured commands and adherence to the project's constitution and specifications.

## How to Request Content Generation

To initiate content generation, you will primarily use a command similar to `/sp.specify` (or an equivalent direct instruction to the agent). This command allows you to specify the type of content, its topic, and any specific requirements.

### Basic Content Generation

The core interaction involves specifying the `content_type` and `topic`.

**Example: Generating a Chapter Outline**

To generate an outline for a new chapter on "Embodied Intelligence":

```
/sp.specify content_type="outline" topic="Embodied Intelligence"
```

**Example: Generating a Full Chapter**

To generate a complete chapter on "Cognitive Robotics":

```
/sp.specify content_type="chapter" topic="Cognitive Robotics" word_count_target="2500"
```

**Example: Generating a Section within a Chapter**

To generate a section on "ROS2 Tutorials" within the "Humanoid Morphology" chapter:

```
/sp.specify content_type="section" topic="ROS2 Tutorials" chapter_context="Humanoid Morphology" word_count_target="750"
```

### Advanced Content Generation

You can provide more detailed instructions using the `specific_requirements` parameter for finer control over the generated output.

**Example: Generating a Section with Specific Focus**

To generate a section on "Perception Systems" focusing on recent advancements in sensor fusion:

```
/sp.specify content_type="section" topic="Perception Systems" specific_requirements="Focus on recent advancements in sensor fusion and include a brief comparison of LiDAR and camera data processing techniques."
```

### Available Parameters

Here are the key parameters you can use when requesting content generation:

*   `content_type` (Required):
    *   `"outline"`: Generates a high-level structure for a chapter or the entire textbook.
    *   `"chapter"`: Generates a full chapter, including all its sections based on the content structure defined in the constitution.
    *   `"section"`: Generates a single section of a chapter.
*   `topic` (Required for "chapter" or "section" types): The main subject or title of the content.
*   `chapter_context` (Optional): Provide the title of the parent chapter when generating a section to ensure contextual relevance.
*   `section_context` (Optional): Provide the title or a summary of the parent section if generating sub-sections or specific content blocks.
*   `word_count_target` (Optional): Specify a desired word count. If omitted, default word counts from the constitution will apply (e.g., 500-1000 for sections, 1000-2000 for chapters).
*   `specific_requirements` (Optional): A free-form text field for any additional, detailed instructions, stylistic preferences, or constraints (e.g., "include pseudocode", "cite three recent papers", "avoid complex mathematical derivations").
*   `citations_required` (Optional, default: `true`): Set to `false` to suppress citation requirements for specific content (use sparingly).
*   `examples_required` (Optional, default: `true`): Set to `false` to suppress hands-on examples for specific content.

## Expected Output

Upon successful generation, the agent will provide:

*   A `status` of "success".
*   The `file_path` to the generated Markdown file(s).
*   A `content_summary` of what was generated.
*   The actual `word_count` of the output.
*   A `citations_list` of all references used.

In case of any issues, an `error` status will be returned with an `error_code` and a `message` explaining the problem.

## Next Steps

After generation, you can review the content, request revisions, or instruct the agent to build the Docusaurus site to preview the content. Remember, all generated content is subject to the stringent quality gates defined in the project's constitution.