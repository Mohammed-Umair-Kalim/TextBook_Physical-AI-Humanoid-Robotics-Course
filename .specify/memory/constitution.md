<!--
---
Sync Impact Report
---
Version change: 0.0.0 → 1.0.0
Modified principles: 4 Core Principles added.
Added sections: Key Standards, Content Requirements, Constraints, Formatting Requirements, Success Criteria, Authoring Voice, System Behavior Instructions.
Removed sections: None
Templates requiring updates:
- ✅ .specify/templates/plan-template.md
- ✅ .specify/templates/spec-template.md
- ✅ .specify/templates/tasks-template.md
Follow-up TODOs: None
-->
# "Physical AI & Humanoid Robotics" Textbook Constitution

## Core Principles

### 1. Technical Accuracy
Use verified, up-to-date sources from robotics, AI, control systems, and embodied intelligence. Prioritize peer-reviewed research, academic robotics labs (MIT, Stanford, CMU, ETH), and reputable industry sources (OpenAI, Tesla, Figure AI, Boston Dynamics, Sanctuary AI).

### 2. Pedagogical Clarity
Explain all concepts as if teaching an undergraduate/graduate audience. Introduce foundational ideas before advanced ones. Use descriptions, analogies, and structured breakdowns.

### 3. Engineering Rigor
Maintain consistency with robotics standards: ROS/ROS2, SLAM, kinematics, control theory, perception pipelines, reinforcement learning, LLM-based planning.

### 4. Reproducibility
Every factual claim must be traceable to a citable source. All pseudocode and algorithms must describe replicable steps.

## Key Standards

- **Citation format**: APA 7
- **Source requirements**:
  - Minimum 50% peer-reviewed research or authoritative robotics literature
  - Remaining from credible industry technical reports, robotics companies, and standards docs
- **Writing style**:
  - Clear, academic, readable
  - Flesch-Kincaid grade 11–14
  - Concise, structured paragraphs
- **Content style**:
  - Consistent terminology across all chapters
  - All equations written in LaTeX format
  - All code samples language-labeled (Python, ROS2, C++, etc.)

## Content Requirements

The textbook must cover at minimum:

1.  Foundations of Physical AI
2.  Humanoid Robotics Fundamentals
3.  Perception Systems
4.  Control & Motion Planning
5.  Learning-Based Robotics
6.  LLM-based Robotics (2023–2025)
7.  Case Studies: Tesla Optimus, Figure 01, Digit, Sanctuary Phoenix, BD Atlas
8.  Hands-on Robotics Simulation
9.  Ethics, Safety, and Future of Physical AI

## Constraints

- Total textbook length target: 1000–5000 words
- Each chapter: 1000–2000 words
- Each section must be modular for Docusaurus, formatted as Markdown
- Include diagrams (describe them in text for later rendering)
- Zero plagiarism tolerance
- All chapters must pass factual verification
- No hallucinated citations (must be real, checkable sources)

## Formatting Requirements

- Output format for each file: Markdown (.md) compatible with Docusaurus v3
- Include:
  - frontmatter, slug, title
  - headings H2–H4
  - lists, tables, equations, code blocks
- Use consistent folder structure:
  docs/<chapter>/<section>.md

## Success Criteria

A chapter or output is considered successful when:

1.  All factual claims are accurately sourced
2.  Zero hallucinated references
3.  Meets textbook clarity standards for engineering education
4.  Internal consistency across definitions, terminology, and examples
5.  Content is technically correct and aligns with robotics standards
6.  No plagiarism detected
7.  Readable in Docusaurus with minimal editing
8.  Explainability: a motivated undergraduate can learn from it

## Authoring Voice

- Neutral academic tone
- Clear engineering communication
- Encouraging, structured, precise
- Avoid hype, speculation without sources, or marketing tone

## System Behavior Instructions

- Interpret all future spec-kit requests under this constitution
- Enforce all constraints automatically
- Validate outputs for accuracy before generating
- Ask for clarification only if required for correctness

## Governance

This constitution governs the creation of the "Physical AI & Humanoid Robotics" textbook. All content, tools, and processes must adhere to these principles and standards. Amendments to this constitution require a documented proposal, review, and approval process, ensuring that any changes align with the project's educational and technical goals.

**Version**: 1.0.0 | **Ratified**: 2025-12-06 | **Last Amended**: 2025-12-06