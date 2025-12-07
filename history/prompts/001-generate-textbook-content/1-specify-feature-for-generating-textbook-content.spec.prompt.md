---
id: 1
title: Specify Feature for Generating Textbook Content
stage: spec
date: 2025-12-06
surface: agent
model: gemini-cli
feature: 001-generate-textbook-content
branch: 001-generate-textbook-content
user: user
command: /sp.specify
labels: ["textbook-generation", "content-creation"]
links:
  spec: specs/001-generate-textbook-content/spec.md
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-generate-textbook-content/spec.md
  - specs/001-generate-textbook-content/checklists/requirements.md
tests:
  - none
---

## Prompt

Generate the requested textbook content according to the following specifications and constraints derived from the project constitution: 1. Purpose of /sp.specify This specification defines exactly what content Gemini should generate, at what depth, style, and format, based on the governing standards of /sp.constitution. Use this specification every time the user asks for a chapter, subsection, lesson, module, code sample, or explanation for the textbook “Physical AI & Humanoid Robotics” 2. Output Requirements All generated content must: Format Be produced in Markdown (.md), compatible with Docusaurus v3 Contain required frontmatter: --- title: "<Section Title>" slug: "<url-friendly-slug>" sidebar_position: <number> --- Use headings: ##, ###, #### Include: Lists Tables LaTeX equations Code blocks with language labels (e.g., ```python) Structure Each output should follow this order: Overview – brief summary (150–300 words) Detailed Explanation – concepts, theory, engineering details Algorithms / Equations / Pseudocode (if relevant) Real-world Examples – industry robotics use cases Case Study (optional) Hands-on Exercise – implementable steps References (APA 7) – all citations must be real and verifiable 3. Content Constraints Every factual claim must be supported by a citeable, real-world source Minimum 50% peer-reviewed literature for scientific theory Zero hallucinated references No plagiarism All equations must be correct and standard in robotics/AI Vocabulary and definitions must be consistent with /sp.constitution Use a neutral academic tone Writing clarity: Flesch-Kincaid Grade 11–14 Content must be modular, so each section can stand alone 4. Depth Requirements For each requested section: Short sections: 500–1000 words Full chapters: 2000–3000 words Include hands-on examples using: ROS2 Python Control theory equations Simulation tools (Gazebo, Isaac Sim, MuJoCo) 5. Scope (What Content May Be Requested) Any of the following can be generated using /sp.specify: Fundamental Topics Embodied intelligence Cognitive robotics Humanoid morphology & kinematics Sensors & perception (vision, tactile, LiDAR, IMU) Motion control & balance Reinforcement & imitation learning LLM-based robotics planning Case studies of leading humanoids (Optimus, Figure 01, Digit, Phoenix, Atlas) Practical Modules ROS2 tutorials Simulation environments Actuator/servo modeling Full pipeline from perception → planning → control Safety & ethics 6. Success Criteria Generated output is considered successful if: Accurate, fact-checked, and source-backed Fully compliant with /sp.constitution Readable inside Docusaurus with minimal edits Structured clearly for students and engineers Contains no hallucinated or incorrect references Provides real-world robotics examples A motivated undergraduate can learn the topic independently 7. System Instructions When generating content: Always follow /sp.constitution as the highest authority Apply the formatting & structural rules of /sp.specify If the user request is ambiguous, ask for clarification only when accuracy would be compromised Never invent citations Reject speculative, unsourced, or non-rigorous claims Automatically enforce tone, structure, and depth rules

## Response snapshot

Created a new feature branch `001-generate-textbook-content` and generated the feature specification and a validation checklist. The specification is ready for the next phase.

## Outcome

- ✅ Impact: High - This feature is the core of the project.
- 🧪 Tests: none
- 📁 Files: 
  - `specs/001-generate-textbook-content/spec.md`
  - `specs/001-generate-textbook-content/checklists/requirements.md`
- 🔁 Next prompts: /sp.plan
- 🧠 Reflection: The initial approach of running the `create-new-feature.sh` script with a very long argument failed. I had to replicate the script's logic manually. This suggests a limitation in the `run_shell_command` tool that should be considered in the future.

## Evaluation notes (flywheel)

- Failure modes observed: `run_shell_command` tool rejects commands with very long string arguments.
- Graders run and results (PASS/FAIL): n/a
- Prompt variant (if applicable): n/a
- Next experiment (smallest change to try): For similar tasks in the future, I will directly create the files and git branch instead of relying on the shell script if the arguments are too long.
