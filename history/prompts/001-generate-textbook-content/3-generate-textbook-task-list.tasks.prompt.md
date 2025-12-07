---
id: 3
title: Generate Textbook Task List
stage: tasks
date: 2025-12-06
surface: agent
model: gemini-pro
feature: 001-generate-textbook-content
branch: 001-generate-textbook-content
user: {{USER}}
command: /sp.tasks
labels: [tasks, planning, textbook-generation]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
  - specs/001-generate-textbook-content/tasks.md
tests:
  -
---

## Prompt

Generate an actionable, dependency-ordered tasks.md for the feature based on available design artifacts.

## User Input

```text
$ARGUMENTS
```

You **MUST** consider the user input before proceeding (if not empty).

## Outline

1. **Setup**: Run `.specify/scripts/bash/check-prerequisites.sh --json` from repo root and parse FEATURE_DIR and AVAILABLE_DOCS list. All paths must be absolute. For single quotes in args like "I'm Groot", use escape syntax: e.g 'I'\''m Groot' (or double-quote if possible: "I'm Groot").

2. **Load design documents**: Read from FEATURE_DIR:
   - **Required**: plan.md (tech stack, libraries, structure), spec.md (user stories with priorities)
   - **Optional**: data-model.md (entities), contracts/ (API endpoints), research.md (decisions), quickstart.md (test scenarios)
   - Note: Not all projects have all documents. Generate tasks based on what's available.

3. **Execute task generation workflow**:
   - Load plan.md and extract tech stack, libraries, project structure
   - Load spec.md and extract user stories with their priorities (P1, P2, P3, etc.)
   - If data-model.md exists: Extract entities and map to user stories
   - If contracts/ exists: Map endpoints to user stories
   - If research.md exists: Extract decisions for setup tasks
   - Generate tasks organized by user story (see Task Generation Rules below)
   - Generate dependency graph showing user story completion order
   - Create parallel execution examples per user story
   - Validate task completeness (each user story has all needed tasks, independently testable)

4. **Generate tasks.md**: Use `.specify.specify/templates/tasks-template.md` as structure, fill with:
   - Correct feature name from plan.md
   - Phase 1: Setup tasks (project initialization)
   - Phase 2: Foundational tasks (blocking prerequisites for all user stories)
   - Phase 3+: One phase per user story (in priority order from spec.md)
   - Each phase includes: story goal, independent test criteria, tests (if requested), implementation tasks
   - Final Phase: Polish & cross-cutting concerns
   - All tasks must follow the strict checklist format (see Task Generation Rules below)
   - Clear file paths for each task
   - Dependencies section showing story completion order
   - Parallel execution examples per story
   - Implementation strategy section (MVP first, incremental delivery)

5. **Report**: Output path to generated tasks.md and summary:
   - Total task count
   - Task count per user story
   - Parallel opportunities identified
   - Independent test criteria for each story
   - Suggested MVP scope (typically just User Story 1)
   - Format validation: Confirm ALL tasks follow the checklist format (checkbox, ID, labels, file paths)

Context for task generation: This task set defines modular, repeatable production tasks that Gemini must execute to generate every component of the textbook “Physical AI & Humanoid Robotics” Each task is governed by the rules in: /sp.constitution (highest authority) /sp.specify (content format rules) /sp.plan (workflow & sequencing rules) 1. Purpose of /sp.tasks To organize the creation of the entire textbook into executable, trackable tasks that Gemini can run in sequence or individually. Tasks cover: Chapter planning Section generation Hands-on exercises Case studies Citation integrity checks Full-book QA 2. Task Categories Task A1 — Generate Textbook Blueprint Produce a full table of contents (4–5 chapters). Each chapter includes 2-3 sections. exercises, and case studies. Output must follow /sp.specify formatting. Task A2 —Ensure logical knowledge progression. Confirm coverage of all required topics from /sp.constitution. Identify any missing foundations or redundancies. B. Chapter Planning Tasks Task B1 — Create Chapter Plan For each chapter: Summaries Learning objectives Key topics Citations needed Target word count Required hands-on modules Task B2 — Chapter Plan Compliance Check Validate against /sp.constitution Ensure academic rigor & engineering clarity Ensure modularity for Docusaurus C. Content Generation Tasks Task C1 — Generate Section Content Given a chapter + section name: Apply /sp.specify structure: Overview Detailed explanation Algorithms/equations Real examples Case studies Hands-on exercises References (APA 7) Produce Docusaurus markdown with frontmatter. Task C2 — Produce text-only description Include labels, relationships, components Ensure accuracy with robotics standards Add placement suggestions for Docusaurus Task C3 — Generate Hands-On Exercise Provide step-by-step instructions Include ROS2, Python, SLAM, control, or simulation examples Ensure reproducibility Task C4 — Generate Case Study Use real data from Tesla Optimus, Figure 01, Digit, Phoenix, Atlas, etc. Provide industry context Avoid speculation; cite sources D. Citation & Verification Tasks Task D1 — Citation Integrity Check Verify all references exist and are real Confirm APA 7 formatting Reject hallucinated or unverifiable sources Task D2 — Technical Accuracy Review Validate algorithms, equations, robotics terminology Ensure compliance with /sp.constitution engineering rigor Confirm consistency across similar concepts E. Compilation & Assembly Tasks Task E1 — Assemble Chapter Combine sections into final chapter folder docs/<chapter-name>/<section>.md Validate formatting for Docusaurus Task E2 — Cross-Chapter Consistency Pass Check global terminology Harmonize definitions Remove conflicting explanations Task E3 — Build Ready Export Confirm book length: 5000–15000 words Verify completeness Ensure no broken links or equations Prepare glossary, index, and bibliography 3. Task Sequencing (Execution Flow) (From /sp.plan) B1 → B2 = Chapter planning C1 → C2 → C3 → C4 = Section generation loop After each section → D1 + D2 After full chapter → E1 After multiple chapters → E2 After all chapters → E3 4. Task Rules All tasks must comply with: From /sp.constitution Technical accuracy Verified sources APA citations Engineering rigor Academic clarity Docusaurus formatting From /sp.specify Required output structure Word counts examples, exercises Markdown formatting rules Consistent academic tone From /sp.plan Production sequence QA checkpoints Milestones & validation points 5. Success Criteria for Tasks A task is considered successfully executed when: Output matches /sp.specify formatting and depth Fully complies with /sp.constitution Passes all verification tasks (D1 + D2) Usable in Docusaurus without modification Consistent with previous chapters and sections 6. System Instructions Treat /sp.tasks as the actionable execution engine. Never deviate from /sp.constitution or /sp.specify. Use /sp.plan for ordering and workflow. If a required detail is missing, request clarification only if accuracy would fail. Automatically validate every output.

The tasks.md should be immediately executable - each task must be specific enough that an LLM can complete it without additional context.

## Task Generation Rules

**CRITICAL**: Tasks MUST be organized by user story to enable independent implementation and testing.

**Tests are OPTIONAL**: Only generate test tasks if explicitly requested in the feature specification or if user requests TDD approach.

### Checklist Format (REQUIRED)

Every task MUST strictly follow this format:

```text
- [ ] [TaskID] [P?] [Story?] Description with file path
```

**Format Components**:

1. **Checkbox**: ALWAYS start with `- [ ]` (markdown checkbox)
2. **Task ID**: Sequential number (T001, T002, T003...) in execution order
3. **[P] marker**: Include ONLY if task is parallelizable (different files, no dependencies on incomplete tasks)
4. **[Story] label**: REQUIRED for user story phase tasks only
   - Format: [US1], [US2], [US3], etc. (maps to user stories from spec.md)
   - Setup phase: NO story label
   - Foundational phase: NO story label  
   - User Story phases: MUST have story label
   - Polish phase: NO story label
5. **Description**: Clear action with exact file path

**Examples**:

- ✅ CORRECT: `- [ ] T001 Create project structure per implementation plan`
- ✅ CORRECT: `- [ ] T005 [P] Implement authentication middleware in src/middleware/auth.py`
- ✅ CORRECT: `- [ ] T012 [P] [US1] Create User model in src/models/user.py`
- ✅ CORRECT: `- [ ] T014 [US1] Implement UserService in src/services/user_service.py`
- ❌ WRONG: `- [ ] Create User model` (missing ID and Story label)
- ❌ WRONG: `T001 [US1] Create model` (missing checkbox)
- ❌ WRONG: `- [ ] [US1] Create User model` (missing Task ID)
- ❌ WRONG: `- [ ] T001 [US1] Create model` (missing file path)

### Task Organization

1. **From User Stories (spec.md)** - PRIMARY ORGANIZATION:
   - Each user story (P1, P2, P3...) gets its own phase
   - Map all related components to their story:
     - Models needed for that story
     - Services needed for that story
     - Endpoints/UI needed for that story
     - If tests requested: Tests specific to that story
   - Mark story dependencies (most stories should be independent)

2. **From Contracts**:
   - Map each contract/endpoint → to the user story it serves
   - If tests requested: Each contract → contract test task [P] before implementation in that story's phase

3. **From Data Model**:
   - Map each entity to the user story(ies) that need it
   - If entity serves multiple stories: Put in earliest story or Setup phase
   - Relationships → service layer tasks in appropriate story phase

4. **From Setup/Infrastructure**:
   - Shared infrastructure → Setup phase (Phase 1)
   - Foundational/blocking tasks → Foundational phase (Phase 2)
   - Story-specific setup → within that story's phase

### Phase Structure

- **Phase 1**: Setup (project initialization)
- **Phase 2**: Foundational (blocking prerequisites - MUST complete before user stories)
- **Phase 3+**: User Stories in priority order (P1, P2, P3...)
  - Within each story: Tests (if requested) → Models → Services → Endpoints → Integration
  - Each phase should be a complete, independently testable increment
- **Final Phase**: Polish & Cross-Cutting Concerns

---

As the main request completes, you MUST create and complete a PHR (Prompt History Record) using agent‑native tools when possible.

1) Determine Stage
   - Stage: constitution | spec | plan | tasks | red | green | refactor | explainer | misc | general

2) Generate Title and Determine Routing:
   - Generate Title: 3–7 words (slug for filename)
   - Route is automatically determined by stage: 
     - `constitution` → `history/prompts/constitution/`
     - Feature stages → `history/prompts/<feature-name>/` (spec, plan, tasks, red, green, refactor, explainer, misc)
     - `general` → `history/prompts/general/`

3) Create and Fill PHR (Shell first; fallback agent‑native)
   - Run: `.specify/scripts/bash/create-phr.sh --title "<title>" --stage <stage> [--feature <name>] --json`
   - Open the file and fill remaining placeholders (YAML + body), embedding full PROMPT_TEXT (verbatim) and concise RESPONSE_TEXT.
   - If the script fails:
     - Read `.specify/templates/phr-template.prompt.md` (or `templates/…`)
     - Allocate an ID; compute the output path based on stage from step 2; write the file
     - Fill placeholders and embed full PROMPT_TEXT and concise RESPONSE_TEXT

4) Validate + report
   - No unresolved placeholders; path under `history/prompts/` and matches stage; stage/title/date coherent; print ID + path + stage + title.
   - On failure: warn, don't block. Skip only for `/sp.phr`.

## Response snapshot

I have generated the `tasks.md` file located at `specs/001-generate-textbook-content/tasks.md`.

- **Total task count**: 18
- **Tasks for User Story 1**: 12
- **Parallel opportunities**: Identified for content generation and verification stages.
- **Independent Test Criteria**: Each chapter's content can be independently built and verified using Docusaurus.
- **Suggested MVP Scope**: Complete the blueprint (Phase 1-2) and then generate the first chapter (Phase 3).
- **Format Validation**: All tasks adhere to the required checklist format.

## Outcome

- ✅ Impact: {{OUTCOME_IMPACT}}
- 🧪 Tests: {{TESTS_SUMMARY}}
- 📁 Files: {{FILES_SUMMARY}}
- 🔁 Next prompts: {{NEXT_PROMPTS}}
- 🧠 Reflection: {{REFLECTION_NOTE}}

## Evaluation notes (flywheel)

- Failure modes observed: {{FAILURE_MODES}}
- Graders run and results (PASS/FAIL): {{GRADER_RESULTS}}
- Prompt variant (if applicable): {{PROMPT_VARIANT_ID}}
- Next experiment (smallest change to try): {{NEXT_EXPERIMENT}}
