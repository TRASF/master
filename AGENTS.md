This consolidated version keeps the broader development-cycle coverage from your current rules while making Ponytail a mandatory, visible two-stage workflow for non-trivial implementation and refactoring. 

# Repository Agent Operating Rules

## Purpose

These rules define how an agent operates throughout the software development
lifecycle in this repository.

They apply to:

* implementation;
* debugging;
* troubleshooting;
* refactoring;
* code review;
* testing;
* builds;
* documentation;
* research;
* learning and explanation;
* configuration;
* infrastructure;
* deployment;
* maintenance;
* incident investigation.

A task prompt defines the current objective, scope, constraints, and acceptance
criteria. It does not automatically suspend these rules.

The user does not need to repeat repository navigation, tool routing, bounded
retrieval, testing, safety, Ponytail, or reporting requirements in every task.

Repository-specific requirements may be added near the end of this file.

---

## Instruction precedence

Apply instructions in this order:

1. system and safety requirements;
2. active policy hooks and tool restrictions;
3. more specific directory-level or repository-level instructions;
4. the current user request;
5. these repository defaults.

When instructions conflict, follow the more specific and more restrictive rule
unless an authorized instruction explicitly overrides it.

Active policy hooks are authoritative for tool execution.

Do not bypass a restriction through:

* equivalent commands;
* aliases;
* alternate tools;
* subprocess wrappers;
* another programming language;
* generated helper scripts;
* indirect filesystem access.

---

## Core operating principles

Use existing conversation context, repository context, prior confirmed findings,
and available code intelligence before retrieving more information.

Default to a targeted task.

Do not rediscover the repository because the session is new.

Do not inspect files merely to become familiar with the project.

Do not collect context without a concrete question that the context will
answer.

Stop retrieving information when sufficient evidence is available.

Prefer:

* the smallest reliable evidence set;
* the smallest sufficient implementation;
* the smallest relevant test;
* the smallest safe write scope;
* direct solutions over speculative abstractions.

Do not perform unrelated cleanup during focused work.

Do not narrate routine retrieval steps unless they:

* affect the conclusion;
* reveal a material risk;
* expand authorized scope;
* identify a failure;
* require user action;
* block further work.

Distinguish clearly between:

* confirmed;
* inferred;
* assumed;
* unverified;
* blocked;
* failed.

Never present inference as confirmed behavior.

Never claim that a test passed, an analysis was performed, behavior was
preserved, or side effects did not occur without evidence.

---

## Zero-tool completion

When the current conversation already contains enough information to answer,
plan, request authorization, or identify the next action, use no repository
tools.

Do not inspect the repository to reconfirm information already established in
the active conversation.

Tasks that often require no repository access include:

* asking whether to begin a new phase;
* explaining that a phase requires authorization;
* planning based on supplied requirements;
* answering a conceptual question;
* explaining previously confirmed behavior;
* rewriting or drafting text;
* identifying the next action from an established plan.

Before the first tool call, determine:

```text
Can this task be completed accurately from the active conversation and
already-confirmed project context?
```

If yes, do not call repository tools.

---

## Task classification

Silently classify the request before using tools.

### Explanation or learning

Use supplied context and general knowledge first.

Inspect the repository only when:

* the user asks how this repository implements the concept;
* repository-specific behavior must be verified;
* exact code evidence is required.

### Targeted development

Work only on the named feature, file, symbol, test, configuration, or behavior.

This is the default classification for implementation work.

### Diagnostic debugging

Begin with the exact:

* symptom;
* traceback;
* failed test;
* error message;
* log event;
* reproduction command.

Expand only to components implicated by evidence.

### Troubleshooting

Determine whether the failure belongs to:

* command usage;
* environment;
* dependencies;
* configuration;
* permissions;
* networking;
* storage;
* services;
* runtime;
* build tooling;
* application code.

Do not assume every failure requires a code change.

### Structural or refactoring work

Use semantic navigation, execution-flow analysis, dependency analysis, and
impact analysis before changing shared structure.

Preserve behavior unless the task explicitly authorizes a behavior change.

Ponytail is mandatory for non-trivial refactoring.

### Code review

Inspect only the requested changes and the code needed to understand their
effects.

Prioritize:

* correctness;
* safety;
* regressions;
* compatibility;
* missing verification;
* maintainability.

### Operational work

Treat these as side-effecting:

* deployments;
* infrastructure changes;
* remote jobs;
* package changes;
* migrations;
* service restarts;
* destructive commands.

Verify target, environment, scope, impact, and recovery path first.

### Exhaustive work

Repository-wide discovery is allowed only when the user explicitly requests:

* an audit;
* inventory;
* architecture analysis;
* migration assessment;
* security review;
* repository-wide refactor;
* another genuinely exhaustive task.

A long plan does not automatically authorize exhaustive inspection.

---

## Scope control

Only the current objective, phase, and step are authorized.

Later phases in a plan are reference material, not permission to inspect,
modify, execute, or test their components.

Do not inspect adjacent files or subsystems because they might be useful.

Expand scope only to resolve a specific:

* unknown symbol;
* caller or callee;
* dependency;
* configuration source;
* data flow;
* contradiction;
* failing test;
* runtime failure;
* compatibility requirement;
* missing acceptance criterion.

When the authorized work is complete, stop.

Do not begin the next phase without explicit authorization.

When a task is underspecified but can be completed safely, make the narrowest
reasonable assumption and state it.

Ask for clarification only when competing interpretations would produce
materially different, destructive, or unsafe outcomes.

---

## Before every tool call

Silently verify:

1. Is the call necessary for the current objective?
2. Is the target exact and task-specific?
3. Is the answer already present in current context?
4. Is this the narrowest suitable tool?
5. Would a symbol, reference, or graph query be better than a file read?
6. Is the output bounded?
7. Has the same unchanged information already been retrieved?
8. Is the operation inside the authorized scope?
9. Is the operation destructive, expensive, or externally visible?
10. Can a smaller query, range, test, or command answer the question?
11. Could the command produce continuous or high-volume output?
12. Does the command require explicit authorization?
13. Will the call retrieve evidence, or merely gather more context?

Do not narrate this self-check.

---

## Discovery and navigation routing

Do not use file reads as the default method of code discovery.

Choose the retrieval method according to the question.

### Existing task and planning information

Use the current conversation and established task state first.

Do not search the repository for:

* phase names;
* task names;
* acceptance criteria;
* authorization status;
* requirements already in the conversation;
* conclusions already confirmed in the current session.

Do not read these through tools when their relevant contents are already loaded
or present in the conversation:

* `AGENT.md`;
* `AGENTS.md`;
* `CLAUDE.md`;
* `.agent/PROJECT.md`;
* `.agent/STATE.md`;
* equivalent instruction or state files.

Read an instruction or state file only when:

* the task directly concerns the file;
* a nested directory may contain more specific instructions;
* the host did not load repository instructions;
* an instruction conflict must be resolved;
* required project state is absent from the conversation.

### Architecture and execution flow

Use a repository graph or semantic code-intelligence tool for:

* unfamiliar implementations;
* execution paths;
* subsystem relationships;
* callers and callees;
* shared consumers;
* dependency analysis;
* blast radius;
* entry-point tracing;
* configuration-loading flows;
* data flow;
* state transitions;
* cross-file refactoring;
* public interface analysis.

When GitNexus is available, use:

For understanding:

```text
query
→ context
→ bounded implementation read
```

For shared or structural changes:

```text
query
→ context
→ impact
→ bounded implementation read
```

Use `rename` for code-symbol renaming when available.

Use change detection for non-trivial structural changes when the installed
integration exposes it.

Do not query GitNexus with planning labels such as:

* `Phase 1`;
* `Phase 2`;
* `Phase 3`;
* `next phase`;
* `current migration`.

Translate planning concepts into concrete code questions.

Bad:

```text
Query GitNexus for Phase 3.
```

Good:

```text
Locate training entry points, configuration consumers, dataset construction,
preprocessing initialization, and execution flows for pretraining, linear
probing, and fine-tuning.
```

Treat symbols, paths, callers, flows, relationships, and risk information
returned by the MCP response as the retrieval result.

Consume the structured MCP response directly.

Do not open host-generated MCP, brain, cache, spill, or temporary file URIs to
recover the same response in another format.

When an MCP result is too broad or incomplete, repeat the request with narrower
parameters.

Do not install, update, configure, or reindex GitNexus unless the task
explicitly concerns GitNexus maintenance.

If an index appears stale, report the evidence.

### Exact symbol questions

When language-server or symbol-navigation tools are available, use them for:

* definitions;
* references;
* implementations;
* incoming calls;
* outgoing calls;
* type information;
* diagnostics;
* document symbols;
* workspace symbols;
* safe symbol rename.

Do not read a complete file to locate one known symbol.

### Structural code patterns

Use syntax-aware or AST search when available for:

* API call patterns;
* decorators;
* inheritance patterns;
* exception-handler structures;
* repeated constructs;
* import patterns;
* deprecated API usage.

Use text search only when syntax-aware retrieval is unavailable or the target
is textual rather than structural.

### Textual information

Use bounded exact search for:

* configuration keys;
* error messages;
* environment variables;
* documentation;
* test names;
* logs;
* generated strings;
* command-line flags;
* literal paths.

Restrict searches by:

* exact file;
* known subsystem;
* file type;
* exact directory;
* result count;
* exact term.

### Code reads

Read code only after discovery identifies the exact symbol or range required.

A normal targeted sequence should usually require:

* one focused graph or symbol query;
* one context or impact query when needed;
* one bounded implementation read.

Additional retrieval must answer a specific unresolved question.

Do not continue retrieval merely to gather more context.

A second code range must answer a question not resolved by the first.

Do not read complete implementation and test files merely because they exist or
because their tests passed.

### Tool fallback order

Use this order:

1. current conversation and established task state;
2. repository graph or semantic index;
3. language-server symbol navigation;
4. syntax-aware structural search;
5. bounded exact-text search;
6. bounded file read;
7. filesystem discovery.

Do not move to a broader level until the previous level is insufficient.

---

## Repository discovery

Do not begin normal work with broad repository discovery.

Avoid cold-start actions such as:

* recursive directory listings;
* `tree`;
* repository-wide `find`;
* repository-wide `fd`;
* `git ls-files`;
* `rg --files`;
* recursive globbing;
* repository-wide line counts;
* reading all configuration files;
* reading all documentation;
* reading all tests;
* reading complete large source files;
* checking every dependency;
* checking every tool;
* inspecting hooks or MCP configuration when unrelated;
* running `git status` merely to understand the repository.

Do not reproduce broad discovery through:

* Python filesystem traversal;
* Node.js filesystem traversal;
* shell loops;
* subprocess wrappers;
* alternate Git commands;
* generated manifests;
* another tool with equivalent behavior.

Broad discovery is acceptable only when:

* the user explicitly requests exhaustive work;
* repository structure is the subject of the task;
* targeted discovery failed and broader scope is justified;
* no semantic or indexed navigation capability is available.

When broad discovery is justified, bound it by:

* directory;
* depth;
* subsystem;
* file type;
* result count;
* exact question.

---

## Code intelligence and impact analysis

Use semantic tools for:

* unfamiliar implementations;
* execution flows;
* callers and callees;
* shared consumers;
* dependency relationships;
* public API changes;
* entry-point changes;
* cross-file refactors;
* configuration-loading paths;
* state and data flow;
* symbol renaming;
* blast-radius analysis.

Do not use semantic repository tools for:

* conceptual questions;
* comments;
* documentation-only edits;
* fixtures;
* exact known text files;
* formatting;
* isolated value changes;
* trivial local changes with confirmed callers.

Run available impact analysis before changing:

* public APIs;
* shared functions, classes, or methods;
* entry points;
* symbols with unknown callers;
* cross-file behavior;
* configuration-loading paths;
* persistence formats;
* schemas;
* protocols;
* deployment behavior;
* security-sensitive behavior;
* symbols used by several subsystems;
* code-symbol names.

Impact analysis is usually unnecessary for:

* documentation;
* comments;
* formatting;
* fixtures;
* test data;
* isolated values;
* local private helpers with confirmed callers;
* exact-path non-symbol edits.

Warn before proceeding when impact analysis reports high or critical risk.

Do not claim impact analysis occurred unless the operation appears in the
execution record.

---

## File reading

Prefer symbol-level retrieval for:

* functions;
* classes;
* methods;
* references;
* callers;
* callees;
* imports;
* execution flows.

Use bounded text reads for:

* exact strings;
* errors;
* configuration keys;
* test cases;
* documentation;
* logs;
* generated output.

Follow the active hook’s read limit.

When no repository-specific limit exists, default to approximately 200 lines or
fewer per read.

Do not read a complete large file unless:

* the complete file is explicitly under review;
* the file itself is the task target;
* symbolic retrieval is unavailable;
* bounded retrieval was insufficient;
* whole-file consistency must be verified.

Do not read the same unchanged content twice.

A later read is valid only when it:

* retrieves unseen content;
* verifies a specific edit;
* resolves a specific unanswered question.

Do not reconstruct a large file through many consecutive reads.

Before a second range from one file, identify the unresolved question requiring
it.

After two bounded reads from one file, reconsider whether a symbol query, exact
search, changed hunk, or structural query is more efficient.

Do not inspect neighboring files preemptively.

Do not read host-internal:

* agent files;
* MCP spill files;
* cache files;
* brain files;
* hidden reasoning files;
* temporary internal reports;

unless the task explicitly concerns those systems and access is authorized.

---

## Search behavior

Use exact:

* symbols;
* strings;
* error messages;
* configuration keys;
* paths;
* test names.

Text searches should normally target:

* one exact file;
* one known directory;
* one subsystem;
* one exact term;
* a bounded result count.

Do not run repository-wide searches for broad terms such as:

* `main`;
* `config`;
* `model`;
* `service`;
* `test`;
* `error`;
* `TODO`;
* `data`;
* `train`;
* `phase`;

unless exhaustive work requires it.

A search with no result does not automatically justify broader discovery.

Refine the term, inspect a known caller, query a symbol, or identify another
concrete evidence source.

Do not repeat equivalent searches with superficial wording changes.

---

# Mandatory Ponytail Workflow

## Purpose

Ponytail is the required simplicity and anti-overengineering workflow for
non-trivial implementation and refactoring.

Ponytail is not:

* an output compressor;
* a substitute for testing;
* a substitute for semantic navigation;
* a substitute for impact analysis;
* a substitute for compatibility analysis;
* a general statement that code is simple.

Reading a Ponytail skill file alone does not count as using Ponytail.

Mentioning Ponytail in a final report does not count as completing a Ponytail
review.

## When Ponytail is mandatory

Use the full Ponytail workflow when a task:

* adds a source module;
* adds a class;
* adds two or more functions;
* introduces a helper;
* introduces a wrapper;
* introduces an adapter;
* introduces a factory;
* introduces a registry;
* introduces a service;
* introduces an interface;
* introduces a dependency;
* adds configuration structure;
* changes module boundaries;
* moves responsibility between components;
* changes a public or shared interface;
* removes or replaces existing code;
* performs a cross-file refactor;
* is explicitly described as refactoring;
* produces a non-trivial source-code diff.

Ponytail is normally optional for:

* comments;
* spelling;
* formatting;
* documentation-only edits;
* isolated value changes;
* one-line fixes without structural effects;
* fixture content;
* generated files.

When uncertain, treat Ponytail as mandatory.

## Required stages

A mandatory Ponytail workflow contains two visible stages:

```text
Pre-change Ponytail analysis
        ↓
Implementation and focused tests
        ↓
Post-change Ponytail review
        ↓
Final affected tests
```

Both stages must appear in the execution record.

---

## Stage 1 — Pre-change Ponytail analysis

Before writing code, activate or follow the installed Ponytail implementation
or planning skill.

The analysis must produce a compact decision record:

```text
Pre-change Ponytail analysis

Existing behavior that can be reused:
Standard-library alternatives:
Installed-library alternatives:
New abstractions considered:
Why each proposed abstraction is necessary:
Simpler designs considered:
Code that can be deleted or consolidated:
Speculative features rejected:
Selected smallest sufficient design:
```

Before adding any module, class, function, helper, wrapper, adapter, service,
factory, registry, interface, configuration layer, or dependency, answer:

1. Does equivalent behavior already exist?
2. Can an existing implementation be reused?
3. Can an existing function be extended safely?
4. Can the standard library provide the behavior?
5. Can an installed dependency provide the behavior?
6. Can the change remain local to an existing module?
7. Is a function sufficient instead of a class?
8. Is a direct call sufficient instead of a wrapper?
9. Does the abstraction have at least two confirmed consumers?
10. Is the abstraction required by the current task?
11. Is it being introduced only for a hypothetical future phase?
12. Can existing code be removed instead of adding new code?

Do not begin implementation until the smallest sufficient design has been
identified.

Do not create extension points without a confirmed current consumer.

Do not add architecture solely because a future task may need it.

### Stage 1 evidence

The execution record must contain one of:

* invocation of the installed Ponytail planning or implementation skill;
* invocation of an installed Ponytail pre-change command;
* an explicitly labeled `Pre-change Ponytail analysis` containing the required
  decision fields.

Reading `SKILL.md` without applying its procedure is insufficient.

---

## Stage 2 — Post-change Ponytail review

After implementation and focused tests pass, inspect only the task-bounded
diff.

Activate or follow the installed diff-focused Ponytail review skill when
available.

Review the actual changed code for:

* unnecessary modules;
* unnecessary classes;
* one-use helpers;
* pass-through wrappers;
* duplicate logic;
* duplicate validation;
* duplicate parsing;
* speculative configuration;
* speculative extensibility;
* unnecessary adapters;
* unnecessary factories;
* unnecessary registries;
* excessive indirection;
* dead code;
* compatibility code without confirmed consumers;
* custom code already provided by the standard library;
* custom code already provided by an installed dependency;
* comments compensating for unnecessarily complex code.

Each concrete candidate must be reported as:

```text
Candidate:
Location:
Why it may be unnecessary:
Safe simplification:
Behavior that must remain unchanged:
Decision: APPLY | REJECT | DEFER
Reason:
```

A report containing only:

```text
Ponytail review completed.
```

is invalid.

Finding one unused helper does not prove a complete Ponytail review unless the
review explicitly covers all required categories.

---

## Applying Ponytail simplifications

Automatically apply a recommendation only when it:

* remains inside authorized scope;
* preserves required behavior;
* preserves public compatibility;
* preserves validation;
* preserves error handling;
* preserves deterministic behavior;
* does not weaken security;
* does not remove required observability;
* is covered by focused tests.

Before applying a recommendation that changes:

* public structure;
* configuration shape;
* compatibility behavior;
* module boundaries;
* external interfaces;

obtain authorization when that structural change is not already authorized by
the task.

After applying simplifications:

1. rerun the exact affected tests;
2. rerun relevant parity or compatibility tests;
3. inspect the final task-bounded diff;
4. confirm the simplification introduced no replacement abstraction;
5. record the final decision.

---

## Ponytail completion gate

A mandatory Ponytail workflow is complete only when:

* pre-change analysis is visible;
* the smallest sufficient design is stated;
* reuse opportunities are documented;
* new abstractions have explicit justification;
* speculative features are rejected explicitly;
* the post-change review examines the actual diff;
* concrete candidates and decisions are reported;
* accepted simplifications are applied;
* rejected or deferred recommendations include reasons;
* affected tests are rerun after simplification.

When any required item is absent, Ponytail status is:

```text
UNVERIFIED
```

Do not report Ponytail as `PASS`.

Successful tests do not compensate for an absent Ponytail review.

---

## Ponytail reporting format

For every mandatory Ponytail task, report:

```text
Ponytail status: PASS | FAIL | UNVERIFIED

Pre-change analysis:
- Reused:
- Added:
- Rejected as unnecessary:
- Selected design:

Post-change review:
- Candidates examined:
- Applied:
- Rejected:
- Deferred:

Verification after simplification:
- Commands:
- Results:
```

Do not claim Ponytail was used unless the execution record supports the claim.

---

## Implementation workflow

For normal development:

1. Confirm the objective.
2. Confirm acceptance criteria.
3. Identify the exact implementation target.
4. Retrieve only required context.
5. Assess impact when shared behavior is involved.
6. Identify existing behavior that can be reused.
7. Run pre-change Ponytail analysis when required.
8. Establish the smallest relevant baseline test.
9. Design the smallest sufficient change.
10. Implement in small, coherent steps.
11. Run the smallest relevant verification.
12. Inspect the task-bounded diff.
13. Run the post-change Ponytail review when required.
14. Apply safe, in-scope simplifications.
15. Rerun affected tests.
16. Report verified results and risks.

Do not introduce unrelated cleanup.

Record unrelated defects separately unless they block the task.

Do not claim completion until acceptance criteria are verified.

Do not create speculative extension points.

Prefer direct, explicit implementations over frameworks when a framework is not
required.

A non-trivial implementation is not complete when mandatory Ponytail stages are
missing.

---

## Debugging workflow

Begin with evidence, not repository exploration.

Use this sequence:

1. Capture the exact symptom.
2. Preserve the reproduction command or triggering action.
3. Confirm whether the failure is reproducible.
4. Identify the first meaningful error.
5. Determine the smallest implicated component.
6. Form explicit hypotheses.
7. Test the cheapest discriminating hypothesis first.
8. Inspect only implicated code or configuration.
9. Fix the root cause with the smallest justified change.
10. Add or update a regression test.
11. Rerun the original reproduction.
12. Run the smallest relevant surrounding tests.
13. Use Ponytail when the fix introduces non-trivial structure.

Do not change code because it merely looks suspicious.

Do not apply several speculative fixes simultaneously.

Do not hide failures using:

* broad exception handling;
* arbitrary retries;
* warning suppression;
* disabled validation;
* ignored exit codes;
* unverified fallback behavior.

When reproduction is impossible, state:

* what was attempted;
* what was observed;
* what remains unknown;
* what is needed to reproduce it.

---

## Troubleshooting workflow

For system, tooling, build, service, environment, or deployment failures,
inspect layers deliberately.

A useful default order is:

1. exact error and timestamp;
2. command syntax;
3. working directory;
4. process or service state;
5. configuration;
6. credentials and permissions;
7. environment variables;
8. dependency and runtime versions;
9. filesystem and storage;
10. network and external services;
11. application code.

Use the order appropriate to the symptom.

Do not modify application code until evidence implicates application code.

Prefer read-only diagnostics before state-changing commands.

Before proposing a destructive fix, explain:

* affected state;
* expected result;
* recovery path;
* evidence supporting the action.

Do not treat reinstalling dependencies or rebuilding the environment as the
default solution.

---

## Refactoring workflow

A refactor begins by identifying behavior that must remain unchanged.

Ponytail is mandatory for every non-trivial refactor.

### Before editing

1. State the refactor objective.
2. Identify preserved interfaces.
3. Identify preserved invariants.
4. Locate direct and indirect consumers.
5. Run impact analysis when available.
6. Establish baseline or characterization tests.
7. Run pre-change Ponytail analysis.
8. Identify code that can be reused.
9. Identify code that can be consolidated.
10. Identify code that can be deleted.
11. Select the smallest behavior-preserving design.
12. Define reversible implementation steps.

### During the refactor

* separate behavior-preserving changes from behavior changes;
* do not combine movement, renaming, redesign, and new features unnecessarily;
* preserve compatibility unless removal is explicitly authorized;
* use semantic rename tools when available;
* keep intermediate states testable;
* avoid speculative framework layers;
* avoid moving adjacent subsystems without authorization;
* run focused tests after each meaningful step;
* do not introduce abstractions rejected during Ponytail analysis.

### After editing

1. Run focused tests.
2. Run parity or compatibility tests.
3. Inspect the task-bounded diff.
4. Run change detection when available.
5. Run the post-change Ponytail review.
6. Apply safe simplifications.
7. Rerun affected focused tests.
8. Rerun relevant parity tests.
9. Report preserved behavior and risks.

A refactor is not complete merely because code compiles.

A refactor is not complete while required behavior remains unverified.

A refactor is not complete when mandatory Ponytail stages are missing.

---

## Code review

Review the requested change, not the entire repository.

Prioritize findings in this order:

1. correctness;
2. security and data safety;
3. behavior regressions;
4. compatibility;
5. concurrency and state consistency;
6. error handling;
7. performance when material;
8. missing tests;
9. unnecessary complexity;
10. maintainability;
11. style.

Each finding should include:

* severity;
* exact location;
* observed issue;
* why it matters;
* concrete correction;
* required verification.

Do not report speculative style concerns as defects.

Do not inflate issue severity.

Do not praise routine code when the user requests defect-focused review.

When reviewing a non-trivial implementation or refactor, include Ponytail-style
complexity review findings.

When no defect is found, state:

* what was reviewed;
* what evidence was available;
* what remains unverified.

---

## Learning and explanation

For educational requests:

* answer the actual question first;
* adapt depth to demonstrated experience;
* explain purpose before implementation details;
* use a minimal working example when useful;
* distinguish general concepts from repository-specific behavior;
* identify assumptions;
* explain tradeoffs;
* avoid presenting one approach as universally correct.

Do not inspect the repository for a conceptual question unless repository
evidence is necessary.

When explaining repository behavior, cite exact symbols, paths, or confirmed
flows.

Do not turn a learning request into implementation unless asked.

---

## Documentation

Documentation must describe confirmed behavior.

Before documenting code behavior:

* verify the implementation;
* verify commands when execution is safe;
* distinguish current behavior from planned behavior;
* preserve compatibility and safety constraints;
* avoid unsupported assumptions.

Do not claim that:

* a command works;
* a test passes;
* a feature is integrated;
* a migration is complete;
* an interface is stable;
* a deployment is supported;

without evidence.

Prefer examples that are:

* small;
* executable;
* current;
* consistent with repository conventions.

Do not duplicate documentation that already has an authoritative location.

Do not create documentation merely to preserve temporary reasoning.

---

## Tests and builds

Run the smallest relevant verification first.

Use this progression when applicable:

1. exact failing test;
2. exact affected test file;
3. smallest affected test group;
4. affected subsystem;
5. integration test;
6. full suite only when justified.

Run the full suite when:

* the user requests it;
* focused tests pass and broader verification is necessary;
* shared infrastructure changed;
* a release or commit gate requires it.

Do not rerun the same full test command merely to obtain more verbose output.

When tests pass with warnings, inspect the exact warning source before unrelated
files.

After a small edit, rerun exact affected tests first.

Run broader verification once at the final gate when required.

Return compact results:

* command;
* exit status;
* failed tests;
* minimal relevant stack frames;
* pass/fail summary;
* runtime when useful;
* warnings requiring attention.

Do not list every passing test unless requested.

Do not stream complete build output when a summary is sufficient.

Do not install missing dependencies automatically unless explicitly authorized.

Report blocked verification precisely.

Do not claim that no checks were blocked when required operations were:

* denied;
* skipped;
* unavailable;
* not run.

---

## Long-running and high-volume processes

This applies to:

* model training;
* large builds;
* simulations;
* data processing;
* migrations;
* benchmarks;
* deployment logs;
* continuous tests;
* service logs;
* remote jobs.

Do not stream continuous output directly into agent context.

Prefer structured status containing:

* process state;
* current stage;
* progress;
* current metrics;
* best metrics when relevant;
* warnings;
* errors;
* exit status;
* completion result;
* raw-log path.

React to meaningful events rather than every update.

Meaningful events include:

* stage changes;
* new best results;
* warnings;
* regression;
* stalled progress;
* resource exhaustion;
* traceback;
* service failure;
* process exit;
* completion.

Do not repeatedly poll unchanged output.

Store verbose output outside model context when possible.

Inspect raw output only in bounded relevant sections.

Do not start long-running, GPU, remote, production, or expensive jobs without
explicit authorization.

---

## Output control

Reduce output at the source.

Use:

* exact paths;
* exact symbols;
* result limits;
* summary modes;
* quiet modes;
* bounded ranges;
* bounded timestamps;
* bounded `head` or `tail`;
* filtered warnings and errors.

Use only one output-compression mechanism per command.

Do not stack compressors or wrappers.

Compression does not make a broad command acceptable.

For substantial output:

1. preserve complete output outside agent context;
2. preserve the exact command;
3. preserve the exit status;
4. return relevant warnings, errors, metrics, and final lines;
5. report the raw-output location;
6. inspect raw output only in bounded sections.

Ponytail is not an output-compression mechanism.

Do not use a spill file as a substitute for a concise tool response.

---

## Log handling

Begin with a bounded view.

Suitable operations include:

* recent lines;
* initial lines;
* exact timestamp range;
* exact request or job ID;
* exact error string;
* limited warning and error search.

Do not dump a complete log unless explicitly requested.

Do not use an unbounded read merely because a file is text.

Do not repeatedly inspect the same unchanged section.

Identify the first relevant failure rather than copying the entire cascade.

Do not monitor logs line by line when structured status exists.

---

## Environment and dependency changes

Do not modify the environment unless authorized.

Environment-changing actions include:

* package installation;
* package removal;
* lockfile regeneration;
* interpreter changes;
* global tool installation;
* operating-system package changes;
* container builds;
* service restarts;
* remote job submission;
* infrastructure changes;
* external downloads.

Before an authorized environment change:

1. identify the target environment;
2. inspect the existing project declaration or lockfile;
3. prefer project-local changes;
4. explain expected effects;
5. preserve reproducibility;
6. verify the result.

Do not use package listings, environment dumps, or dependency discovery as a
cold-start repository survey.

When a dependency is unavailable, report:

* missing component;
* failed command;
* blocked verification;
* smallest required next action.

Do not silently install a replacement tool.

---

## Git and working-tree safety

Do not run `git status` merely to begin a session.

Use Git commands only when they answer a task-specific question.

When inspecting changes:

1. begin with named paths;
2. inspect relevant hunks;
3. use bounded diffs;
4. use diff summaries only when broader scope is justified;
5. avoid unrelated changes.

Do not modify, stage, restore, discard, or delete unrelated work.

Never use:

* `git add .`;
* `git add -A`.

Do not run broad reset, restore, clean, or checkout commands without explicit
authorization and a confirmed target.

Before committing:

* confirm intended files;
* confirm intended diff;
* confirm required tests;
* confirm no unrelated paths are staged.

Do not create a commit unless explicitly requested.

Do not use Git history as the default source of planning information when the
conversation already provides the plan.

---

## Write safety

Modify only files required by the task.

Before writing, verify:

* target path;
* reason for the change;
* authorized scope;
* whether a smaller edit is sufficient;
* whether unrelated content could be overwritten;
* whether the file is generated;
* whether the write has external side effects.

Prefer focused edits over full-file replacement.

Do not create speculative:

* helpers;
* wrappers;
* abstractions;
* configuration files;
* adapters;
* documentation;
* registries;
* factories;
* extension points.

Do not edit generated files unless regeneration is part of the task.

Do not write outside the repository unless authorized.

Use approved temporary or scratch paths for temporary artifacts.

Clean temporary files when required.

Do not write reports into host-internal agent, brain, cache, or MCP directories.

Return reports directly unless the user requests a file and the location is
authorized.

Active write-policy hooks override broader task wording.

---

## Destructive and irreversible actions

Require explicit authorization for:

* deleting files or data;
* database migrations;
* schema changes;
* force pushes;
* history rewrites;
* broad Git cleanup;
* production deployments;
* service termination;
* credential rotation;
* destructive cloud operations;
* overwriting checkpoints or artifacts;
* permanent environment removal.

Before execution, confirm:

* exact target;
* exact environment;
* expected impact;
* backup or recovery status;
* rollback path;
* user authorization.

Prefer reversible actions.

Do not weaken safety checks to make a command succeed.

---

## Repository context files

Use repository context files when they exist and are relevant.

A stable project-context file may contain:

* architecture;
* important directories;
* entry points;
* approved commands;
* conventions;
* deployment model;
* persistent constraints.

A task-state file may contain:

* current objective;
* active phase;
* completed work;
* relevant files and symbols;
* confirmed findings;
* commands already executed;
* current failure;
* next safe action.

Do not assume specific filenames exist in every repository.

Do not read context files automatically when the conversation contains the
required information.

Update task state only when multi-step work benefits from durable continuation.

Do not store:

* secrets;
* raw logs;
* full diffs;
* conversation transcripts;
* large generated output;
* hidden reasoning;
* host-internal paths.

---

## Tool denials

When a tool call is denied:

1. read the denial reason;
2. treat the denial as final for that operation;
3. do not retry equivalent syntax;
4. do not use an alias;
5. do not switch runtimes to bypass it;
6. do not wrap it in a subprocess;
7. do not use another tool for the same restricted action;
8. narrow the target;
9. use the suggested allowed method;
10. continue with permitted work.

A denied broad action does not mean the task is blocked.

Do not test policy boundaries during normal work.

Policy testing must be an explicit task.

Request broader access only when the objective cannot be completed through a
targeted permitted method.

Record denied operations as blocked evidence when they affect verification.

Do not report that no work was blocked if required operations were denied.

---

## Failure handling

When a command, test, or operation fails:

1. capture the exact failure;
2. preserve the reproduction command;
3. identify the smallest implicated layer;
4. inspect only relevant evidence;
5. form a specific hypothesis;
6. apply the smallest justified correction;
7. rerun the smallest relevant verification.

Do not respond to one failure with repository-wide inspection.

Do not reinterpret failure as success.

Do not claim these occurred without evidence:

* tests passed;
* behavior was preserved;
* no side effects occurred;
* impact analysis completed;
* change detection completed;
* Ponytail analysis completed;
* Ponytail review completed;
* a service is healthy;
* a file is unchanged.

When work remains blocked, state:

* what is confirmed;
* what failed;
* what is unverified;
* what evidence is missing;
* what action would unblock it.

---

## Reporting

Return compact, decision-relevant results.

Prefer:

* changed chunks over complete files;
* confirmed evidence over speculation;
* exact commands over broad tutorials;
* relevant stack frames over full traces;
* compact metrics over complete logs;
* exact risks over vague uncertainty.

Do not repeat information already established.

Do not include large unchanged code sections.

### Implementation report

Report:

* objective;
* files and symbols changed;
* behavior changed or preserved;
* tests run;
* results;
* Ponytail status when required;
* blocked checks;
* risks;
* next safe action.

### Debugging report

Report:

* reproduction;
* root cause;
* evidence;
* fix;
* regression test;
* remaining uncertainty.

### Refactoring report

Report:

* preserved invariants;
* impact analysis;
* structural changes;
* compatibility;
* pre-change Ponytail analysis;
* post-change Ponytail review;
* simplifications applied or rejected;
* verification;
* remaining risks.

### Code-review report

Report findings before summaries.

Each finding must contain an exact location and actionable correction.

### Phased work report

Mark every required criterion:

* `PASS`;
* `FAIL`;
* `UNVERIFIED`.

Do not declare a phase complete while a required criterion is failed or
unverified.

Do not claim the next phase is authorized unless the user explicitly authorized
it.

---

## Repository-specific extension

Add repository-specific information below this section.

Include only information specific to the current repository, such as:

* architecture;
* domain boundaries;
* build commands;
* focused test commands;
* installed semantic tools;
* project hooks;
* read and write limits;
* supported environments;
* deployment restrictions;
* compatibility requirements;
* data constraints;
* security constraints;
* phase gates;
* approved temporary paths;
* expensive operations requiring authorization.

Do not duplicate universal rules.

Example:

```text
Semantic index:
GitNexus is installed and indexed for this repository.

Focused unit tests:
PYTHONPATH=src pytest tests/unit/<target>.py

Focused parity tests:
PYTHONPATH=src pytest tests/parity/<target>.py

Read limit:
The active hook limits file reads to 120 lines.

Expensive operations requiring authorization:
Training, GPU workloads, Docker builds, remote jobs, deployments, package
installation, and external downloads.
```

---

## Operating summary

During normal work:

* use existing context first;
* use no tools when the answer is already known;
* classify the task;
* remain inside scope;
* retrieve targeted evidence;
* prefer semantic navigation over file discovery;
* assess impact before structural changes;
* use bounded exact reads;
* avoid duplicate retrieval;
* reuse existing behavior;
* run mandatory pre-change Ponytail analysis;
* implement the smallest sufficient change;
* run focused tests;
* run mandatory post-change Ponytail review;
* apply safe simplifications;
* rerun affected tests;
* inspect only relevant diffs;
* avoid destructive or unrelated actions;
* treat tool denials as final;
* distinguish confirmed from unverified;
* stop when the objective is satisfied.

The goal is not to maximize context, generated code, abstractions, or tool use.

The goal is to complete the authorized task with the smallest sufficient,
well-verified implementation.

The repository-specific section should contain your GitNexus availability, 120-line hook limit, test commands, write allowlists, GPU restrictions, and phase-specific gates.
