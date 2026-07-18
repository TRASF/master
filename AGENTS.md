# Repository Agent Operating Rules

## 1. Purpose

These rules define how an agent operates in this repository.

They apply to:

* implementation;
* debugging;
* troubleshooting;
* refactoring;
* code review;
* testing;
* packaging;
* builds;
* documentation;
* configuration;
* infrastructure;
* deployment;
* maintenance;
* research;
* incident investigation.

The objective is to complete the explicitly authorized task with the smallest
sufficient, well-verified change while preserving the repository’s actual
current behavior.

The objective is not to maximize:

* tool use;
* repository exploration;
* generated code;
* abstractions;
* file changes;
* context collection;
* output volume.

A task prompt defines the requested objective. It does not automatically
override repository state, active policy hooks, completed later-phase work, or
existing compatibility requirements.

---

# 2. Authority and instruction precedence

Apply instructions in this order:

1. system and safety requirements;
2. active execution hooks and tool restrictions;
3. more specific directory-level instructions;
4. repository instructions;
5. externally controlled phase and write authorization;
6. confirmed current repository state;
7. the exact current user task;
8. these repository defaults.

Active hooks are authoritative.

A task prompt cannot override an active hook.

A task prompt cannot silently authorize:

* a later phase;
* removal of existing later-phase behavior;
* broad repository access;
* environment modification;
* destructive actions;
* writes outside an authorized path set;
* commits or remote pushes.

Do not bypass restrictions through:

* aliases;
* alternate commands;
* another shell;
* another programming language;
* subprocess wrappers;
* generated scripts;
* indirect filesystem access;
* another tool with equivalent behavior.

When instructions conflict, follow the more restrictive valid instruction.

---

# 3. Hard operating principles

Use the active conversation and confirmed project state before calling tools.

Do not rediscover the repository because a new session started.

Do not inspect files merely to become familiar with the project.

Do not gather context without a concrete question that the context will answer.

Stop retrieving information when sufficient evidence exists.

Prefer:

* targeted semantic queries;
* exact symbols;
* exact paths;
* bounded reads;
* focused tests;
* narrow diffs;
* direct implementations;
* reuse of existing behavior;
* preservation of current functionality.

Do not perform unrelated cleanup.

Do not enter a later phase automatically.

Do not force the repository backward to satisfy an obsolete task prompt.

Distinguish clearly between:

* `CONFIRMED`;
* `INFERRED`;
* `ASSUMED`;
* `UNVERIFIED`;
* `BLOCKED`;
* `FAILED`;
* `NOT REQUIRED`.

Never present inference as confirmed behavior.

Never claim that:

* a test passed;
* behavior was preserved;
* Ponytail was completed;
* impact analysis was performed;
* no files were changed;
* no side effects occurred;
* a phase is complete;

without visible supporting evidence.

---

# 4. Zero-tool completion

Before the first tool call, determine whether the active conversation already
contains enough information to complete the request.

Use no repository tools when the task can be completed from current context.

Examples include:

* drafting a task prompt;
* writing documentation;
* explaining a known failure;
* identifying a phase conflict already shown in logs;
* asking whether to begin another phase;
* producing a plan from supplied requirements;
* explaining previously confirmed behavior;
* rewriting repository instructions.

Silently ask:

```text
Can this request be completed accurately from the active conversation and
confirmed project state?
```

When the answer is yes, do not call repository tools.

---

# 5. Explicit objective requirement

## 5.1 The current objective must be explicit

Do not infer the authorized objective or phase from:

* previous conversation intent;
* an old task template;
* a branch name;
* a commit message;
* repository contents;
* an earlier agent’s conclusion;
* an implementation plan from another session.

Use the exact objective stated in the current task.

When the current task does not explicitly state an objective or phase, report:

```text
Requested objective: UNSPECIFIED
Requested phase: UNSPECIFIED
State decision: BLOCK
```

Do not convert prior-session intent into current authorization.

## 5.2 Previous context is not authorization

Previous context may help identify:

* known repository behavior;
* possible conflicts;
* files that may have been touched;
* prior failed approaches.

Previous context cannot independently authorize:

* code changes;
* a migration phase;
* package installation;
* Docker builds;
* training;
* commits;
* destructive actions.

---

# 6. Evidence provenance

For every state, verification, or completion claim, identify its provenance.

Use these categories:

* current-session evidence;
* prior-session evidence;
* repository evidence;
* user-provided evidence;
* inference;
* unverified information.

Do not state that a test currently passes unless it was run successfully in the
current session.

Prior results may be reported only in this form:

```text
Prior-session result: reported PASS
Current-session status: NOT RERUN
```

Do not use prior-session test results as current completion evidence.

Do not state that a file is unchanged unless a current-session targeted check
supports that statement.

Do not state that an operation completed when the corresponding write or command
was denied.

---

# 7. Mandatory authorization envelope

## 7.1 Required authorization fields

Before any production write, establish:

```text
Authorized objective:
Authorized phase:
Authorized write paths:
Authorized test scope:
Authorized environment changes:
Authorized expensive operations:
Explicitly excluded work:
```

The authorized phase and write paths must come from:

* active hook state;
* externally supplied task metadata;
* an explicit current user instruction consistent with the hook.

Do not infer authorization from repository structure or history.

## 7.2 Exact write scope

Writes are allowed only to exact paths or explicitly authorized path patterns.

Before the first write, record:

```text
Planned write targets:
- <exact path>
- <exact path>
```

Do not add another target without:

1. explaining why it is required;
2. confirming that it belongs to the authorized objective;
3. confirming that it belongs to the authorized phase;
4. confirming that the hook permits it;
5. updating the planned target set.

Do not interpret broad terms such as `tests`, `src`, or `docs` as permission to
modify the complete subtree unless that subtree is explicitly authorized.

## 7.3 No permission probing

Never create, edit, append to, rename, or delete a repository file merely to
discover whether the action is permitted.

Forbidden reasoning includes:

```text
I will edit this file to test the write policy.
```

```text
I will create a test in this directory to see whether writes are allowed.
```

```text
I will append a temporary note and remove it later.
```

A write must directly serve the authorized user objective.

When a write is denied:

* do not try another path to test the boundary;
* do not try a nearby directory;
* do not try another write tool;
* do not modify an apparently allowed file as a probe;
* do not attempt a temporary write.

Report the denial and continue only with clearly permitted work.

## 7.4 Existing-file protection

Before using a create operation, verify that the exact target does not already
exist.

When the target exists:

* do not use create;
* use a focused edit only when authorized;
* preserve unrelated content;
* inspect the relevant section before editing.

Do not replace an existing file because a tool labels the intended operation as
creation.

---

# 8. Mandatory repository-state reconciliation

## 8.1 State gate

Before editing production code, reconcile the current task with the actual
repository state.

This is mandatory for:

* phased migrations;
* packaging work;
* configuration changes;
* refactoring;
* entry-point changes;
* public API changes;
* dependency changes;
* tasks based on an earlier implementation plan.

Record:

```text
Requested objective:
Requested phase:
Hook-authorized phase:
Confirmed current implementation:
Existing later-phase functionality:
Overlapping files or symbols:
Behavior that must remain:
Evidence provenance:
State decision: PROCEED | REFRAME | BLOCK
```

No production write is allowed until the state gate is complete.

## 8.2 State decisions

Use `PROCEED` only when:

* the objective is explicit;
* the requested phase matches the hook-authorized phase;
* the work does not remove later-phase behavior;
* overlapping files can be modified safely;
* the baseline state is understood;
* required write targets are authorized;
* required verification is available.

Use `REFRAME` when:

* the objective is useful but based on stale assumptions;
* an earlier phase should be verified rather than recreated;
* the same goal can be achieved without removing current behavior;
* the task should become a verification-only audit;
* existing implementation already satisfies part of the requested work.

Use `BLOCK` when:

* the objective or phase is unspecified;
* the requested phase exceeds the hook-authorized phase;
* a required production write is denied;
* the prompt conflicts with existing later-phase behavior;
* overlapping user changes cannot be isolated;
* the correct baseline cannot be established;
* acceptance criteria require a regression;
* required environment changes are unauthorized;
* the task relies on unsupported prior-session claims.

## 8.3 Phase mismatch

Stop before writing when:

* Phase 1 is requested but Phase 2 or later functionality exists in a target;
* the prompt says to create a file that already contains newer behavior;
* the prompt says to remove commands current users or tests rely on;
* later-phase tests already consume the target interface;
* an active hook reports an earlier authorized phase;
* the prompt assumes functionality is absent when it is present;
* fulfilling the task would downgrade the repository.

Do not relabel later-phase code as earlier-phase work to justify continuing.

Do not simplify an interface by removing required current behavior.

## 8.4 Earlier-phase verification

An earlier phase may be audited after later phases exist.

Allowed:

```text
Verify that package import, help, version, wheel installation, and lightweight
imports remain valid.
```

Not allowed:

```text
Remove the configuration command because the original package-foundation phase
did not expose it.
```

When later functionality exists, reframe earlier-phase implementation work as
verification work unless the user explicitly authorizes a compatible change.

---

# 9. Instruction-file handling

Repository instructions may include:

* `AGENTS.md`;
* `AGENT.md`;
* `CLAUDE.md`;
* `.agent/PROJECT.md`;
* `.agent/STATE.md`;
* equivalent directory-specific instruction files.

When repository instructions have already been loaded by the host, do not
search or read them again merely to locate general rules.

Read a bounded section only when:

* a specific required rule is absent from active context;
* a concrete instruction conflict must be resolved;
* the task directly modifies the instruction file;
* a nested directory may contain more specific instructions;
* the host did not load repository instructions.

Do not reconstruct an instruction file through consecutive reads.

Do not repeatedly search instruction files for phase names after the active hook
has already declared the authorized phase.

---

# 10. Tool-denial handling

## 10.1 A denial is final

When a tool call is denied:

1. read the denial message;
2. treat it as authoritative;
3. do not retry equivalent syntax;
4. do not use an alias;
5. do not switch runtimes;
6. do not wrap the operation in a script;
7. do not use another tool for the same restricted operation;
8. switch to the suggested targeted method;
9. record blocked evidence when it affects completion.

## 10.2 Denial budget

Maintain a task-local denial counter.

After two denials caused by broad or poorly targeted operations:

* stop broad retrieval;
* use only exact paths, exact symbols, focused tests, or approved semantic
  queries.

After three such denials:

* stop implementation planning;
* reassess the task against the authorization envelope;
* perform no writes until a clearly permitted route is established.

After four denials:

* disable further writes for the task;
* return `BLOCK` or `REFRAME` unless all remaining work is read-only.

Do not continue accumulating denials.

Do not test policy boundaries.

## 10.3 Do not inspect policy internals

Do not read or search:

* `.agent/hooks/**`;
* AGY hook source;
* global hook configuration;
* hidden host-policy files;
* plugin implementation directories;
* `.git/**`;

merely because an operation was denied.

The denial message is sufficient operational guidance.

Inspect policy implementation only when the user explicitly requests:

* hook development;
* hook debugging;
* policy maintenance;
* authorized policy testing.

Do not reconstruct a guard file through multiple bounded reads during unrelated
work.

---

# 11. Task classification

Silently classify the current task before using tools.

## 11.1 Explanation or learning

Use supplied context and general knowledge first.

Inspect repository code only when repository-specific evidence is necessary.

## 11.2 Targeted development

Work only on the named:

* feature;
* file;
* symbol;
* command;
* test;
* configuration;
* behavior.

This is the default implementation mode.

## 11.3 Diagnostic debugging

Begin with:

* the exact error;
* exact traceback;
* failed test;
* reproduction command;
* relevant log event.

Expand only to components implicated by evidence.

## 11.4 Troubleshooting

Determine whether the issue belongs to:

* command invocation;
* working directory;
* interpreter;
* import path;
* installation state;
* dependencies;
* configuration;
* permissions;
* storage;
* networking;
* service state;
* build tooling;
* application code.

Do not assume every failure requires a source-code change.

## 11.5 Structural work

Use semantic navigation and impact analysis before changing:

* shared interfaces;
* module boundaries;
* entry points;
* configuration ownership;
* public behavior;
* cross-file flows.

Ponytail is mandatory for non-trivial structural work.

## 11.6 Exhaustive work

Repository-wide discovery is allowed only when explicitly requested for:

* an audit;
* inventory;
* architecture analysis;
* migration assessment;
* security review;
* repository-wide refactor.

A long prompt does not automatically authorize exhaustive work.

---

# 12. Before every tool call

Silently verify:

1. Is the call necessary?
2. Is the answer already in active context?
3. Is the current objective explicit?
4. Did the state gate already require `BLOCK` or `REFRAME`?
5. Is the target exact?
6. Is the operation inside the authorized phase?
7. Is the path inside the authorized scope?
8. Is this the narrowest suitable tool?
9. Would a semantic query be better than a file read?
10. Is the output bounded?
11. Has the same information already been retrieved?
12. Was an equivalent call already denied?
13. Could the command produce excessive output?
14. Does it modify the environment?
15. Is it expensive or externally visible?
16. Will it produce decision-relevant evidence?
17. Can the resulting claim be attributed to current-session evidence?

Do not narrate this checklist.

---

# 13. Discovery and navigation

## 13.1 Retrieval order

Use this order:

1. current conversation;
2. confirmed current-task state;
3. semantic repository graph;
4. language-server symbol navigation;
5. syntax-aware search;
6. bounded exact-text search;
7. bounded file read;
8. filesystem discovery only when no narrower method works.

Do not move to a broader method merely to gather additional context.

## 13.2 GitNexus usage

Use GitNexus for concrete code questions.

Good queries:

```text
Identify all current CLI commands and their direct module dependencies.
```

```text
Trace callers of resolve_config and the tests that verify its behavior.
```

```text
Trace how the Docker build installs dependencies and the local package.
```

```text
Identify entry points that depend on repository-root imports.
```

```text
Determine the blast radius of changing the package CLI parser.
```

Bad queries:

```text
layout
test
requirements
Phase 1
Phase 2
loader.py
wingbeat_ml
current migration
```

Do not use GitNexus as a filesystem listing system.

For understanding:

```text
query
→ context for a returned symbol
→ one bounded implementation read
```

For structural changes:

```text
query
→ context
→ impact
→ one bounded implementation read
```

Use `rename` for code-symbol renames when available.

Use `detect_changes` only after a task-bounded implementation.

Consume structured MCP results directly.

Do not open:

* MCP spill files;
* brain paths;
* cache files;
* temporary MCP response files;
* internal file URIs.

When a result is too broad, issue a narrower query.

## 13.3 Restricted GitNexus operations

Do not use these during ordinary targeted development unless explicitly
authorized:

* repository-wide Cypher enumeration;
* listing every file node;
* generic test-file enumeration;
* `tool_map` exploration;
* generic searches for phase names;
* generic searches for reports;
* graph-schema exploration.

## 13.4 Exact symbols and structural patterns

Use language-server tools for:

* definitions;
* references;
* implementations;
* callers;
* callees;
* diagnostics;
* type information;
* safe symbol rename.

Use AST or syntax-aware search for:

* API call patterns;
* decorators;
* inheritance;
* exception structures;
* import patterns;
* deprecated API usage.

Do not read a complete file to locate one known symbol.

---

# 14. Repository and Git inspection

## 14.1 No cold-start enumeration

Do not begin normal work with:

* `ls` for repository discovery;
* `tree`;
* repository-wide `find`;
* repository-wide `fd`;
* `rg --files`;
* `git ls-files`;
* recursive globbing;
* repository-wide line counts;
* listing every test;
* listing every package;
* `git clean -n`;
* broad `git status`;
* broad `git diff`.

Do not reproduce enumeration through Python, Node.js, or generated scripts.

## 14.2 Targeted Git inspection

Use exact paths:

```bash
git status --short -- <path> <path>
git diff -- <path> <path>
git diff --stat -- <path> <path>
git diff --name-only -- <path> <path>
git log -n 1 --oneline -- <path>
```

When a path-bounded `git status` is denied, do not retry broad status.

Use an allowed targeted diff or report that status evidence is blocked.

Do not use Git history as the only source of repository state.

A commit title does not prove phase completion.

A branch name does not authorize work.

## 14.3 Existing-path checks

When determining whether one known file exists, use an exact-path check.

Do not enumerate a directory to find it.

When determining whether one known path is tracked, use a path-specific Git
query.

Do not list the complete tracked-file set.

---

# 15. File reading

Prefer symbol-level retrieval for code.

Use bounded reads for:

* exact documentation sections;
* exact configuration fragments;
* test cases;
* logs;
* generated output;
* small known files.

Follow the active read limit.

For this repository, do not request more than 120 lines per read unless the hook
explicitly permits it.

Do not reconstruct a large file through consecutive reads.

After two reads from one file, reconsider whether:

* a symbol query;
* exact text search;
* targeted diff;
* AST query;

would answer the question more efficiently.

Do not read unchanged content twice.

A repeated read must:

* verify an edit;
* retrieve unseen required content;
* answer a named unresolved question.

Do not read host-internal:

* brain files;
* spill files;
* cache files;
* hidden reasoning files;
* temporary MCP reports.

---

# 16. Test and baseline resolution

## 16.1 Resolve the supported command first

Before treating a test failure as a product failure, confirm:

* interpreter;
* working directory;
* package installation state;
* project layout;
* supported test command;
* required `PYTHONPATH`;
* relevant test target.

Do not run plain `pytest` reflexively.

For this repository, the default source-layout test form is:

```bash
PYTHONPATH=src pytest <target>
```

Use another command only when repository instructions or the current task
explicitly require it.

## 16.2 Import failures

A `ModuleNotFoundError` may indicate:

* an uninstalled source-layout package;
* missing `PYTHONPATH`;
* wrong interpreter;
* wrong working directory.

Verify those before editing source code.

Do not change package code to compensate for an incorrect test invocation.

## 16.3 No environment fishing

Do not try a sequence of guessed environments such as:

* multiple Conda environments;
* Poetry;
* `.venv`;
* `venv`;
* arbitrary interpreters;

unless concrete evidence indicates one is required.

Do not enumerate every package or environment during normal test resolution.

Use the documented or already confirmed project environment.

## 16.4 Baseline record

Before production edits, record:

```text
Baseline command:
Interpreter:
Working directory:
Environment assumptions:
Current-session result:
Warnings:
Baseline status: PASS | FAIL | BLOCKED
```

When the initial command is wrong, correct it once.

When the correct baseline fails for a product reason, stop unless the current
task explicitly authorizes fixing that failure.

## 16.5 Test progression

Run:

1. exact failing test;
2. exact affected test file;
3. smallest affected test group;
4. parity or compatibility tests;
5. integration tests;
6. full suite only when justified.

Do not rerun the same broad suite merely for additional verbosity.

Do not regenerate fixtures, snapshots, or baselines to hide a failure.

---

# 17. Dependency and packaging ownership

Before editing dependency metadata, identify the authoritative source.

For this repository during the current packaging migration:

```text
Runtime dependency authority: requirements.txt
Package metadata: pyproject.toml
```

Do not duplicate all runtime dependencies into `pyproject.toml` unless the task
explicitly migrates dependency ownership.

A valid installation pattern is:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps .
```

Do not:

* guess dependency constraints;
* upgrade dependencies;
* repin TensorFlow;
* alter CUDA or cuDNN versions;
* add a package-management framework;
* redesign dependency ownership;
* install missing tools automatically.

When package metadata and dependency ownership conflict, report the conflict and
use `REFRAME` or `BLOCK`.

---

# 18. Mandatory Ponytail workflow

## 18.1 Purpose

Ponytail is the required simplicity and anti-overengineering workflow for
non-trivial implementation and refactoring.

Ponytail is not:

* output compression;
* a replacement for testing;
* a replacement for impact analysis;
* permission to remove required functionality;
* a statement that code appears simple.

Reading the Ponytail skill file alone does not count.

Mentioning Ponytail in the final report does not count.

## 18.2 Mandatory triggers

Use Ponytail when a task:

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
* changes module boundaries;
* changes public behavior;
* moves responsibility between components;
* adds configuration structure;
* removes or replaces code;
* performs a cross-file refactor;
* produces a non-trivial source diff.

Ponytail is normally optional for:

* spelling;
* comments;
* formatting;
* documentation-only edits;
* isolated literal changes;
* generated files.

When uncertain, use Ponytail.

## 18.3 Stage 1: pre-change analysis

Before writing code, record:

```text
Pre-change Ponytail analysis

Existing behavior to reuse:
Standard-library alternatives:
Installed-library alternatives:
New abstractions considered:
Confirmed consumers:
Simpler designs considered:
Code that may be deleted or consolidated:
Speculative features rejected:
Smallest sufficient design:
Compatibility behavior preserved:
Later-phase behavior preserved:
```

Answer:

1. Does equivalent behavior already exist?
2. Can an existing implementation be extended?
3. Can the standard library provide it?
4. Can an installed dependency provide it?
5. Can the change remain local?
6. Is a function sufficient instead of a class?
7. Is a direct call sufficient instead of a wrapper?
8. Are current consumers confirmed?
9. Is the abstraction required now?
10. Is it only for a hypothetical future phase?
11. Can code be removed instead of added?
12. Would the simplification remove newer behavior?
13. Would it conflict with passing tests?
14. Would it recreate an earlier phase over later work?

When questions 12, 13, or 14 are true, use `REFRAME` or `BLOCK`.

Do not begin implementation until the smallest compatible design is selected.

## 18.4 Stage 1 evidence

The execution record must contain:

* an actual Ponytail skill invocation; or
* a complete explicitly labeled pre-change analysis.

Reading a short Ponytail skill file without applying the procedure is
insufficient.

## 18.5 Stage 2: post-change review

After focused tests pass, inspect the actual task-bounded diff.

Review for:

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
* unsupported compatibility layers;
* custom code already provided by the standard library;
* custom code already provided by installed dependencies.

Report each candidate:

```text
Candidate:
Location:
Why it may be unnecessary:
Safe simplification:
Behavior that must remain:
Decision: APPLY | REJECT | DEFER
Reason:
```

A statement such as `Ponytail review completed` is invalid without concrete
candidate decisions.

## 18.6 Applying simplifications

Apply a recommendation only when it:

* remains in authorized scope;
* preserves current behavior;
* preserves later-phase behavior;
* preserves compatibility;
* preserves validation and error handling;
* preserves deterministic behavior;
* does not weaken security;
* does not remove required observability;
* is covered by focused tests.

After simplification:

1. rerun affected tests;
2. rerun relevant parity tests;
3. inspect the final bounded diff;
4. confirm no replacement abstraction was introduced.

## 18.7 Ponytail status

Report `PASS` only when:

* pre-change analysis is complete;
* current-state compatibility is confirmed;
* the smallest design is recorded;
* abstractions are justified;
* the actual diff is reviewed;
* candidate decisions are reported;
* accepted simplifications are applied;
* affected tests are rerun.

Otherwise:

```text
Ponytail status: UNVERIFIED
```

Successful tests do not compensate for missing Ponytail stages.

---

# 19. Implementation workflow

Use this sequence:

1. confirm the exact current objective;
2. establish the authorization envelope;
3. reconcile the requested phase with current state;
4. identify exact write targets;
5. identify behavior that must remain;
6. retrieve only required context;
7. run semantic impact analysis when necessary;
8. resolve the supported baseline command;
9. run the smallest relevant baseline;
10. perform pre-change Ponytail analysis when required;
11. add focused tests when required;
12. implement the smallest compatible change;
13. run focused tests;
14. inspect the task-bounded diff;
15. run post-change Ponytail review;
16. apply safe simplifications;
17. rerun affected tests;
18. run required parity or compatibility checks;
19. run semantic change detection when justified;
20. report only current-session verified results.

Do not:

* add unrelated cleanup;
* create speculative extension points;
* remove current functionality because an older phase did not require it;
* enter another phase automatically;
* write outside the authorized path set;
* treat prior-session results as current evidence.

---

# 20. Debugging workflow

Use:

1. capture the exact symptom;
2. preserve the reproduction command;
3. verify interpreter and working directory;
4. verify import and installation state;
5. reproduce the failure;
6. identify the first meaningful error;
7. identify the smallest implicated component;
8. form explicit hypotheses;
9. test the cheapest discriminating hypothesis;
10. inspect only implicated code;
11. apply the smallest root-cause fix;
12. add or update a regression test;
13. rerun the original reproduction;
14. run surrounding focused tests;
15. use Ponytail when structure changes.

Do not change code because it merely appears suspicious.

Do not apply several speculative fixes simultaneously.

Do not hide failures through:

* broad exception handling;
* warning suppression;
* ignored exit codes;
* arbitrary retries;
* disabled validation;
* unverified fallback behavior.

---

# 21. Refactoring workflow

Ponytail is mandatory for every non-trivial refactor.

Before editing:

1. establish authorization;
2. reconcile repository state;
3. identify preserved interfaces;
4. identify preserved invariants;
5. identify direct and indirect consumers;
6. run impact analysis;
7. establish characterization tests;
8. run pre-change Ponytail analysis;
9. select the smallest behavior-preserving design;
10. define reversible steps.

During refactoring:

* separate movement from behavior changes;
* do not combine renaming, redesign, and new features unnecessarily;
* preserve compatibility;
* preserve later-phase behavior;
* use semantic rename tools;
* keep intermediate states testable;
* avoid speculative frameworks;
* run focused tests after meaningful steps.

After refactoring:

1. run focused tests;
2. run parity or compatibility tests;
3. inspect the bounded diff;
4. run semantic change detection;
5. run post-change Ponytail review;
6. apply safe simplifications;
7. rerun affected tests;
8. report preserved behavior and remaining risks.

A refactor is not complete merely because the code imports or compiles.

---

# 22. Environment and expensive operations

Do not modify the environment without explicit current-task authorization.

Environment changes include:

* package installation;
* package removal;
* lockfile regeneration;
* interpreter changes;
* system package changes;
* container builds;
* service restarts;
* downloads;
* remote job submission.

Do not start without explicit authorization:

* model training;
* GPU workloads;
* Docker builds;
* remote jobs;
* deployments;
* large benchmarks;
* data migrations.

Before an authorized operation:

1. identify the target environment;
2. identify expected cost and side effects;
3. identify the recovery path;
4. preserve reproducibility;
5. verify the result.

---

# 23. Git and write safety

Modify only authorized files.

Before every write, verify:

* exact path;
* file existence;
* create versus edit operation;
* reason for change;
* authorized objective;
* authorized phase;
* authorized path set;
* risk of overwriting later work;
* whether the file is generated;
* whether a smaller change is sufficient.

Do not:

* stage unrelated files;
* restore unrelated files;
* reset unrelated work;
* clean the repository;
* overwrite existing files;
* write reports into host-internal directories.

Never use:

```bash
git add .
git add -A
```

Do not commit unless explicitly authorized in the current task.

Before committing:

* confirm exact intended paths;
* inspect the bounded diff;
* confirm current-session test evidence;
* confirm no unrelated staged changes;
* confirm the state gate passed;
* confirm Ponytail passed when required.

Do not create a completion commit when any required criterion is:

* `FAIL`;
* `BLOCKED`;
* `UNVERIFIED`.

---

# 24. Output and log control

Reduce output at the source.

Use:

* exact targets;
* quiet modes;
* summary modes;
* result limits;
* bounded ranges;
* focused warnings;
* focused error lines.

Do not stream continuous output into agent context.

For long-running commands, report:

* command;
* process state;
* progress;
* warnings;
* errors;
* exit status;
* final result;
* raw-log path when authorized.

Do not repeatedly poll unchanged output.

Do not state that you will pause and wait after a synchronous command has
already returned.

Do not paste complete test logs when a compact result is sufficient.

---

# 25. Reporting

## 25.1 State report

For phased work, report:

```text
Requested objective:
Requested phase:
Hook-authorized phase:
Confirmed repository state:
Existing later-phase behavior:
Evidence provenance:
State decision: PROCEED | REFRAME | BLOCK
Conflicts:
```

When the objective is missing:

```text
Requested objective: UNSPECIFIED
Requested phase: UNSPECIFIED
State decision: BLOCK
```

## 25.2 Implementation report

Report:

* objective;
* authorized write targets;
* files and symbols changed;
* behavior changed;
* behavior preserved;
* current-session tests and commands;
* current-session results;
* prior-session results, clearly labeled;
* Ponytail status;
* blocked checks;
* concrete remaining risks;
* next safe action.

## 25.3 Verification statuses

Use only:

* `PASS`;
* `FAIL`;
* `BLOCKED`;
* `UNVERIFIED`;
* `NOT REQUIRED`;
* `NOT RERUN`.

Do not report `PASS` when:

* a required write was denied;
* required checks were skipped;
* the baseline was not run in the current session;
* a wheel was not built when required;
* Docker was not tested when required;
* an impact review was not run when required;
* Ponytail post-review was absent;
* the repository state remained unclear.

## 25.4 Unsupported claims

Do not claim:

```text
Dependency ownership was modified.
```

when the write was denied.

Do not claim:

```text
CLI simplification was verified.
```

when no simplification occurred.

Do not claim:

```text
Phase complete.
```

when acceptance criteria remain unverified.

The final status must follow the evidence, not the intended outcome.

---

# 26. Stop conditions

Stop before writing when:

* the current objective is unspecified;
* the requested phase is unspecified where phase authorization is required;
* task phase conflicts with hook phase;
* repository state conflicts with the prompt;
* the change would remove later-phase functionality;
* an exact required write is denied;
* the target path is not authorized;
* the baseline command cannot be determined safely;
* the correct baseline fails;
* overlapping user changes cannot be isolated;
* dependency ownership is unclear;
* the task requires an unauthorized environment change;
* the denial budget is exhausted;
* the prompt is demonstrably obsolete;
* required current-session evidence cannot be obtained.

Stop after writing when:

* focused tests fail;
* parity tests fail;
* compatibility changes unexpectedly;
* the final diff contains unrelated changes;
* Ponytail identifies unresolved structural risk;
* required acceptance criteria remain unverified.

Do not weaken tests or regenerate baselines to force completion.

---

# 27. Repository-specific rules

## Semantic code intelligence

GitNexus is installed and indexed for this repository.

Use it for:

* execution flow;
* callers and callees;
* dependency relationships;
* entry-point impact;
* cross-file behavior;
* structural change analysis.

Do not use GitNexus for generic filesystem enumeration.

## Read limit

The active hook permits a maximum of 120 lines per file read.

Do not reconstruct large files through repeated reads.

## Test command

The default source-layout test invocation is:

```bash
PYTHONPATH=src pytest <target>
```

Use the smallest relevant target first.

A prior-session successful test is not current-session evidence.

## Dependency authority

During the current packaging migration:

```text
requirements.txt remains authoritative for runtime dependencies.
pyproject.toml provides build and package metadata.
```

Do not duplicate or migrate dependency ownership without explicit
authorization.

## Phase authority

The active phase is determined by external hook state.

The agent must not:

* infer the active phase;
* advance the active phase;
* edit a file to test phase permissions;
* treat prior-session intent as current authorization;
* treat a user prompt as overriding a phase hook.

## Existing later-phase behavior

Earlier phases may be verified, but they must not overwrite or remove
functionality already introduced by later phases.

## Expensive operations

These require explicit authorization:

* model training;
* GPU workloads;
* Docker builds;
* remote jobs;
* deployments;
* package installation;
* dependency changes;
* external downloads.

## Protected paths

Do not inspect these during ordinary development:

```text
.agent/hooks/**
.git/**
~/.gemini/**
**/brain/**
**/cache/**
**/spill/**
```

Access them only for an explicitly authorized policy-maintenance task.

---

# 28. Operating summary

During normal work:

* use active context first;
* use no tools when no tools are needed;
* require an explicit current objective;
* do not inherit authorization from a previous session;
* establish exact authorization;
* reconcile the task with actual repository state;
* preserve later-phase functionality;
* stop on phase mismatch;
* distinguish current evidence from prior evidence;
* never probe permissions through writes;
* use exact write targets;
* treat denials as final;
* stop after repeated denials;
* do not inspect hooks during unrelated work;
* do not reread loaded instruction files without a specific reason;
* prefer semantic navigation;
* use concrete GitNexus questions;
* avoid filesystem enumeration;
* resolve the correct test command first;
* preserve dependency ownership;
* perform pre-change Ponytail analysis;
* implement the smallest compatible change;
* run focused tests;
* inspect the bounded diff;
* perform post-change Ponytail review;
* rerun affected tests;
* report only supported claims;
* stop when the authorized objective is complete.

The repository’s confirmed current behavior is more authoritative than an
obsolete implementation prompt.

Simplicity means removing unnecessary complexity while preserving all required
current functionality.
