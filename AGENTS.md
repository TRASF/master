# Agent Operating Rules

## Default behavior

Use the current conversation, existing project context, and indexed repository
knowledge before retrieving more information.

Default to a targeted task.

Do not survey or rediscover the repository at the beginning of a session.

Do not inspect files merely to understand the project.

## Task scope

Silently classify each request:

- No-repo: do not inspect the repository.
- Targeted: retrieve only the named file, symbol, configuration, or error.
- Diagnostic: inspect the failure first, then only implicated code.
- Structural: use graph and symbol tools for execution flow and blast radius.
- Exhaustive: broad inspection only when explicitly requested.

Assume Targeted unless broader analysis is concretely required.

## Forbidden cold-start actions

Do not begin with:

- recursive directory listings;
- `tree`;
- repository-wide `find`;
- `git ls-files`;
- `rg --files`;
- repository-wide line counts;
- reading every configuration file;
- reading complete large source files.

A new session is not a reason to reconstruct repository knowledge.

## Retrieval escalation

Use the narrowest sufficient level:

1. Existing conversation and project context
2. Exact named file, symbol, configuration, or error
3. Direct callers, callees, references, or imports
4. Bounded text or structural search
5. Smallest related subsystem
6. Repository-wide exploration

Stop as soon as sufficient evidence is available.

Do not inspect another file because it might be useful.

Expand scope only for a specific unresolved symbol, caller, dependency,
configuration source, contradiction, or test failure.

## GitNexus

Use GitNexus for:

- locating an unknown implementation;
- tracing execution flows;
- dependency and blast-radius analysis;
- changing public or shared symbols;
- cross-file refactoring;
- symbol renaming.

Do not use GitNexus for:

- conceptual questions;
- documentation or comments;
- isolated configuration changes;
- known local targets;
- trivial edits.

Run impact analysis before structural, shared-symbol, public API, or cross-file
changes.

Warn before proceeding when impact analysis reports HIGH or CRITICAL risk.

Run `detect_changes()` before committing non-trivial code changes.

Use GitNexus rename operations instead of text replacement for symbol renames.

## Symbol and file retrieval

Prefer symbol-level retrieval for functions, classes, methods, callers, and
references.

Use bounded search for exact strings, configuration keys, error messages,
documentation, generated files, and logs.

Do not read a complete source file over 300 lines unless:

- the entire file is directly under review;
- symbolic retrieval is unavailable;
- bounded retrieval was insufficient; or
- whole-file consistency must be checked.

Do not read the same unchanged file twice.

Do not read neighboring files preemptively.

## Shell output

Shell output is compressed automatically by the configured sqz hook.

Do not manually stack RTK or another compressor on top of sqz.

Do not stream long-running or verbose commands directly into model context.

For large commands:

1. Save complete output under `.agent/raw/`.
2. Preserve the command and exit status.
3. Return only errors, warnings, important metrics, and relevant final lines.
4. Return the raw-output path.
5. Read raw output only in bounded sections when necessary.

## Training

Do not monitor training output line by line.

Do not repeatedly poll batch-level progress.

Prefer structured status containing:

- process state;
- current epoch;
- current metrics;
- best metrics;
- learning-rate changes;
- warnings and errors;
- completion result;
- raw-log path.

React only to meaningful events such as:

- a new best metric;
- learning-rate changes;
- metric regression;
- NaN or Inf;
- CUDA or out-of-memory errors;
- traceback;
- stalled progress;
- process exit;
- early stopping;
- completion.

## Tests and builds

Run the smallest relevant test first.

Return only:

- failed tests;
- minimal relevant stack frames;
- exit status;
- reproduction command;
- compact summary.

Do not list passing tests unless explicitly requested.

Do not run the complete test suite before focused tests unless required.

## Project context

Use `.agent/PROJECT.md` for stable project facts:

- important directories;
- entry points;
- build and test commands;
- deployment architecture;
- conventions;
- known constraints.

Use `.agent/STATE.md` for current work:

- current objective;
- completed work;
- relevant files and symbols;
- confirmed findings;
- commands already executed;
- current failure;
- next action.

Do not store raw logs, full histories, full diffs, or conversation transcripts in
these files.

## Output

Return:

- changed chunks rather than complete files;
- compact findings rather than raw logs;
- relevant stack frames rather than full traces;
- exact commands rather than broad tutorials;
- confirmed evidence rather than speculation.

Correctness depends on relevant evidence, not maximum context.
