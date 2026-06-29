# Agent Operating Rules

## Prime directive

Minimize token use while preserving correctness. Prefer the smallest sufficient action.

Do not dump large files, logs, JSON, experiment histories, web pages, artifacts, or command output into context. Fetch narrowly, summarize outside the model, and return only useful results.

## Tool priority

For non-trivial tasks, use this order:

1. `ponytail` — choose the simplest sufficient solution.
2. `repo_map` — inspect repo structure, symbols, imports, and blast radius.
3. `jcodemunch` — retrieve exact symbols, AST nodes, callers, or file outlines.
4. `rg` — bounded text search with limited matches and context lines.
5. `ast_grep` — structural code search or rewrite when text search is too noisy.
6. `jq_yq` — slice JSON/YAML before reasoning over it.
7. `sqz` — compress long logs, build output, command output, traces, stack traces, and large JSON.
8. `test_slicer` — run tests and return only failures, key errors, minimal stack frames, and reproduction commands.
9. `diff_tool` — compare or edit using unified diffs and changed chunks only.

Read full files only when they are small or targeted retrieval is insufficient.

## Large-fetch protocol

For tools that can return large data — W&B, web fetch, APIs, databases, logs, traces, artifacts, metrics, search results, or files — use this escalation ladder:

metadata → schema/keys → filtered list → aggregate summary → sampled rows → selected raw rows → full export.

Stop as soon as the user’s question can be answered correctly.

Before fetching data, apply filters such as project, run ID, tag, state, time range, metric key, file glob, query predicate, or page section.

Before returning data, reduce it with summaries such as top-k, best/worst, min/max/mean, last value, diff, failure count, histogram, outliers, regression, or compact table.

For full exports, save to disk and return only the file path, row count, columns, filters used, and compact findings.

## W&B-specific rules

For Weights & Biases work:

* Start with run metadata and summaries.
* Prefer run name, run ID, state, tags, config, created time, and selected summary metrics.
* Do not scan full metric history first.
* Use sampled history for trends and plots.
* Use full history only for exact analysis, and always restrict by run IDs, metric keys, step range, and page size.
* Do not paste full metric tables, raw histories, media blobs, artifacts, or raw JSON into context.
* Cache downloaded histories or artifacts locally when useful.

Default W&B flow:

1. List filtered runs.
2. Fetch selected summary metrics.
3. Rank or aggregate locally.
4. Return a compact table.
5. Fetch sampled or full history only for the few runs that need deeper analysis.

## Output rules

Return only what is needed:

* changed chunks, not full files;
* concise diagnosis, not raw logs;
* exact commands, not long explanations;
* compact tables, not raw datasets;
* key findings, not entire traces.

Avoid speculative abstractions, broad rewrites, unnecessary dependencies, and extra tools when the task is trivial.
