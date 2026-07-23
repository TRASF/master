#!/usr/bin/env python3

import hashlib
import json
import os
import re
import sys
from pathlib import Path


def load_input():
    try:
        return json.load(sys.stdin)
    except Exception:
        return {}


def output_deny(event_name, reason):
    if event_name == "BeforeTool":
        # Gemini / Antigravity
        print(json.dumps({
            "decision": "deny",
            "reason": reason,
            "systemMessage": "Repository token policy blocked this tool call."
        }))
    else:
        # Claude Code / Codex
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason
            }
        }))


def output_allow(event_name):
    if event_name == "BeforeTool":
        print(json.dumps({"decision": "allow"}))
    else:
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow"
            }
        }))


def first_value(mapping, names, default=""):
    for name in names:
        value = mapping.get(name)
        if value is not None:
            return value
    return default


def resolve_path(raw_path, cwd):
    if not raw_path:
        return None

    path = Path(str(raw_path)).expanduser()

    if not path.is_absolute():
        path = Path(cwd) / path

    try:
        return path.resolve()
    except OSError:
        return path


def is_inside(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def file_line_count(path):
    try:
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return 0


def file_hash(path):
    digest = hashlib.sha256()

    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


def _merge_read_ranges(ranges):
    normalized = []

    for value in ranges:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue

        try:
            start = int(value[0])
            end = int(value[1])
        except (TypeError, ValueError):
            continue

        if start < 0 or end < start:
            continue

        normalized.append([start, end])

    normalized.sort(key=lambda item: (item[0], item[1]))

    merged = []

    for start, end in normalized:
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return merged


def _read_range_is_covered(ranges, start, end):
    for existing_start, existing_end in _merge_read_ranges(ranges):
        if existing_start <= start and existing_end >= end:
            return True

    return False


def duplicate_read(
    session_id,
    target,
    digest,
    state_dir,
    scope="full",
):
    """
    Return True only when the requested content has already been covered
    in this session.

    Changing the file content resets coverage because the digest changes.
    """

    import hashlib
    import json
    from pathlib import Path

    target = Path(target).resolve()
    state_dir = Path(state_dir)

    range_state_dir = state_dir / "read-ranges-v2"
    range_state_dir.mkdir(parents=True, exist_ok=True)

    session_key = hashlib.sha256(
        str(session_id).encode("utf-8")
    ).hexdigest()[:24]

    state_path = range_state_dir / f"{session_key}.json"

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))

        if not isinstance(state, dict):
            state = {}
    except (OSError, ValueError, TypeError):
        state = {}

    target_key = str(target)
    record = state.get(target_key)

    if not isinstance(record, dict) or record.get("digest") != digest:
        record = {
            "digest": digest,
            "full": False,
            "line_ranges": [],
            "offset_ranges": [],
            "other_scopes": [],
        }

    duplicate = False
    changed = False

    if record.get("full") is True:
        duplicate = True

    elif scope == "full":
        # A full read after partial reads still adds unseen content.
        record["full"] = True
        changed = True

    elif scope.startswith("lines:"):
        try:
            _, raw_start, raw_end = scope.split(":", 2)
            start = int(raw_start)
            end = int(raw_end)
        except (TypeError, ValueError):
            start = end = None

        if start is None or end is None or start < 1 or end < start:
            scopes = set(record.get("other_scopes", []))
            duplicate = scope in scopes

            if not duplicate:
                scopes.add(scope)
                record["other_scopes"] = sorted(scopes)
                changed = True
        else:
            ranges = record.get("line_ranges", [])
            duplicate = _read_range_is_covered(ranges, start, end)

            if not duplicate:
                ranges.append([start, end])
                record["line_ranges"] = _merge_read_ranges(ranges)
                changed = True

    elif scope.startswith("offset:"):
        try:
            _, raw_offset, _, raw_limit = scope.split(":", 3)
            start = int(raw_offset)
            limit = int(raw_limit)
            end = start + limit - 1
        except (TypeError, ValueError):
            start = end = None

        if start is None or end is None or start < 0 or end < start:
            scopes = set(record.get("other_scopes", []))
            duplicate = scope in scopes

            if not duplicate:
                scopes.add(scope)
                record["other_scopes"] = sorted(scopes)
                changed = True
        else:
            ranges = record.get("offset_ranges", [])
            duplicate = _read_range_is_covered(ranges, start, end)

            if not duplicate:
                ranges.append([start, end])
                record["offset_ranges"] = _merge_read_ranges(ranges)
                changed = True

    else:
        scopes = set(record.get("other_scopes", []))
        duplicate = scope in scopes

        if not duplicate:
            scopes.add(scope)
            record["other_scopes"] = sorted(scopes)
            changed = True

    if changed:
        state[target_key] = record

        temporary_path = state_path.with_suffix(".tmp")
        temporary_path.write_text(
            json.dumps(state, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary_path.replace(state_path)

    return duplicate




data = load_input()

event_name = data.get("hook_event_name", "PreToolUse")
tool_name = str(data.get("tool_name", ""))
tool_input = data.get("tool_input") or {}
cwd = Path(data.get("cwd") or os.getcwd()).resolve()

try:
    repo_root_text = os.popen(
        f"git -C {json.dumps(str(cwd))} rev-parse --show-toplevel 2>/dev/null"
    ).read().strip()
except Exception:
    repo_root_text = ""

repo_root = Path(repo_root_text or cwd).resolve()
state_dir = repo_root / ".agent" / "hook-state"
session_id = str(data.get("session_id") or data.get("turn_id") or "default")

normalized_tool = tool_name.lower()

# -------------------------------------------------------------------------
# Shell-command policy
# -------------------------------------------------------------------------

if normalized_tool in {
    "bash",
    "run_shell_command",
    "shell",
    "execute",
    "execute_command",
}:
    command = str(first_value(
        tool_input,
        ["command", "cmd", "shell_command"],
        ""
    )).strip()

    forbidden_discovery = [
        (
            r"(^|[;&|]\s*)git\s+ls-files\b",
            "Repository-wide git file enumeration is blocked. "
            "Use GitNexus or a targeted symbol/file query."
        ),
        (
            r"(^|[;&|]\s*)rg\s+--files\b",
            "Repository-wide file enumeration is blocked. "
            "Search a specific symbol, path, or file pattern."
        ),
        (
            r"(^|[;&|]\s*)(ls\s+-[^\n]*R|tree)(\s|$)",
            "Recursive repository listing is blocked. "
            "Use GitNexus or list only a specific subdirectory."
        ),
        (
            r"\bfind\b.*\|\s*xargs\s+wc\s+-l\b",
            "Repository-wide line counting is blocked."
        ),
        (
            r"\bfind\s+(?:\.|src|model|configs)\b.*"
            r"(?:-name|-type).*\|\s*(?:xargs\s+)?(?:wc|sort)\b",
            "Broad repository discovery is blocked. "
            "Use GitNexus or a bounded search."
        ),
        (
            r"\bwc\s+-l\s+(?:src|model|configs)/.*\*",
            "Repository-wide source line counting is blocked."
        ),
    ]

    for pattern, reason in forbidden_discovery:
        if re.search(pattern, command, flags=re.IGNORECASE | re.DOTALL):
            output_deny(event_name, reason)
            sys.exit(0)

    direct_training = re.search(
        r"\bpython(?:3)?\b[^\n;&|]*"
        r"(?:train(?:_[A-Za-z0-9_-]+)?\.py|"
        r"-m\s+[A-Za-z0-9_.]*train[A-Za-z0-9_.-]*)",
        command,
        flags=re.IGNORECASE,
    )

    controlled_training = (
        "tools/trainctl.py" in command
        or (
            ".agent/raw/" in command
            and re.search(r">\s*\S+", command)
            and re.search(r"(?:&\s*$|nohup|setsid)", command)
        )
    )

    if direct_training and not controlled_training:
        output_deny(
            event_name,
            "Direct foreground training is blocked because it can stream "
            "thousands of lines into context. Use tools/trainctl.py, or run "
            "the process detached with complete output redirected under "
            ".agent/raw/."
        )
        sys.exit(0)

# -------------------------------------------------------------------------
# Directory-listing policy
# -------------------------------------------------------------------------

if normalized_tool in {
    "listdir",
    "list_dir",
    "list_directory",
    "directory_list",
}:
    raw_path = first_value(
        tool_input,
        ["path", "directory", "dir_path"],
        "."
    )
    target = resolve_path(raw_path, cwd)

    if target and target == repo_root:
        output_deny(
            event_name,
            "Repository-root enumeration is blocked by default. "
            "Use existing project context or GitNexus. Listing a specific "
            "subdirectory remains allowed."
        )
        sys.exit(0)

# -------------------------------------------------------------------------
# File-read policy
# -------------------------------------------------------------------------

if normalized_tool in {
    "read",
    "read_file",
    "readfile",
    "file_read",
}:
    raw_path = first_value(
        tool_input,
        ["file_path", "path", "filename"],
        ""
    )
    target = resolve_path(raw_path, cwd)

    if target and target.is_file() and is_inside(target, repo_root):
        line_count = file_line_count(target)

        bounded_fields = [
            "offset",
            "limit",
            "start_line",
            "end_line",
            "line_start",
            "line_end",
        ]
        bounded = any(
            tool_input.get(field) not in (None, "", 0)
            for field in bounded_fields
        )

        if line_count > 300 and not bounded:
            output_deny(
                event_name,
                f"Unbounded read blocked: {target} has {line_count} lines. "
                "Use symbol retrieval or request a bounded line range."
            )
            sys.exit(0)

        digest = file_hash(target)

        start_line = tool_input.get("start_line")
        if start_line in (None, ""):
            start_line = tool_input.get("line_start")

        end_line = tool_input.get("end_line")
        if end_line in (None, ""):
            end_line = tool_input.get("line_end")

        offset = tool_input.get("offset")
        limit = tool_input.get("limit")

        if (
            start_line not in (None, "")
            and end_line not in (None, "")
        ):
            scope = f"lines:{start_line}:{end_line}"

        elif (
            offset not in (None, "")
            and limit not in (None, "")
        ):
            scope = f"offset:{offset}:limit:{limit}"

        else:
            scope = "full"

        if duplicate_read(
            session_id,
            target,
            digest,
            state_dir,
            scope,
        ):
            output_deny(
                event_name,
                f"Duplicate unchanged read blocked: {target}. "
                "Use the content already in context or retrieve only a "
                "specific missing section."
            )
            sys.exit(0)

output_allow(event_name)
