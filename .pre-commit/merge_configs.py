#!/usr/bin/env python3
from pathlib import Path

precommit_dir = Path(".pre-commit")
merged_file = Path(".pre-commit-config.yaml")

# Collect all fragment files
fragments = sorted(precommit_dir.glob("*.yaml"))
merged_lines = ["repos:\n"]

for frag in fragments:
    with frag.open() as f:
        for line in f:
            # Skip top-level 'repos:' lines in fragments
            if line.strip() == "repos:":
                continue
            merged_lines.append(line if line.strip() else "\n")

# Write merged file
_ = merged_file.write_text("".join(merged_lines))
print(f"Merged {len(fragments)} fragment(s) into {merged_file}")
