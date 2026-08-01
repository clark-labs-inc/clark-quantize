# Research-agent review registry

This directory preserves research reviews produced by bounded Codex side
agents for the BackdropKit Swin2SR program.

These reports are scientific leads, not primary evidence. Load-bearing claims
must resolve to the linked checkpoint, manifest, source file, official upstream
artifact, or deterministic evaluation before they change a promotion decision.

The original Codex session databases are machine-private and are not committed.
Every report records its source task identity so the local transcript can be
reopened if needed. `manifest.json` pins the captured report bytes and source
session identities.

`codex_outputs/` preserves every visible commentary and final-answer message
from the three bounded research turns. It excludes hidden reasoning, system
instructions, user messages, credentials, tool internals, inherited history,
and session machinery.
