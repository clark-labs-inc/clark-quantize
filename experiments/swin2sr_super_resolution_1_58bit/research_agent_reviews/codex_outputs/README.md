# Visible Codex output archive

This directory contains only visible assistant output emitted by the bounded
research side-agent turns for this Swin2SR program.

Included fields are timestamp, phase, message ID, task, session, turn, role,
and visible output text. Hidden reasoning, system/developer instructions, user
messages, credentials, tool internals, inherited conversation history, and
machine-private session machinery are excluded.

The JSONL files preserve commentary and final-answer messages in chronological
order. The adjacent manifest pins their source identities, message counts,
hashes, and byte sizes.
