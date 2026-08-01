# Failed browser-export preseal

The v1 materialization command failed before creating any model state because
the sealed overlay omitted `retained_precision_islands.py`, which is imported by
the new reusable retained-parent loader. The import error occurred before the
materialization output directory was created. This immutable v1 manifest is not
reused or amended; v2 adds the missing dependency to the hashed overlay and is
sealed independently.
