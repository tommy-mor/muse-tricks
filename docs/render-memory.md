# Rendering a ranking into MEMORY.md, gracefully

`sorter render` produces a standalone view file. That is not enough for a
living memory document: a MEMORY.md is edited by many hands (agents,
gardeners, the user) and carries material the sorter knows nothing about.
`sorter render-memory` closes that gap under one rule:

> **The tool owns a marked region of the file. It never owns the file.**

## The contract

1. **One region per scope.** The rendered ranking lives between
   `<!-- sorter:begin scope=<scope> -->` and `<!-- sorter:end -->`.
   Everything outside the markers — a treaty header, hand-written
   sections, editorial notes — is preserved byte-for-byte.
2. **The INBOX stays last.** A MEMORY.md may end with an append-only
   `## INBOX` tray where new claims land before they are ranked. When a
   file has no region yet, the region is inserted directly above the
   INBOX; the tray is never moved, edited, or absorbed by a render.
3. **Idempotent.** The region contains no wall-clock and no randomness:
   rendering an unchanged state into an already-rendered file produces
   byte-identical output. A render is safe to run on a schedule.
4. **Provenance on every bullet.** Each ranked line ends with an HTML
   comment — `<!-- sorter:rank R score S id mem-XXXX -->` — so a reader
   (or a diff) can always tell which item a line came from, where it
   stood, and that a machine put it there. Scores are the sorter's raw
   centrality values; they are evidence, not grades.
5. **Atomic writes.** Output goes to a temp file in the same directory
   and is renamed over the target. A crashed render never leaves a
   half-written MEMORY.md.
6. **Multiple scopes coexist.** Each scope gets its own region; rendering
   one never disturbs another.
7. **Only the largest connected component is a ranking.** The region
   renders the scope's largest connected component of voted pairs,
   ranked on its own induced subgraph — never whole-pool scores. Items
   with no votes are *undefined*: they are not seated at a prior, not
   counted in the header's "top N of M" (M is the component's size),
   and never appear. Smaller components likewise wait outside until
   the dealer bridges them in. If no vote has landed yet, the region
   renders empty: `top 0 of 0`.

## What the tool deliberately does not do

- **No silent re-bucketing.** The ranking is rendered as a single ranked
  section, not scattered into the file's editorial sections (Facts,
  Preferences, …). Sorting claims into an editorial structure is an
  agent's judgment call, made in the open — not a side effect of a render.
- **No pruning.** Ranking governs attention, never deletion. Items below
  the `--top` cut simply do not appear in the region; the source bank is
  untouched, and nothing in the file outside the region is removed.
- **No folding.** New claims in the INBOX are not merged into the region
  by the renderer. Folding the tray into rank is a deliberate, reviewed
  act (in the reference workflow: the agent folds in session, then
  re-renders).

## Usage

```
# first render into an existing MEMORY.md (in place)
sorter --state <dir> render-memory --scope world --top 50 --input MEMORY.md

# render to a review copy instead
sorter --state <dir> render-memory --scope world --input MEMORY.md --output MEMORY.next.md

# fresh file (title + region + empty INBOX tray)
sorter --state <dir> render-memory --scope world --output MEMORY.md
```

Output is JSON like every other subcommand, including a `mode` field:
`created`, `replaced`, `inserted-before-inbox`, or `appended`.

## Reference workflow

The workflow this was designed for keeps MEMORY.md as the agent's ranked
working set under a written treaty, with gardener processes appending new
claims to the INBOX only. The render step is the last, mechanical mile:
judgment happens in the votes and in the folding; the renderer just makes
the current ranking visible inside the document — gracefully, and never
at the expense of anything a human or another agent wrote there.
