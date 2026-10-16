# FoodSeg

## Working rules

- **Never do work yourself. Always delegate.** All implementation, code edits, experiments, reviews and
  investigations must be handed to Sonnet subagents (`Agent` tool, `model: "sonnet"`). The top-level
  session only briefs agents, checks their results, and reports back to the user. The sole exception is
  trivial bookkeeping needed to brief an agent (e.g. listing a directory to find the right files).
