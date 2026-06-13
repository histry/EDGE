# V34.1 strict warp-budget negotiation fix

## Root cause

The failed V34 retrieval ranked all events first, kept a generic top-K, then
computed a candidate-specific dynamic transition. For a 57-frame locked slot,
the legacy physical-duration heuristic frequently saturated at the slot budget
(39 transition frames), leaving only 18 content frames. If indexed events start
around 24 frames, every top-K candidate has warp below 0.82 and the beam dies.

This is not evidence that the event database has no valid event. It is a budget
ordering bug: the transition heuristic was treated as a hard condition before
strict warp feasibility was negotiated.

## Fix

1. Construct the shortlist inside the global strict-warp-feasible event set.
2. Convert each event's warp interval into an exact integer content interval.
3. Project the desired music/physical transition length onto that legal interval.
4. Record every adjustment and apply a score penalty.
5. Keep `V34_WARP_HARD_PRUNE=1` and `V32_MAX_WARP_VIOLATIONS=0`.
6. Let the post-generation cross-boundary absolute gate decide physical safety.

This preserves music-boundary lock and exact song length. It does not relax the
paper warp range.
