# Stage 5: bounded temporal deep result lifetime

Timeout: 12 seconds from analysis start. Freshness: at most 15 seconds from the
snapshot. Display TTL: 5 seconds from application, also capped by snapshot age.
The suggested 5-second freshness would reject every normal approximately
10-second SmolVLM run; 15 seconds permits these runs while retaining a short
display lifetime. Slow startup/model loading counts toward the 12-second timeout.

AnalysisSnapshot already freezes pixels, WorldState (including detections and
predicted tracks), observation timestamp and motion evidence through immutable
bytes/JSON. Its optional snapshot_monotonic field records the observation age at
capture. All later duration checks use monotonic time. Wall timestamps are only
metadata/log values. Older snapshot constructors remain supported.

DeepAnalysisState is the safe active deep state, separate from latest_reasoning.
It owns analysis ID, monotonic sequence, session generation and a single worker
slot. Results carry source, snapshot/start/completion timestamps, age, sequence,
status, and track/prediction IDs when available. Completion runs one guarded
publication callback; rejected temporal results cannot create evaluation records,
update calibration or publish reasoning. Scene descriptions now have independent
admission and lifetime (see SCENE_DIAGNOSIS.md). Fusion/orchestrator work
before admission remains local to the captured snapshot, never current UI evidence.

Statuses:

- stale: snapshot age exceeds 15 seconds.
- timed_out: analysis execution exceeds 12 seconds (also checked on UI ticks).
- superseded: an equal/newer sequence was already applied, or the originating
  session was invalidated by close/reset.
- fresh: admitted by all checks.

A newer snapshot is not queued while a worker is active. Sequence rejection also
protects against duplicate/out-of-order completions. Timeout does not free the
slot while the old daemon is still running. New work can start after it finishes;
a permanently stuck model leaves deep analysis unavailable for that session,
while camera events and live inference remain responsive. No thread is killed.
Close invalidates publication first, then joins for at most 0.10 seconds.

Selection: current temporal_live has priority. Otherwise an admitted, unexpired
deep result is available as fallback. After deep expires, selection returns to
the current reasoning (normally live or indeterminate). The display hold cache
cannot resurrect a deep result. Deep expiry does not reset EVAL/CAL memory.
No extra overlay line is added. No tracker/calibration/model changes are made.

Events: TEMPORAL_DEEP_STARTED, APPLIED, STALE, TIMED_OUT, SUPERSEDED and CLEARED,
each with analysis ID, sequence, age and duration. APPLIED/CLEARED refer to the
temporal deep active view; ordinary synchronous photo analysis retains its API.

Validation: 94 unittest cases passed, including fresh admission, stale/timeout/
superseded publication exclusion, sequence ordering, single-worker limit, TTL,
display-cache exclusion, live priority, snapshot isolation, wall-clock jumps,
bounded shutdown and camera event pumping during blocked deep work. Existing
evaluation, calibration, tracker and observation-lifetime tests also passed.

Camera checks (not yet performed):

1. Start live camera and move steadily. Watch TEMPORAL_DEEP_STARTED and verify
   the snapshot time/sequence correspond to the analyzed frame.
2. For a completion within 12 seconds and snapshot age <=15, check APPLIED when
   a temporal deep candidate exists. Fresh live motion remains preferred.
3. Stop live motion so deep can be selected as fallback. Verify it disappears
   within five seconds of application, or earlier at the snapshot-age limit.
4. For a run exceeding 12 seconds, verify TIMED_OUT and no deep UI/evaluation
   publication. The worker may finish later; no second worker should pile up.
5. Close the camera during analysis. It should return promptly; late completion
   must be SUPERSEDED and cannot alter a reopened session.
6. Confirm TTL clearing leaves the existing EVAL/CAL records intact. Use IDs in
   the event log to distinguish the expired deep view from evaluation history.
