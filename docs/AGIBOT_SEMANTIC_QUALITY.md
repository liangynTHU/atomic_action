# AgiBot Semantic Annotation Quality

## Scope And Ground Truth

The initial audit covers all 126 generated segments of ten episodes, not all
episodes in the AgiBot source dataset. It found 19 confirmed issue segments,
including seven physical-arm attribution errors. Each segment counts once even
when multiple claims are wrong.

Iterations use the same five episodes and 56 fixed segments: tasks 463, 511,
532, 751 and 765. This set includes a normal control, handover, released-object
withdrawal, a knife seen by both wrist cameras, and empty closure beside shelf
bottles. The initial labels have 15 confirmed issue segments in this matched set.
This is an error-enriched development set, not an unbiased dataset error rate.

The audits inspect all five distinct chronological teacher observations when
available, across head and both wrists, alongside named physical-arm measured
signals. They do not inspect every frame of every source video. "No clear issue"
means no clear contradiction in this evidence, not a certificate of correctness.
Ambiguous contact, transient phases and static finger geometry remain uncertain.

## Method

1. Anchor physical arm identity before writing an object or activity.
   The wrist camera name identifies its mounting arm, not every gripper visible
   in its image. Trace the nearby mounted gripper, the other distant arm, and
   cross-view motion. Never use screen-left/right as physical arm identity.
2. Inspect current start and target fingertips, then intermediate observations.
   Treat held, empty and unknown as distinct states. Nearby objects, outside
   contact and encoder closure do not prove a grasp. Require visible finger
   geometry or consistent object-gripper co-motion.
3. Separate transport, approach, contact, grasp, release and empty withdrawal.
   Opening an empty gripper is not release. A supported object that stays still
   while an empty wrist retreats is not carried or rotated by that wrist.
   A cloth in motion is not necessarily wiping; a tool nearby does not establish
   the functional activity or the purpose of a motion.
4. Remove coarse source phases from the teacher input. Keep task wording as
   context only. Supply the preceding interval's full sampled transition,
   including its target, so historical holding cannot override current release.
5. Generate a candidate, then request a separate visual correction using
   chronological strips for each named camera. Reviewers receive measured facts
   and the untrusted candidate, not the source phase or task narrative. Only
   accepted corrected sub_task strings enter training; visual states stay in
   annotation caches for diagnosis.
6. Carry the previous accepted target-state record into the next batch.
   Adjacent target/start frames are identical observations. Reject confident
   held/empty contradictions both within and across batches and provide the
   exact frame and physical arm in retry feedback. Unknown is allowed, and
   previous model observations are explicitly not ground truth. Object names
   are not compared literally: knife and cleaver may name the same visible tool.
7. Test on frozen, paired inputs; inspect every generated label against images.
   Save findings, evidence sheets, API responses, retry counts and provenance.
   Do not repair the pilot's labels manually or reuse old teacher labels.

## Prompt Versus Program Responsibilities

Prompts enforce the visual reasoning policy: physical arm anchoring, fingertip
ownership, temporal precedence, supported activities, and conservative wording.
The program enforces the parts that are actually checkable: response shape,
segment order, required physical-arm verbs, valid state enums, evidence frame
membership, and shared-frame state consistency. Neither is a general semantic
truth oracle. The same GLM model generates and reviews, so correlated visual
mistakes can survive both stages. A consistent but wrong ownership ledger can
also survive continuity checks.

The added checks do not infer holding from encoder thresholds, swap physical
arms, change segmentation, or force static finger descriptions from numerical
openness. They do not modify student inputs, numeric targets, guide_action,
num_chunks, or training JSON shape.

## Paired Results

| Version | Change | Confirmed / 56 | Arm Errors | Uncertain |
| --- | --- | ---: | ---: | ---: |
| Original | Original GLM teacher | 15 (26.79%) | 7 | 1 |
| Grounded v2 | Physical-arm facts and ownership prompt | 6 (10.71%) | 0 | 0 |
| Grounded v3 | Full previous transition, no coarse source phases | 6 (10.71%) | 4 | 1 |
| Grounded v4 | Candidate plus separate camera-strip review | 3 (5.36%) | 0 | 2 |
| Grounded v6 | Shared-frame checks, larger strips, strict fenced JSON parsing | 4 (7.14%) | 0 | 2 |
| Grounded v7 | Two-row camera sheets plus full head boundary images | 0 (0.00%) | 0 | 3 |

The v3 arm-attribution regression demonstrates why prompt length and stronger
instructions alone are not evidence of improvement. The v4 shelf example also
records the left hand as empty at frame 339 in one batch and held at that same
frame in the next, while publishing a false release. Grounded v5 adds explicit
shared-frame continuity and activity-evidence requirements to address these
remaining failure modes.

Grounded v5 completed four of the five requested episodes. Its 45 committed
segments have four confirmed object-category errors and two uncertain phases.
All four errors are the same knife misnamed as a spatula in the review of the
first batch; the candidate had called it a knife. Task 765 passed its first eight
segments without the false left-hand release, but the last review failed JSON
validation three times, so no part of that episode was exported. V5 is not a
complete 56-segment paired result and its 4/45 rate must not be presented as one.

Grounded v6 preserves single-camera strip frames at 640x480 rather than 320x240,
adds whole-object category evidence rules, and supplies a JSON output skeleton.
Its parser accepts only a complete plain JSON object or a single complete
JSON-fenced object; it does not repair malformed JSON, extract embedded prose,
or read private reasoning. This follows the real v5 failures: two fenced
responses followed by a malformed extra bracket on the third response.

V6 completed all five episodes but misnamed the knife as a carrot in its first
four segments. Its reviewer corrected arm ownership while introducing an object
category error. V7 changes the five-frame single-camera layout from a very wide
3200x516 strip to a 1920x1032 two-row sheet, with explicit reading order, and
adds separate full head images at each requested interval's start and target.
All source observations remain unchanged; this only changes teacher evidence
presentation. The knife's category and left-hand ownership remain consistent
through all 11 segments in the real V7 output.

All 56 V7 labels were inspected against the five sampled observations.
No confirmed issue was found under the conservative baseline standard; three
remain uncertain: cloth-surface pressure at task 463 segment 8, exact alignment
at task 532 segment 6, and slight finger opening at task 765 segment 4 despite
the nearly closed encoder proxy. These are not silently counted as correct.
V7's zero confirmed issues is a development-set result after repeated tuning,
not a generalization estimate. An untouched held-out set should be audited
before assuming production-wide accuracy. No generated labels were hand-edited.

The paired export check passes for all 56 segments and 321 student image
references. Apart from sub_task, the matching previous-command sub_task and
run-specific paths, source identities, splits, segment fields, numeric arrays,
guides, chunks, student messages, image bytes/order and training shape are
identical to the original run. The portable review also passes all segment
seeks, three-camera playback, exact training-message display, image loading,
frame stepping, downloads and mobile overflow checks.

### Real API Usage

| Complete Pilot | Candidate Requests | Review Requests | Content Retries | Input Tokens | Output Tokens | Reported Credit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V4 | 16 | 24 | 8 | 1,066,483 | 16,525 | 7.95 |
| V6 | 17 | 18 | 3 | 1,390,674 | 12,313 | 10.12 |
| V7 | 17 | 18 | 3 | 1,542,398 | 12,341 | 10.65 |

Counts include failed content attempts but exclude startup probes; none of these
runs had transport errors. Credit is the upstream's reported unit, not a
verified currency amount. V7's retries were one candidate index mismatch and
two review held states without object names; all were corrected on bounded
retry. Requests and final responses are preserved in each episode's journal.

## Reproduce

Use a new output directory for every prompt or semantic implementation version.
Production freezes code and prompts; re-running an old directory intentionally
continues its old frozen version.

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  /jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python \
  scripts/relabel_agibot_pilot.py \
  --baseline artifacts/agibot_glm53_xhigh_10_20261022_run \
  --output artifacts/agibot_glm53_grounded_v7_reproduce_run \
  --selection artifacts/agibot_glm53_grounded_v2_selection.json

bash agibot.sh start \
  --output artifacts/agibot_glm53_grounded_v7_reproduce_run \
  --selection artifacts/agibot_glm53_grounded_v2_selection.json
```

The pilot helper verifies source identities and committed old numeric, video,
segmentation and media stages. It permits changes only to AgiBot annotation,
teacher contract, media and API response parsing code, checks identical student/numerical contracts
and dependency versions, copies student images byte-for-byte, rebuilds teacher
montages, and never copies labels.

API settings remain glm-5.3-flash-ioa, xhigh, temperature 0.1, max_tokens 8192,
concurrency 4 and batch_size 4. Two stages approximately double the base number
of requests; schema and continuity retries add requests. At most three content
attempts per stage and three bounded transport retries per content attempt are
allowed. Exhausted review never publishes the candidate. Resume retains accepted
candidates, reviews and prior boundary records without regenerating them.

Audit evidence and reports are under artifacts/agibot_glm53_grounded_vN_audit.
Committed exports and hashes are reverified when writing audit summaries.

The completed V7 artifacts are:

- `artifacts/agibot_glm53_grounded_v7_run/accepted.jsonl`: five committed exports.
- `artifacts/agibot_glm53_grounded_v7_review/index.html`: portable offline review.
- `artifacts/agibot_glm53_grounded_v7_audit/REPORT.md`: all-56 audit and uncertainties.
- `artifacts/agibot_glm53_grounded_v7_audit/PAIRED_INPUTS.json`: unchanged-input verification.

```bash
env PYTHONDONTWRITEBYTECODE=1 \
  /jizhicfs/peterrao/miniconda3/envs/cosmos3/bin/python \
  scripts/check_agibot_paired_inputs.py \
  --baseline artifacts/agibot_glm53_xhigh_10_20261022_run \
  --run artifacts/agibot_glm53_grounded_v7_run \
  --output artifacts/agibot_glm53_grounded_v7_audit/PAIRED_INPUTS.json
```
