# C7 FDST unseen confirmation protocol v3

> Protocol id: `C7-fdst-v3`
>
> Amendment date: 2026-08-09 (Asia/Shanghai)
>
> Status: pre-result amendment; no formal split manifest, attempt lock,
> feature cache, prediction, or model result existed when this version was
> written.

## 1. Why v3 exists

V2 was written before the FDST files were inspected.  After download, the
path-only layout inspection found the expected 60 official-train videos and
150 paired frames per video.  A label-only diagnostic then found 134 finite
head points in 133 of 9,000 official-train frames that lie just outside the
decoded image boundary.  All excursions were on the left or top boundary and
the maximum absolute excursion was 8 pixels.  No split, feature, prediction,
metric, or model result had been generated.

V2 allowed only a one-pixel boundary tolerance and therefore halted on the
official files, while the frozen loader already clips coordinates to the image
before assigning patch targets.  V3 records this discrepancy openly and
replaces V2 before any formal run.  V2 remains in the repository as historical
pre-data documentation and must not be presented as the executed protocol.

## 2. Frozen v3 coordinate policy

Every eligible annotation is parsed before formal artifacts are written.

1. Every coordinate must be finite.
2. For an image of width `W` and height `H`, a point is admissible only when
   `x` is within one percent of `W` outside the horizontal frame and `y` is
   within one percent of `H` outside the vertical frame.
3. Admissible points outside `[0, W-1] x [0, H-1]` are counted and then clipped
   by the frozen loader to the nearest frame boundary.
4. A point outside the one-percent envelope halts the preflight.
5. The manifest records audit scope, frames and points checked, empty frames,
   frames containing clipped points, clipped-point count, maximum horizontal
   and vertical overflow, and hard-out-of-bounds count.

The one-percent rule is dimension-relative and fixed before any model access.
It is not adjusted to the observed maximum and may not be changed after the
formal split-freeze commit.

## 3. Dataset evidence and eligible partition

The official author repository states that FDST contains 100 videos from 13
scenes: 60 training videos with 9,000 frames and 40 test videos with 6,000
frames.  Only `train_data` is eligible.  The official `test_data` partition is
excluded from scene selection, splitting, fitting, validation, and evaluation.

Raw images and annotations remain local-only and must not be redistributed.
Source provenance is recorded in `docs/FDST_SOURCE_RECORD.json`.

## 4. Reviewed scene identity

The reviewed mapping is recorded in
`docs/FDST_SCENE_IDENTITY_AUDIT.json` and normalized into the formal split
directory by preflight.  It uses the 13 contiguous official video-number
ranges, covers all 60 official-train directories exactly once, and was checked
against a local-only contact sheet of the first frame of every train video.
The contact sheet is not committed; only its SHA256 is recorded.

Scene identity review may use directory identity and fixed-camera/background
evidence only.  It may not use annotation counts, features, predictions, or
evaluation metrics.

## 5. Domain construction

1. A frame is eligible only when an image and same-directory, same-stem JSON
   annotation both exist below `train_data/<video>/`.
2. Videos with at least 120 paired frames qualify.
3. Within each reviewed scene, retain the qualifying video with the largest
   pair count; ties use lexicographic `video_id`.
4. Retain the six scenes whose representatives have the largest pair counts;
   ties use `scene_id`, then `video_id`.
5. Order the six domains by ascending
   `sha256("fdst|{scene_id}|{video_id}|order-v3")`.

Selection may see file identities and pair availability only.  It may not see
pixels, point counts, features, predictions, or metrics.

## 6. Temporal split and model freeze

Each selected video is put in natural numeric frame order and divided into
non-overlapping blocks of 10 consecutive frames.  Blocks are assigned
contiguously to approximately 60% fit, 20% validation, and 20% test; no block
crosses a split boundary.  The uncertainty unit is the fixed 10-frame block.

The backbone, feature resolution, DCT patch target, ridge penalty, fusion
weight, method roster, fixed-factor grid, train-only selectors, bootstrap, and
decision thresholds are unchanged from Appendix A.  FDST uses full-frame point
counts because it has no WorldExpo-style ROI polygons.

## 7. Single-shot gates

The formal runner accepts no experimental arguments and refuses to start unless:

- Appendix A and the manifest both identify `C7-fdst-v3`;
- the git worktree is clean;
- all config, split, scene-map, source-record, and manifest hashes re-verify;
- six domains have six distinct reviewed scene identities;
- all split ids form valid fixed temporal blocks;
- the v3 label audit has no hard-out-of-bounds point;
- no prior `runs_real/fdst_c7/fdst_c7.json` or `attempt.lock` exists.

The runner creates an atomic `attempt.lock` before feature extraction or model
calculation.  A crash after lock creation is a started formal attempt and is not
silently rerun.

## 8. Decision and reporting rules

Opportunity is present only when the fixed-factor Oracle improves at least
0.5% over `f=1` and its selection-adjusted 95% paired block-bootstrap lower
bound is above zero.  A train-only method is `positive` only with at least 0.5%
relative improvement, a positive lower bound, and no domain harmed by more than
1%.  `safe` requires no worse than -0.5% average improvement and a lower bound
above -1%.

The first formal outcome is reported whether positive, null, harmful, or
halted.  FDST is one external data source, not six independent replications.

## 9. Execution sequence

```bash
cd "${RRCL_ROOT}"
python3 -m pytest -q tests/test_fdst_protocol.py
python3 scripts/preflight_fdst.py \
  --root "${FDST_ROOT}" \
  --scene-map configs/fdst_scene_map_candidate.json
```

Review and commit the generated config, normalized scene map, split lists,
manifest, protocol amendment, and implementation on a clean split-freeze
commit.  Only then run:

```bash
python3 scripts/run_fdst_c7.py
```
