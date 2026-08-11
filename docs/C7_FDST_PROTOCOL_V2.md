# C7 FDST unseen confirmation protocol v2

> Protocol id: `C7-fdst-v2`
>
> Frozen design date: 2026-08-09 (Asia/Shanghai)
>
> Status: pre-result implementation complete; dataset download, identity audit,
> split freeze, and the single model run have **not** occurred.

## 1. Why this is v2

WorldExpo v1.2 was implemented but withdrawn before data access because the
requested dataset was not obtained. No WorldExpo manifest, attempt lock, feature
cache, prediction, or model result was created. Its protocol and code remain in
the repository as an auditable historical record; they are not silently renamed
or reused as FDST evidence.

FDST changes the natural-domain unit, annotation format, ROI policy, and temporal
grouping. It therefore receives a new protocol id and Appendix A. The scientific
question is unchanged: does a fixed-factor test Oracle establish an opportunity
for non-uniform historical weighting, and can either prespecified train-only rule
select a useful weighting without harmful deployment behavior?

## 2. Dataset evidence and scope

The official author repository describes FDST as 100 videos from 13 scenes, with
60 training videos (9,000 frames), 40 test videos (6,000 frames), and 394,081
head annotations. The project provenance and handling boundary are recorded in
`docs/FDST_SOURCE_RECORD.json`.

Only files below the official `train_data` directory are eligible. The official
`test_data` partition is excluded from enumeration, scene selection, splitting,
model fitting, validation, and evaluation. This retains an untouched official
partition and prevents accidental reuse of published benchmark test data.

Raw FDST images and annotations are local-only and must not be redistributed in
the RRCL release. The official repository did not expose an explicit standalone
raw-dataset license during the 2026-08-09 audit, so the paper and artifact README
must report the official download source and citation without claiming a license
that was not found.

## 3. Identity and domain construction

1. A frame is eligible only when an image and a same-directory, same-stem JSON
   label both exist under `train_data/<video>/...`.
2. `video_id` is the image parent directory relative to `train_data`.
3. A reviewed JSON map assigns every paired `video_id` to one `scene_id`.
   Missing, extra, empty, or `REVIEW_REQUIRED` entries halt the preflight.
4. Videos with at least 120 paired frames qualify.
5. Within each scene, retain the qualifying video with the largest pair count;
   ties use lexicographic `video_id`.
6. Retain the six scenes whose representatives have the largest pair counts;
   ties use `scene_id`, then `video_id`.
7. Order the six domains by ascending
   `sha256("fdst|{scene_id}|{video_id}|order-v2")`.

One selected video is one continual-learning domain. The six domains must have
six distinct reviewed scene identities. Selection sees file identities and pair
availability only—never pixels, point counts, features, predictions, or metrics.

## 4. Temporal split and uncertainty unit

Within a selected video, frames are put in natural numeric filename order and
partitioned into non-overlapping blocks of 10 consecutive frames. A final block
smaller than five frames is invalid. Fixed blocks are then assigned contiguously
to approximately 60% fit, 20% validation, and 20% test, with no block crossing a
split boundary.

The validation block is divided at a fixed-block boundary into precision and
gamma-selection halves. Paired uncertainty resamples the fixed 10-frame test
blocks within each domain. Per-frame bootstrap fallback is forbidden. The
selection-adjusted Oracle bootstrap reselects the best factor over the complete
frozen factor grid within every replicate.

## 5. Targets and model

FDST does not provide the official per-camera ROI polygons used by WorldExpo.
Targets therefore count all valid head points in the full image. This change is
explicit and no FDST/WorldExpo metric is compared as if its field of view were
identical.

The backbone, feature resolution, DCT patch target, ridge penalty, fusion weight,
method roster, fixed-factor grid, selectors, and decision thresholds are exactly
those frozen in WorldExpo v1.2 and copied into `docs/C7_FDST_APPENDIX_A.json`.
Dataset-specific code may only load FDST, validate identities and provenance,
create fixed temporal blocks, or write FDST-named artifacts.

## 6. Single-shot gates

The formal runner accepts no experimental arguments and refuses to start unless:

- Appendix A has the exact protocol, method roster, grid, and development commits;
- the git worktree is clean;
- `configs/fdst_splits/manifest.json` has the formal stamp and all config, split,
  scene-map, and source-record hashes re-verify;
- six domains have six distinct scenes and all split ids form valid 10-frame blocks;
- no `runs_real/fdst_c7/fdst_c7.json` or `attempt.lock` exists.

An atomic `attempt.lock` is created before any feature extraction or model
calculation. It is never automatically removed. A crash after lock creation is a
started formal attempt and requires explicit protocol review; it is not silently
rerun.

## 7. Decision rules and reporting

All improvements and harm thresholds are relative rel-MAE units. Opportunity is
present only when the fixed-factor Oracle improves at least 0.5% over `f=1` and
its selection-adjusted 95% paired block-bootstrap lower bound is above zero.

For each train-only method, `positive` requires at least 0.5% relative
improvement, a paired 95% lower bound above zero, and no domain with more than 1%
relative harm. `safe` is also always reported and requires no worse than -0.5%
average improvement with a lower bound above -1%. Oracle status never chooses
which method criterion applies.

The first formal result is reported whether positive, null, harmful, or halted.
FDST counts as one external data source, not six independent replications. Until
the formal artifact exists and validates, unseen-natural-domain generalization
remains unsupported.

## 8. Pre-result execution sequence

```bash
cd "${RRCL_ROOT}"
python3 scripts/inspect_fdst_layout.py \
  --root "${FDST_ROOT}"

# Review configs/fdst_scene_map_candidate.json and replace every
# REVIEW_REQUIRED value using official identity evidence.
python3 scripts/preflight_fdst.py \
  --root "${FDST_ROOT}" \
  --scene-map configs/fdst_scene_map_candidate.json
```

After reviewing and committing the generated config, split files, normalized
scene map, manifest, and implementation on a clean commit, the sole formal entry
is:

```bash
python3 scripts/run_fdst_c7.py
```

Do not run the formal entry before the split-freeze commit.
