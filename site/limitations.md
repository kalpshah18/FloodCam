---
title: Limitations
layout: default
nav_order: 8
description: "What these results do and do not show, and what was disclosed about the evaluation."
---

# Limitations
{: .no_toc }

1. TOC
{:toc}

{% assign n = site.data.numbers %}

## What the results support

On real USGS footage from one shallow-flow camera, a learned plate reader with a quality gate reads depth to about {{ n.maeDevGated }} cm (leave-one-month-out CV) on the frames it accepts, much better than a colour-threshold baseline, and it abstains on {{ n.dropDev }}% of frames. Its error stays within twice the native value up to about {{ n.resSF }} cm per pixel.

## What they do not show

{: .limit }
**Sample size.** Each camera gives an effective sample size of months, not frames. The river camera contributes one flood ({{ n.hancN }} gated frames). Stronger claims need more cameras and tape-measured depths.

{: .limit }
**Labels.** The label is a gauge beside the plate, not a tape measure at the plate. Water below the plate bottom ({{ n.plateBottomCm }} cm) is unobservable, and the reader reports 0 there. About {{ n.sigmaFloor }} cm of the error floor comes from the gauge-to-plate offset and calibration.

{: .limit }
**Transfer to other cameras.** Zero-shot transfer is mixed. The reader beats a no-water guess on two of the three added staff-plate cameras, fails on one (an orange plate it never saw), and does much worse on frames that actually contain water than on dry frames. Calibration on a first month does not fix that.

{: .limit }
**The state machine.** The Normal, Pending, Flood and Recovering logic in the dashboard, and the playground on the [demo page]({% link demo.md %}#state-machine), are demonstrations of the dashboard. They are not an evaluated claim: short clips cannot validate trends that unfold over minutes to hours.

## Disclosures about the evaluation

- **The locked months were looked at twice.** The two locked months ({{ n.lockedBlocks }}) were fixed before any model was trained. The first run used a one-sided sharpness gate. Development-month errors then showed that dirty-lens frames also fail, so the gate became two-sided and the locked run was repeated with an explicit override. Both runs are in `results/real/final_test_log.jsonl`, and the first-run metrics are kept in `results/real/final_v1_gate_one_sided_metrics.json`. The first run reached {{ n.maeLockedVOneGated }} cm at {{ n.covLockedVOne }}% coverage; the second reached {{ n.maeLockedGated }} cm at {{ n.covLocked }}%. The gate was therefore designed with development error in view.
- **Hancock.** The river-camera flood is locked and used for no template, offset or training.
- **Main-camera labels.** They come from two open-source projects' result tables, not from a fresh NWIS query, and were not re-checked.
- **The production model.** `models/in_overland/reader.joblib` was trained on every month, including the locked ones, so it is in-sample on this data. Use it for demos only. This website does not run it: the demo replays stored held-out estimates.
- **Scope.** The method assumes a fixed camera that sees a reference object of known scale. It is not tested on moving cameras or arbitrary video, and it is not a tool for safety decisions.

## AI-use disclosure

Claude Code (Anthropic) helped with brainstorming, code, debugging, experiments, the dashboard, this website and drafting. The pipeline was verified with unit tests and by inspecting strips, failures and split leakage. I am responsible for all results. The full list is in `AI_USE.md` in the repository.
