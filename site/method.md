---
title: Method
layout: default
nav_order: 2
description: "How a frame becomes a depth: registration, plate rectification, a row classifier, a monotone decode and a quality gate."
---

# Method
{: .no_toc }

1. TOC
{:toc}

{% assign n = site.data.numbers %}

## The question

Given one frame from a fixed camera that sees a reference object of known scale, estimate the water depth \(h\) (cm) at the object. Any flood response (a severity class, a warning level) is a function of \(h\), so the target is \(h\) itself, and error is reported by depth range.

One row of data is one frame. Prediction uses the current frame, a one-time setup (reference views and plate geometry) and, for pose recovery only, earlier frames. The gauge is never used at prediction time.

<figure>
  <img src="{{ '/assets/img/fig_pipeline.png' | relative_url }}" alt="Pipeline on one held-out frame">
  <figcaption>One held-out frame through the reader. Try it on real frames in the <a href="{% link demo.md %}">demo</a>.</figcaption>
</figure>

## 1. Register the frame

The camera pose changes between months, so the plate is not at a fixed pixel position. Each frame is matched with SIFT keypoints (after contrast equalisation, which helps infrared and washed-out frames) against a set of reference views of the same camera, and a homography to the main reference is fitted with RANSAC. Registration succeeds on {{ n.pctRegistered }}% of frames. When it fails, the reader reuses the last good pose from an earlier frame (causal, never a later one); {{ n.pctBorrowed }}% of frames use a borrowed pose. The reference views come from development months and use geometry only, not labels.

## 2. Rectify the plate

The staff plate is a flat rectangle, so its outline in the reference view maps to an axis-aligned strip by a homography. Every frame is resized to 1280 px wide and the plate is resampled to a 360 &times; 48 pixel strip, which is about 0.16 cm per row. Row 0 is the top of the plate, and rows are linear in height.

## 3. Classify each row

For frame \(i\) and strip row \(k\) (height \(h_k\)), a gradient-boosted classifier (scikit-learn) outputs

$$p_{ik} = P(\text{row } k \text{ is submerged})$$

from row colour, change against a dry template, texture, context from neighbouring rows and frame-level quality. The labels need no hand annotation:

$$y_{ik} = \mathbf{1}[h_k < g_i]$$

where \(g_i\) is the official gauge height at the time of frame \(i\).

## 4. Decode the water level

Water is below some boundary and air above it, so the level is the single monotone change point that best explains the row probabilities:

$$b^\star = \arg\max_b \; \sum_{k<b} \log(1 - p_{ik}) + \sum_{k \ge b} \log p_{ik}$$

The boundary is mapped to a height, then to centimetres by a least-squares affine fit on training frames. Levels at or below the plate bottom ({{ n.plateBottomCm }} cm) are reported as 0, because water below the plate cannot be seen. The log-likelihood gap to the runner-up boundary is kept as a confidence score, the *margin*.

## 5. Quality gate

A frame is rejected when its Laplacian variance (a sharpness measure) falls outside \([{{ n.gateLo }}, {{ n.gateHi }}]\) times the training median. A low value means fog, blur or darkness. A high value means dirt on the lens or glare. Rejected frames get no reading; they are logged and the reader abstains.

The gate was chosen from development-month errors only. It started one-sided and became two-sided after the dev errors showed that dirty-lens frames also fail; that change is part of the locked-test disclosure on the [Limitations]({% link limitations.md %}) page.

## Baselines

- **B0, always 0 cm.** The trivial guess of no water.
- **B1, colour threshold.** A row counts as wet if its colour change from the dry template exceeds a threshold \(\tau\) tuned on training months, followed by the same decode and calibration.

## Error model

Depth error should come from how well the waterline can be located in pixels, scaled by how many centimetres one pixel spans, plus a floor from the gauge-to-plate offset and calibration. With scale \(s\) (cm per pixel) and waterline error \(\sigma_v\) (pixels):

$$\mathrm{Var}(\hat h) = s^2 \sigma_v^2 + \sigma_g^2$$

Fitting RMSE² to the resolution sweep gives \(\sigma_v = {{ n.sigmaV }}\) px and \(\sigma_g = {{ n.sigmaFloor }}\) cm. The floor dominates at fine scales, so at this camera's native resolution the binding constraint is the gauge-to-plate reference, not pixels. The [Results]({% link results.md %}) page shows the sweep, and the [demo]({% link demo.md %}) lets you coarsen a strip yourself.

## Cross-camera protocol

For each target camera the classifier is trained on the *other* cameras only, with labels from their nominal plate geometry. The target's dry template and gate thresholds come from its earliest month. *Zero-shot* uses the nominal geometry. *Calibrated* fits an affine map on that first month, and later months are evaluated. For the river camera (Hancock), one level offset is fitted from {{ n.hancCalN }} development frames, and a *pooled* variant adds those frames to training.
