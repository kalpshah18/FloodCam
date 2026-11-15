---
title: Home
layout: default
nav_order: 1
permalink: /
description: "Estimating shallow flood depth from a fixed camera that sees a staff plate, evaluated on real USGS gauge cameras."
---

{% include hero.html %}
{% assign n = site.data.numbers %}
{% assign r = site.data.results %}

<figure>
  <img src="{{ '/assets/img/fig_pipeline.png' | relative_url }}" alt="The reader on one held-out frame: frame with plate outline, rectified strip, row probabilities and decoded level, and error against image sharpness">
  <figcaption>The reader on one held-out frame. (a) Frame and plate outline. (b) Rectified strip. (c) Per-row probability of being submerged, decoded level (red) and gauge height (dashed). (d) Cross-validated error against image sharpness; green points pass the quality gate.</figcaption>
</figure>

## Abstract

How accurately can a fixed camera estimate flood water depth, and how well does that transfer across cameras? I use real USGS staff-plate cameras whose official gauge heights serve as labels. A learned reader registers each frame, rectifies the plate, classifies plate rows as submerged and decodes the water level. An image-quality gate abstains on bad frames.

On the main shallow-flow camera ({{ n.nFrames }} frames, {{ n.nBlocks }} months), leaving out whole months gives **{{ n.maeDevGated }} cm** mean absolute error on the {{ n.covDev }}% of frames passing the gate ({{ n.maeBaseDevGated }} cm for a colour-threshold baseline), and **{{ n.maeLockedGated }} cm** on two locked months. Error stays below twice its native value up to {{ n.resSF }} cm per pixel.

Applied unchanged to three cameras downloaded afterwards, with each camera held out of training in turn, the reader beats a no-water guess on two cameras and fails on one. A river camera is read to {{ n.hancMaeCal }} cm over a {{ n.hancRangeCm }} cm flood.

## Headline numbers

Mean absolute error in cm on frames that pass the quality gate. Every value on this page is generated from the files in `results/` (see [Reproduce]({% link reproduce.md %})).

<div class="table-wrap" markdown="0">
<table>
  <thead>
    <tr><th>Evaluation</th><th>Frames</th><th class="num">Learned reader</th><th class="num">Colour-threshold baseline</th><th class="num">Always 0 cm</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>Leave-one-month-out CV ({{ n.nDevBlocks }} months)</td>
      <td class="num">{{ r.cv.gated.learned.n }}</td>
      <td class="num"><b>{{ r.cv.gated.learned.mae | round: 2 }}</b> <small>[{{ r.cv.gated.learned.ci_lo | round: 2 }}, {{ r.cv.gated.learned.ci_hi | round: 2 }}]</small></td>
      <td class="num">{{ r.cv.gated.threshold_baseline.mae | round: 2 }}</td>
      <td class="num">{{ r.cv.gated.always_no_flood.mae | round: 2 }}</td>
    </tr>
    <tr>
      <td>Locked test months ({{ n.lockedBlocks }})</td>
      <td class="num">{{ r.locked.gated.learned.n }}</td>
      <td class="num"><b>{{ r.locked.gated.learned.mae | round: 2 }}</b></td>
      <td class="num">{{ r.locked.gated.threshold_baseline.mae | round: 2 }}</td>
      <td class="num">{{ r.locked.gated.always_no_flood.mae | round: 2 }}</td>
    </tr>
  </tbody>
</table>
</div>

Brackets are 95% intervals from a bootstrap over whole months. The two locked months are too few to give an interval. The gate drops {{ n.dropDev }}% of CV frames and {{ n.dropLocked }}% of locked frames; without it the errors are {{ n.maeDevAll }} cm (CV) and {{ n.maeLockedAll }} cm (locked). See [Results]({% link results.md %}) for the full breakdown.

{: .limit }
The effective sample size is the number of months per camera, not the number of frames. Labels are a gauge beside the plate, not tape measurements, and water below the plate bottom ({{ n.plateBottomCm }} cm) cannot be seen. The locked months were evaluated twice, and the reason is disclosed in [Limitations]({% link limitations.md %}).

## Where to go next

| | |
|---|---|
| [Method]({% link method.md %}) | Registration, plate rectification, the row classifier, the decode step and the error model |
| [Data]({% link data.md %}) | The USGS cameras, where the gauge labels come from, and how the splits keep months apart |
| [Results]({% link results.md %}) | Held-out and locked-test errors, resolution sweep, other cameras, failure gallery |
| [Demo]({% link demo.md %}) | Step through a real flood event frame by frame, in your browser |
| [Paper]({% link paper.md %}) | The four-page PDF and a BibTeX entry |
| [Reproduce]({% link reproduce.md %}) | Setup, make targets, and which script produces which result |
| [Limitations]({% link limitations.md %}) | What these results do and do not show |
