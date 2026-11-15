---
title: Results
layout: default
nav_order: 4
description: "Held-out and locked-test depth error, the resolution sweep, transfer to other cameras and a failure audit."
---

# Results
{: .no_toc }

1. TOC
{:toc}

{% assign n = site.data.numbers %}
{% assign r = site.data.results %}

All numbers on this page are rendered from `results/real/` by `scripts/build_site_assets.py`. Errors are in cm. Depth ranges are by gauge height: dry (up to 0.5 cm), 0.5&ndash;10 cm and above 10 cm.

## Main result

Mean absolute error (MAE) is the primary metric. Intervals are 95% bootstrap intervals over whole months, so they reflect month-to-month spread rather than frame count. The two locked months give no interval.

<div class="table-wrap" markdown="0">
<table>
  <thead>
    <tr><th>Evaluation</th><th>Frames</th><th>Method</th><th class="num">n</th><th class="num">MAE [95% CI]</th><th class="num">RMSE</th><th class="num">Bias</th><th class="num">Dry</th><th class="num">0.5&ndash;10</th><th class="num">&gt;10</th></tr>
  </thead>
  <tbody>
  {% assign sets = "cv,locked" | split: "," %}
  {% for key in sets %}
    {% assign blk = r[key] %}
    {% if key == "cv" %}{% assign label = "Leave-one-month-out CV" %}{% else %}{% assign label = "Locked test" %}{% endif %}
    {% assign rows = "gated:learned,gated:threshold_baseline,all_frames:learned" | split: "," %}
    {% for row in rows %}
      {% assign parts = row | split: ":" %}
      {% assign scope = parts[0] %}{% assign method = parts[1] %}
      {% assign m = blk[scope][method] %}
      <tr>
        <td>{% if forloop.first %}{{ label }}{% endif %}</td>
        <td>{% if scope == "gated" %}gated{% else %}all{% endif %}</td>
        <td>{% if method == "learned" %}learned reader{% else %}threshold baseline{% endif %}</td>
        <td class="num">{{ m.n }}</td>
        <td class="num">{{ m.mae | round: 2 }}{% if key == "cv" %} [{{ m.ci_lo | round: 1 }}, {{ m.ci_hi | round: 1 }}]{% endif %}</td>
        <td class="num">{{ m.rmse | round: 1 }}</td>
        <td class="num">{% if m.bias >= 0 %}+{% endif %}{{ m.bias | round: 1 }}</td>
        <td class="num">{{ m.bins["0-0.5"] | round: 1 }}</td>
        <td class="num">{{ m.bins["0.5-10"] | round: 1 }}</td>
        <td class="num">{{ m.gt10 | round: 1 }}</td>
      </tr>
    {% endfor %}
  {% endfor %}
  </tbody>
</table>
</div>

Gated means the frame passed the sharpness gate (coverage {{ n.covDev }}% in CV, {{ n.covLocked }}% locked).

- On gated CV frames the learned reader has {{ n.maeDevGated }} cm MAE (95% CI {{ n.ciDevLo }}&ndash;{{ n.ciDevHi }}) against {{ n.maeBaseDevGated }} cm for the threshold baseline and {{ n.maeZeroDev }} cm for always-0.
- On the two locked months it reaches {{ n.maeLockedGated }} cm (baseline {{ n.maeBaseLockedGated }}, always-0 {{ n.maeZeroLocked }}).
- Without the gate the errors are {{ n.maeDevAll }} cm (CV) and {{ n.maeLockedAll }} cm (locked). The gate buys reliability by abstaining on {{ n.dropDev }}% of CV and {{ n.dropLocked }}% of locked frames.
- Per-month MAE has median {{ n.blockMedianDev }} cm and worst {{ n.blockWorstDev }} cm.
- A causal five-reading median makes things worse ({{ n.maeRawDev }} to {{ n.maeSmoothDev }} cm), because levels move tens of centimetres per hour.

<figure>
  <img src="{{ '/assets/img/fig_main.png' | relative_url }}" alt="Held-out predictions: one event, camera versus gauge, river camera, and error versus plate scale">
  <figcaption>Held-out predictions. (a) One event; ticks mark rejected frames. (b) Camera against gauge, coloured by month. (c) River camera, locked flood. (d) CV error against plate scale; dashed line is the error model.</figcaption>
</figure>

### Error by month

Gated MAE for each held-out month (leave-one-month-out CV). Bars are scaled to the largest month.

{% assign maxmae = 0.0 %}
{% for b in r.cv.blocks %}{% if b.mae > maxmae %}{% assign maxmae = b.mae %}{% endif %}{% endfor %}
<div class="table-wrap" markdown="0">
<table>
  <thead><tr><th>Month</th><th class="num">Frames</th><th class="num">Largest gauge (cm)</th><th class="num">MAE (cm)</th><th style="width:35%"></th></tr></thead>
  <tbody>
  {% for b in r.cv.blocks %}
    {% assign w = b.mae | times: 100.0 | divided_by: maxmae | round %}
    <tr><td>{{ b.block }}</td><td class="num">{{ b.n }}</td><td class="num">{{ b.max_gauge | round: 1 }}</td><td class="num">{{ b.mae | round: 2 }}</td>
        <td><div style="background:var(--fc-accent);height:.7rem;width:{{ w }}%;border-radius:2px"></div></td></tr>
  {% endfor %}
  </tbody>
</table>
</div>

## Resolution sweep

Registration is computed once at native resolution. Frames are then area-downsampled by a factor $$f$$ from 1 to 64, the plate is resampled with 360/$$f$$ rows, and the reader is retrained for each $$f$$. Plate scale is $$s = {{ n.sNative }} f$$ cm per pixel.

<div class="table-wrap" markdown="0">
<table>
  <thead><tr><th class="num">Downsample f</th><th class="num">cm per pixel</th><th class="num">Learned MAE</th><th class="num">Learned RMSE</th><th class="num">Baseline MAE</th></tr></thead>
  <tbody>
  {% for s in site.data.resolution %}
    <tr><td class="num">{{ s.factor }}</td><td class="num">{{ s.cm_per_px | round: 2 }}</td><td class="num">{{ s.mae }}</td><td class="num">{{ s.rmse }}</td><td class="num">{{ s.baseline_mae }}</td></tr>
  {% endfor %}
  </tbody>
</table>
</div>

Gated MAE is {{ n.resMaeA }} cm at native scale, {{ n.resMaeE }} cm at {{ n.resSE }} cm/px, {{ n.resMaeF }} cm at {{ n.resSF }} cm/px and {{ n.resMaeG }} cm at {{ n.resSG }} cm/px, which is over twice native. With scale $$s$$ and waterline error $$\sigma_v$$, $$\mathrm{Var}(\hat h) = s^2\sigma_v^2 + \sigma_g^2$$. Fitting RMSE² to the gated sweep gives $$\sigma_v = {{ n.sigmaV }}$$ px and $$\sigma_g = {{ n.sigmaFloor }}$$ cm. The $$\sigma_g$$ floor dominates at fine scales, so resolution is not the binding constraint at this camera.

You can see what coarser scales do to the plate in the [demo]({% link demo.md %}#resolution).

Ranking frames by decode margin instead of the sharpness gate gives {{ n.marginMaeAtCov }} cm at the gate's coverage (the gate gives {{ n.maePassGate }} cm), so a cheap image statistic abstains better than the classifier's own confidence.

## Other cameras

Table values are gated MAE in cm. "/ 0" is the no-water guess on the same frames. The three added cameras and the river camera were not used to build the method.

<div class="table-wrap" markdown="0">
<table>
  <thead><tr><th>Camera</th><th>Scene</th><th class="num">Largest gauge</th><th class="num">In-site / 0</th><th class="num">Held-out zero-shot / 0</th><th class="num">Held-out calibrated</th><th class="num">Gate coverage (held-out, %)</th></tr></thead>
  <tbody>
  {% for c in r.cameras %}
    <tr><td><code>{{ c.camera }}</code></td><td>{{ c.plate }}</td><td class="num">{{ c.max_gauge }}</td>
        <td class="num">{{ c.in_site }} / {{ c.in_site_zero }}</td>
        <td class="num">{{ c.heldout }} / {{ c.heldout_trivial }}</td>
        <td class="num">{{ c.heldout_calibrated }}</td><td class="num">{{ c.coverage }}</td></tr>
  {% endfor %}
    <tr><td><code>hancock</code></td><td>river plate</td><td class="num">{{ r.hancock.range_cm }}&dagger;</td><td class="num">&ndash;</td>
        <td class="num">{{ r.hancock.zero_shot }} / {{ r.hancock.plate_bottom_guess }}</td><td class="num">{{ r.hancock.calibrated }}</td><td class="num">&ndash;</td></tr>
  </tbody>
</table>
</div>

&dagger; Hancock: the flood range above the plate bottom. Its trivial guess is the plate bottom.

- **In-site** (leave-one-month-out), the reader beats always-0 on `in_overland`, `in_north` ({{ n.camBInMae }} vs {{ n.camBInZero }} cm) and `oh_wetland` ({{ n.camCInMae }} vs {{ n.camCInZero }} cm). `wi_mt1` stays nearly dry (gauge up to {{ n.camDMax }} cm), and always-0 wins there.
- **With the camera held out, zero-shot,** the reader beats always-0 on `in_overland` ({{ n.camALocoZero }} vs {{ n.camALocoTriv }} cm, but only {{ n.camACov }}% of frames pass the gate), `in_north` ({{ n.camBLocoZero }} vs {{ n.camBLocoTriv }}) and `oh_wetland` ({{ n.camCLocoZero }} vs {{ n.camCLocoTriv }}).
- It **fails on `wi_mt1`** ({{ n.camDLocoZero }} vs {{ n.camDLocoTriv }}). Having never seen an orange plate, the classifier reads it as fully submerged in sunny frames (69 cm at gauge 0).
- On frames with water (gauge above 0.5 cm) its error is much larger ({{ n.camBWetMae }} and {{ n.camCWetMae }} cm against {{ n.camBWetZero }} and {{ n.camCWetZero }} cm for the no-water guess), and calibration does not help.
- **River plate.** Zero-shot MAE over the {{ n.hancRangeCm }} cm flood is {{ n.hancMaeZero }} cm (plate-bottom guess {{ n.hancTrivial }}), and {{ n.hancMaeCal }} cm calibrated. Pooling helps inside the stage range it saw ({{ n.hancMaePoolA }}, {{ n.hancMaePoolB }} against {{ n.hancMaeCalA }}, {{ n.hancMaeCalB }} cm at 11&ndash;16 ft) and hurts beyond it ({{ n.hancMaePoolC }} against {{ n.hancMaeCalC }} cm at 16&ndash;19 ft).

<figure>
  <img src="{{ '/assets/img/fig_frames.png' | relative_url }}" alt="Real frames: median-error examples, largest accepted errors, rejected frames, and cameras held out of training">
  <figcaption>Real frames (crop and plate strip). Row 1: median-error examples, and Hancock. Row 2: largest accepted errors and rejected frames. Row 3: cameras held out of training, zero-shot; the last one is the <code>wi_mt1</code> failure.</figcaption>
</figure>

## Failure audit

Infrared is not worse than daylight ({{ n.maeIR }} vs {{ n.maeDay }} cm MAE). The costly cases are a borrowed pose ({{ n.maeBorrowed }} vs {{ n.maeRegistered }} cm for a registered pose) and frames outside the gate ({{ n.maeFailGate }} vs {{ n.maePassGate }} cm). In the worst accepted day frame, glare washes out the plate and water streaks on the panel look like wet rows.

The overlays below are saved by `scripts/real_report.py`. Each shows the frame, the plate strip and the reading. Captions give the gauge height and the reader's estimate.

### Worst accepted frames

<div class="gallery">
{% for f in site.data.failures %}{% if f.kind == "worst_accepted" %}
  <figure><a href="{{ '/assets/failures/' | append: f.overlay | relative_url }}"><img loading="lazy" src="{{ '/assets/failures/' | append: f.overlay | relative_url }}" alt="Worst accepted frame, gauge {{ f.gauge_cm }} cm, estimate {{ f.est_cm }} cm"></a>
  <figcaption>gauge {{ f.gauge_cm }} cm, reader {{ f.est_cm }} cm{% if f.is_ir %}, infrared{% endif %}{% if f.borrowed %}, borrowed pose{% endif %}</figcaption></figure>
{% endif %}{% endfor %}
</div>

### Rejected frames

<div class="gallery">
{% for f in site.data.failures %}{% if f.kind == "rejected" %}
  <figure><a href="{{ '/assets/failures/' | append: f.overlay | relative_url }}"><img loading="lazy" src="{{ '/assets/failures/' | append: f.overlay | relative_url }}" alt="Rejected frame, gauge {{ f.gauge_cm }} cm"></a>
  <figcaption>gauge {{ f.gauge_cm }} cm, would have read {{ f.est_cm }} cm{% if f.is_ir %}, infrared{% endif %}{% if f.borrowed %}, borrowed pose{% endif %}</figcaption></figure>
{% endif %}{% endfor %}
</div>

## More figures

<figure>
  <img src="{{ '/assets/img/res_risk_coverage_cv.png' | relative_url }}" alt="Error against coverage when frames are ranked by confidence">
  <figcaption>Error against coverage in CV: how MAE changes as the least trustworthy frames are dropped.</figcaption>
</figure>

<figure>
  <img src="{{ '/assets/img/res_timeseries_2026-09.png' | relative_url }}" alt="Time series of gauge and reader for September 2026">
  <figcaption>Held-out time series for September 2026, the event used in the <a href="{% link demo.md %}">demo</a>.</figcaption>
</figure>
