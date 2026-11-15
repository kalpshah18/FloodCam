---
title: Demo
layout: default
nav_order: 5
description: "Step through a real flood event, coarsen the plate to see what resolution does, and try the flood state machine. Runs entirely in your browser."
---

# Demo
{: .no_toc }

Everything here runs in your browser from small precomputed files; nothing is sent anywhere.

{% assign n = site.data.numbers %}

{: .note }
The depth readings in the first two panels are **held-out** estimates: each one comes from a model trained without that month (September 2026), so the reader had never seen these frames. The reading is not computed live in the page. The page replays stored results next to the official USGS gauge height.

<noscript><p class="noscript">This demo needs JavaScript. The figures on the <a href="{% link results.md %}">Results</a> page show the same data as static images.</p></noscript>

<div id="demo-root" data-base="{{ '/assets/demo/' | relative_url }}" data-sigma-v="{{ n.sigmaV }}" data-sigma-g="{{ n.sigmaFloor }}">

<div class="demo" id="explorer">
  <h3>1. Frame explorer: a real event, frame by frame</h3>
  <p class="note-small">Camera <code>in_overland</code>, 17&ndash;22 September 2026, every third frame. The yellow outline on the photo is the staff plate found by registration; the strip on the right is that plate rectified to the reader's input. Drag the slider or press play.</p>

  <div class="controls">
    <button class="btn btn-sm" id="ex-play" type="button">Play</button>
    <input type="range" id="ex-slider" min="0" max="0" value="0" aria-label="Frame">
    <span id="ex-time" class="note-small"></span>
  </div>

  <div class="demo-grid">
    <div>
      <div class="frame-box"><img id="ex-img" alt="Camera frame with the staff plate outlined"></div>
      <div class="readout">
        <div class="gauge">USGS gauge<b id="ex-gauge">&ndash;</b></div>
        <div class="reader">Reader (held out)<b id="ex-reader">&ndash;</b></div>
        <div>Colour-threshold baseline<b id="ex-base">&ndash;</b></div>
        <div>Reader error<b id="ex-err">&ndash;</b></div>
      </div>
      <p id="ex-status"></p>
    </div>
    <div>
      <div class="strip-box"><canvas id="ex-strip" aria-label="Rectified plate strip with gauge and reader levels"></canvas></div>
      <p class="note-small" id="ex-strip-note"></p>
    </div>
  </div>

  <canvas id="ex-timeline" aria-label="Gauge height and reader estimates over time"></canvas>
  <div class="legend">
    <span><i style="background:var(--fc-gauge)"></i>USGS gauge</span>
    <span><i style="background:var(--fc-reader)"></i>reader, frames that passed the gate</span>
    <span><i style="background:#b8bcc4"></i>reader, rejected by the gate (hollow)</span>
    <label><input type="checkbox" id="ex-base-toggle"> show colour-threshold baseline</label>
  </div>
  <p class="note-small">Lines on the strip use the plate's nominal geometry (ruler marks in cm). The reader also applies a fitted affine correction and the gauge sits beside the plate, so a line can sit a few cm from the visible waterline. Frames are the 640 px public copies, so the strip is softer than the 1280 px input the reader normally gets. Some frames are rejected on purpose: night, fog and a corrupted image are all in this event.</p>
</div>

<div class="demo" id="resolution">
  <h3>2. Resolution: what happens when each pixel covers more centimetres</h3>
  <p class="note-small">The strip of the frame selected above, coarsened by a factor f. The error table is from the study, where the reader was retrained at every factor.</p>
  <div class="controls">
    <input type="range" id="rs-slider" min="0" max="6" value="0" aria-label="Downsampling factor">
    <span id="rs-label" class="note-small"></span>
  </div>
  <div class="demo-grid">
    <div><canvas id="rs-chart" aria-label="Error against centimetres per pixel"></canvas>
      <div class="legend">
        <span><i style="background:var(--fc-reader)"></i>learned reader, MAE</span>
        <span><i style="background:#b8bcc4"></i>colour-threshold baseline, MAE</span>
        <span><i style="background:#2c5282"></i>error model, RMSE</span>
      </div>
      <div class="readout">
        <div>Scale<b id="rs-s">&ndash;</b></div>
        <div>Learned MAE<b id="rs-mae">&ndash;</b></div>
        <div>Learned RMSE (model)<b id="rs-rmse">&ndash;</b></div>
        <div>Baseline MAE<b id="rs-base">&ndash;</b></div>
      </div>
    </div>
    <div class="strip-box"><canvas id="rs-strip" aria-label="Plate strip at the chosen resolution"></canvas></div>
  </div>
  <p class="note-small">Illustration only: the strip is area-averaged in your browser from the native strip. The study downsampled the whole frame before re-sampling the plate, so it is close but not identical. The numbers come from <code>results/real/tables/resolution.csv</code>. The error model is \(\mathrm{RMSE}^2 = s^2\sigma_v^2 + \sigma_g^2\) with \(\sigma_v = {{ n.sigmaV }}\) px and \(\sigma_g = {{ n.sigmaFloor }}\) cm.</p>
</div>

<div class="demo" id="state-machine">
  <h3>3. State machine playground (dashboard logic, not an evaluated result)</h3>
  <p class="note-small">A JavaScript port of the dashboard's rolling-median filter and four-state machine (Normal, Pending, Flood, Recovering) with a 10.5 cm trigger and a 9.5 cm clear level. Shape a synthetic flood and watch the state change. There is no recorded data here.</p>
  <div>
    <button class="btn btn-sm" type="button" data-preset="typical">Typical event</button>
    <button class="btn btn-sm" type="button" data-preset="spike">Short spike (not a flood)</button>
    <button class="btn btn-sm" type="button" data-preset="edge">Hovering near the trigger</button>
    <button class="btn btn-sm" type="button" id="sm-reseed">New noise</button>
  </div>
  <div class="sm-grid">
    <div><label>Peak depth: <b id="sm-peak-v"></b> cm</label><input type="range" id="sm-peak" min="0" max="60" step="0.5"></div>
    <div><label>Rise time: <b id="sm-rise-v"></b> min</label><input type="range" id="sm-rise" min="1" max="30" step="1"></div>
    <div><label>Time at peak: <b id="sm-hold-v"></b> min</label><input type="range" id="sm-hold" min="0" max="30" step="1"></div>
    <div><label>Fall time: <b id="sm-fall-v"></b> min</label><input type="range" id="sm-fall" min="1" max="40" step="1"></div>
    <div><label>Measurement noise: <b id="sm-noise-v"></b> cm</label><input type="range" id="sm-noise" min="0" max="3" step="0.1"></div>
    <div><label>Persistence: <b id="sm-persist-v"></b> min</label><input type="range" id="sm-persist" min="1" max="15" step="1"></div>
  </div>
  <canvas id="sm-chart" aria-label="Depth, filtered depth and flood state over time"></canvas>
  <div class="controls">
    <input type="range" id="sm-scrub" min="0" max="100" value="0" aria-label="Time">
    <span>State: <span class="state-pill" id="sm-state">Normal</span></span>
    <span class="note-small" id="sm-now"></span>
  </div>
  <div class="legend" id="sm-legend"></div>
  <div id="sm-log" aria-live="polite"></div>
  <p class="note-small">The Recovering state is a dashboard demonstration. Short clips cannot validate trends that unfold over minutes to hours, so draining versus accumulating is not an evaluated claim of this project.</p>
</div>

</div>

<script src="{{ '/assets/js/demo.js' | relative_url }}" defer></script>
